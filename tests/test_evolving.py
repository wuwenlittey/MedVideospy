import importlib
import json
import sys
import threading
import time
import types
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

try:
    import litellm
except ModuleNotFoundError:
    stub = types.ModuleType("litellm")
    stub.completion = lambda **kwargs: None
    sys.modules["litellm"] = stub

from videospy.evolving.cli import main, parse_args
from videospy.evolving.reflector.runner import classify_by_task, select_groups, ReflectionRunner
from videospy.evolving.reflector.evidence import build_evidence
from videospy.evolving.reflector import Reflector
from videospy.evolving.reflector.model import validate_assessment, validate_memory
from videospy.evolving.pipeline import EvolutionPipeline
from videospy.evolving.storage.memory_store import EvolutionStateStore
from videospy.evolving.optimizer import MemoryOptimizer
from videospy.evolving.optimizer.skill_optimizer import validate_task_document, validate_domain_document
from videospy.evolving.common.schemas import ReflectionReport, EvolutionUpdate
from videospy.evolving.common.memory_scope import source_context
from videospy.evolved_config import apply_evolution_state

medvid_splits = importlib.import_module("experiments.prepare_medvidbench_split")


def base_config():
    return {
        "SYSTEM_PROMPT": "Answer based on video evidence.", "TASK_SKILLS": {},
        "agent": {"model": {"model_name": "test-model", "seed": 42, "max_completion_tokens": 1000}, "max_steps": 10},
        "tools": {"enabled": ["overview", "clip_skim", "frame_inspect"],
                  "overview": {"num_frames": 4}, "clip_skim": {"num_frames": 4}, "frame_inspect": {"max_frames": 2},
                  "memory_search": {"enabled": False, "embedding_model": None, "allow_hash_embedding": True,
                                    "top_k": 3, "similarity_threshold": 0.0}},
        "evolving": {"memory_dedup_threshold": .95,
                     "reflector": {"sampling": {"max_samples": 0, "group_size": 2, "bad_ratio": .5, "seed": 42}},
                     "inference": {"skill": True, "memory": True, "hyperparameter": False}}}


def write_source(root, benchmark="lvbench", count=8):
    root.mkdir(parents=True, exist_ok=True)
    (root / "trajectories").mkdir(exist_ok=True)
    records=[]
    for i in range(count):
        sid=str(i)
        observation=f"At 1.0s person {i} opens the red door."
        trajectory={"question": f"Question {i}", "final_answer": "A", "total_steps": 2,
                    "messages": [{"role": "assistant", "tool_calls": [{"id": "tool1", "function": {
                        "name": "frame_inspect", "arguments": '{"timestamps": [1.0]}'}}]},
                                 {"role": "tool", "name": "frame_inspect", "tool_call_id": "tool1", "content": observation}]}
        (root / "trajectories" / f"{sid}.json").write_text(json.dumps(trajectory))
        record = {"sample_key": sid, "status": "success", "prediction": "A",
                  "trajectory": f"trajectories/{sid}.json"}
        if benchmark == "lvbench":
            record.update(video_id=f"video{i}", answer="A", question_types=["retrieval", "reasoning"])
        elif benchmark == "egolife":
            record.update(question=f"Question {i}", answer="A", category="EntityLog" if i % 2 else "EventLog",
                          query_time={"date": "DAY1", "time": f"09{i:02d}0000"})
        else:
            record.update(video_id=f"video{i}", ground_truth="A", qa_type="next_action" if i % 2 else "dense_captioning_v1")
        records.append(record)
    run = {"dataset": benchmark, "agent": "videospy", "mode": "full",
           "sample_keys": [str(i) for i in range(count)], "agent_config": base_config()}
    if benchmark == "egolife":
        run["video_root"] = "/recorded/A1_JAKE"
    (root / "run.json").write_text(json.dumps(run))
    (root / "records.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in records))
    return records


class FakeReflector:
    model_config = {"model_name": "test-model"}
    model_source = "agent"

    def __init__(self, failures=()):
        self.failures=set(failures)
        self.active=self.peak=0
        self.calls=[]
        self.lock=threading.Lock()

    def enter(self, stage, key):
        with self.lock:
            self.active+=1
            self.peak=max(self.peak,self.active)
            self.calls.append((stage,key))
        time.sleep(.01)
        with self.lock:
            self.active-=1
        if (stage,key) in self.failures:
            raise ValueError("injected isolated failure")

    def classify(self, item):
        self.enter("classify", item["sample_id"])
        return {"reason": "Evidence support, reasoning and efficiency", "answer_status": item.get("answer_status", "correct"), "evidence_support": "sufficient",
                "score": 4 if item.get("answer_status", "correct") == "correct" else 2, "reasoning_sound": True}

    def reasoning_case(self, item):
        return None

    def update_skill(self, task, group, current):
        self.enter('skill', task)
        refs = {s['sample_id']: {'sample_id': s['sample_id'], 'message_indices': s['message_indices']}
                for s in (current or {}).get('sources', [])}
        refs.update({i['sample_id']: {'sample_id': i['sample_id'], 'message_indices': [1]} for i in group})
        return validate_task_document({'instruction': 'Inspect decisive observations for ' + task + '.',
                                       'sources': list(refs.values())}, group, current)

    def domain_skill(self, benchmark, skills):
        self.enter('domain', benchmark)
        return validate_domain_document({'instruction': 'Preserve person and temporal scope.',
                                         'source_tasks': list(skills)}, skills)

    def review_skill(self, task, document, task_skills=None):
        self.enter('review', task)
        return {'approved': True, 'feedback': 'The methods are grounded in the supplied evidence.'}

    def revise_skill(self, task, document, feedback, task_skills=None):
        self.enter('revision', task)
        return document

    def memory(self, item, evidence):
        sid=item["sample_id"]
        self.enter("memory",sid)
        quote=evidence["messages"][0]["content"]
        return {"context": f"What did person {sid} do?", "content": quote, "event": "opening a door",
                "time": {"start_seconds": 1., "end_seconds": 1., "label": ""},
                "evidence_refs": [{"tool_call_id": "tool1", "quote": quote}]}


class ClassificationTest(unittest.TestCase):
    def test_assessment_validates_labels_and_requests_process_evaluation(self):
        for invalid in ('unknown', True, None):
            with self.assertRaises(ValueError):
                validate_assessment({'answer_status': invalid, 'evidence_support': 'sufficient', 'reason': 'test'})
        model = Mock()
        model.chat.side_effect = [{'content': '{}'}, {'content': json.dumps({
            'reason': 'Correct guess without support', 'answer_status': 'correct', 'evidence_support': 'insufficient', 'score': 2, 'reasoning_sound': False})}]
        result = Reflector(base_config(), model).classify({'prediction': 'A', 'reference_answer': 'A', 'trajectory': {}})
        self.assertEqual(result['evidence_support'], 'insufficient')
        self.assertEqual(result['score'], 2)
        self.assertEqual(model.chat.call_count, 2)

    def test_outcomes_are_absolute_and_consistent_across_task_types(self):
        items = [{'sample_id': str(i), 'task_types': ['x', 'y']} for i in range(5)]
        statuses = [('correct', 'sufficient'), ('incorrect', 'sufficient'),
                    ('correct', 'partial'), ('unverified', 'sufficient')]
        assessments = {str(i): {'answer_status': a, 'evidence_support': e, 'reason': 'test'}
                       for i, (a, e) in enumerate(statuses)}
        result = classify_by_task(items, assessments)
        self.assertEqual([r['outcome'] for r in result['samples']],
                         [v for v in ('success', 'problem', 'problem', 'unverified', 'failed') for _ in range(2)])

    def test_group_ratio_and_unique_sources(self):
        items = [{'sample_id': str(i), 'task_types': ['x']} for i in range(10)]
        classification = {'samples': [{'sample_id': str(i), 'task_type': 'x',
                                       'outcome': 'problem' if i < 7 else 'success', 'score': 1 if i < 7 else 5} for i in range(10)]}
        groups = select_groups(items, classification, 4, 0, 42)['x']
        self.assertEqual([len(g) for g in groups], [4, 4, 2])
        self.assertEqual({i['score'] for i in groups[0]}, {1, 5})
        self.assertEqual(len({i['sample_id'] for g in groups for i in g}), 10)


class OfflinePipelineTest(unittest.TestCase):
    def test_directory_name_uses_source_and_reflector_models_not_base_agent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            write_source(source, count=1)
            manifest = json.loads((source / 'run.json').read_text())
            manifest['agent_config']['agent']['model']['model_name'] = 'openai/Qwen3.8-27B'
            (source / 'run.json').write_text(json.dumps(manifest))
            cfg = base_config()
            cfg['agent']['model']['model_name'] = 'openai/Qwen3.5-4B'
            reflector = FakeReflector()
            for model in ['openai/Qwen3.8-27B', 'openai/other-reflector']:
                reflector.model_config = {'model_name': model}
                pipeline = EvolutionPipeline('lvbench', root / 'train.json', root / 'out', cfg,
                                             source, reflector=reflector)
                self.assertTrue(pipeline.run_id.startswith(
                    'source_openai_Qwen3.8-27B_reflector_' + model.replace('/', '_') + '_'))
                self.assertNotIn('Qwen3.5-4B', pipeline.run_id)
                self.assertFalse(pipeline.run_dir.exists())
            # Explicit identities and resume paths do not read or rename the source.
            for kwargs, expected in [({'run_id': 'existing-id'}, root / 'out' / 'existing-id'),
                                     ({'run_dir': root / 'existing-run'}, root / 'existing-run')]:
                pipeline = EvolutionPipeline('lvbench', root / 'train.json', root / 'out', cfg,
                                             root / 'unavailable-source', reflector=reflector, **kwargs)
                self.assertEqual(pipeline.run_dir, expected.resolve())

    def test_parallel_routes_single_task_domain_memory_provenance_and_resume(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source'
            records=write_source(source)
            before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            manifest=root/'train.json';manifest.write_text(json.dumps(list(range(8))))
            reflector=FakeReflector(failures={('memory','2')})
            pipeline=EvolutionPipeline('lvbench',manifest,root/'output',base_config(),source,reflector=reflector,workers=3,run_dir=root/'run')
            run=pipeline.run()
            self.assertGreater(reflector.peak,1);self.assertLessEqual(reflector.peak,3)
            self.assertEqual(sum(stage=='classify' for stage,_ in reflector.calls),8)
            self.assertEqual(sum(stage=='memory' for stage,_ in reflector.calls),8)
            skills=json.loads((run/'skills.json').read_text())
            self.assertEqual(set(skills),{'retrieval','reasoning','__domain__'})
            self.assertTrue(all(v['format']=='document' for v in skills.values()))
            self.assertEqual(len(skills['retrieval']['sources']),8)
            memories=json.loads((run/'memories.json').read_text())
            self.assertEqual(len(memories),7)
            self.assertEqual(len({m['source_sample_id'] for m in memories}),7)
            self.assertTrue(all(m['metadata']['source']['video'] and m['metadata']['time']['start_seconds']==1 for m in memories))
            self.assertTrue(all(m['metadata']['evidence_refs'] for m in memories))
            store=EvolutionStateStore(run/'evolution.sqlite3', read_only=True)
            self.assertEqual(store.count_memories(store.resolve_version()),7)
            self.assertEqual(store.load_skills(store.resolve_version()), skills)
            self.assertEqual(sum(stage=='skill' and task=='retrieval' for stage,task in reflector.calls), 4)
            matches=store.search(store.resolve_version(),'opens red door',None,2,-1,
                                 context=source_context('lvbench', 'video0'))
            self.assertEqual(len(matches),1)
            self.assertTrue(all(m['metadata']['event']=='opening a door' for m in matches))
            loaded=apply_evolution_state(base_config(),run/'evolution.sqlite3')
            self.assertIn('__domain__',loaded['TASK_SKILLS'])
            for skill,memory in ((True,False),(False,True),(False,False)):
                cfg=base_config();cfg['evolving']['inference'].update(skill=skill,memory=memory)
                configured=apply_evolution_state(cfg,run/'evolution.sqlite3')
                self.assertEqual(bool(configured['TASK_SKILLS']),skill)
                self.assertEqual(configured['tools']['memory_search']['enabled'],memory)
            calls=len(reflector.calls);pipeline.run();self.assertEqual(reflector.calls[calls:],[('memory','2')])
            self.assertEqual(before,{p:p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_failed_classification_and_skill_keep_other_work_and_cached_resume(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';records=write_source(source)
            first=FakeReflector({('classify','0'),('skill','retrieval')})
            runner=ReflectionRunner(first,base_config(),workers=2)
            report,_=runner.run(root/'reflection',source,records,{})
            self.assertNotIn('retrieval',report.skills)
            self.assertIn('reasoning',report.skills)
            self.assertNotIn('0',report.selection['selected_sample_ids'])
            second=FakeReflector();rerun=ReflectionRunner(second,base_config(),workers=2)
            report,_=rerun.run(root/'reflection',source,records,{})
            self.assertEqual([key for stage,key in second.calls if stage=='classify'],['0'])
            self.assertIn('retrieval',report.skills)

    def test_modes_sample_before_classification_and_bad_source_is_skipped(self):
        for benchmark,stride in [('lvbench',3),('egolife',3),('medvidbench',10)]:
            with self.subTest(benchmark=benchmark),TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source';write_source(source,benchmark,31)
                manifest=root/'train.json';manifest.write_text(json.dumps(list(range(31))))
                cfg=base_config();cfg['evolving']['mode']='lite'
                pipeline=EvolutionPipeline(benchmark,manifest,root/'output',cfg,source,reflector=FakeReflector())
                self.assertEqual(pipeline._prepare_training_inputs(),list(range(31))[::stride])
                (source/'trajectories/0.json').write_text('bad json')
                self.assertEqual(pipeline._prepare_training_inputs(),list(range(31))[::stride][1:])

    def test_cli_workers_and_output_path(self):
        base=['run','--benchmark','lvbench','--train-manifest','train.json','--inference-run-dir','source']
        self.assertEqual(parse_args([*base,'--workers','3']).workers,3)
        with self.assertRaises(ValueError):main([*base,'--workers','0'])
        with TemporaryDirectory() as directory,patch('videospy.evolving.cli.EvolutionPipeline') as pipeline:
            pipeline.return_value.run.return_value=Path(directory)
            pipeline.return_value.result={"status":"complete"}
            result=Path(directory)/'path'
            main([*base,'--workers','3','--run-dir-file',str(result)])
            self.assertEqual(pipeline.call_args.kwargs['workers'],3)
            self.assertEqual(pipeline.call_args.kwargs['output_dir'],'output/evolving/lvbench')
            self.assertEqual(result.read_text().strip(),str(Path(directory).resolve()))
            self.assertNotIn('classification_threshold',pipeline.call_args.kwargs['config']['evolving'])


class MemoryTest(unittest.TestCase):
    def test_quotes_must_match_and_time_must_be_valid(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);records=write_source(root)
            trajectory=json.loads((root/records[0]['trajectory']).read_text())
            evidence=build_evidence(trajectory,'hash')
            item={'sample_id':'0','source':{}}
            value=FakeReflector().memory(item,evidence)
            value.pop('evidence_refs')
            value['evidence_ids'] = ['o0:0']
            self.assertEqual(validate_memory(value,evidence)['content'],value['content'])
            value['evidence_ids'] = ['missing-span']
            with self.assertRaises(ValueError):validate_memory(value,evidence)
            value['time']['start_seconds']=-1
            with self.assertRaises(ValueError):validate_memory(value,evidence)

    def test_dedup_against_existing_and_pending_without_merging_different_videos(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';write_source(source)
            manifest=root/'train.json';manifest.write_text('[0,1,2,3]')
            pipeline=EvolutionPipeline('lvbench',manifest,root/'out',base_config(),source,reflector=FakeReflector())
            pipeline._prepare_training_inputs()
            store=EvolutionStateStore(root/'memory.sqlite3')
            store.create_run(pipeline._run_metadata());store.create_initial_version(pipeline.run_id,'initial',{}, {})
            def candidate(sid,video,content='red door opened'):
                return {'source':{'sample_id':sid,'video':video,'benchmark':'lvbench'},'task_types':['retrieval','reasoning'],
                        'context':'opening door','content':content,'time':{'start_seconds':1,'end_seconds':1,'label':''},
                        'event':'door opens','evidence_refs':[{'tool_call_id':'tool1','quote':content}]}
            optimizer=MemoryOptimizer(embedding_function=lambda _: [1.0]+[0.0]*255)
            report=ReflectionReport({}, {},[candidate('1','a'),candidate('2','a','the door opens'),candidate('3','b')])
            entries,summary=optimizer.optimize(store,'initial',report)
            self.assertEqual(len(entries),2);self.assertEqual(summary['duplicates'],1)
            store.commit_version(pipeline.run_id,'initial','final',1,1,0,'hash',report,EvolutionUpdate({},entries,{},{}))
            more=ReflectionReport({}, {},[candidate('4','a'),candidate('5','c')])
            added,summary=optimizer.optimize(store,'final',more)
            self.assertEqual([e.source_sample_id for e in added],['5'])
            optimizer.embedding_function=Mock(side_effect=RuntimeError('embedding unavailable'))
            added,summary=optimizer.optimize(store,'final',ReflectionReport({}, {},[candidate('6','d')]))
            self.assertEqual(added,[]);self.assertEqual(summary['failed'],1)
class ExperimentAndSplitTest(unittest.TestCase):
    def test_committed_splits_have_only_train_and_test(self):
        root = Path(__file__).resolve().parents[1] / "experiments/data_splits/main"
        expected = {
            "lvbench": (1234, 315),
            "egolife": (405, 95),
            "medvidbench": (5146, 1124),
        }
        for benchmark, counts in expected.items():
            split_root = root / benchmark
            train = json.loads((split_root / "train.json").read_text())
            test = json.loads((split_root / "test.json").read_text())
            self.assertEqual((len(train), len(test)), counts)
            self.assertFalse(set(train) & set(test))
            self.assertFalse((split_root / "validation.json").exists())

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "data/MedVidBench/MedVidU_ECCV2026_TrainVal/"
            "medvidu_eccv2026_trainval.json"
        ).is_file(),
        "MedVidBench metadata is not available",
    )
    def test_medvidbench_split_is_group_disjoint_and_stratified(self):
        root = Path(__file__).resolve().parents[1]
        items = json.loads(
            (
                root
                / "data/MedVidBench/MedVidU_ECCV2026_TrainVal/"
                "medvidu_eccv2026_trainval.json"
            ).read_text(encoding="utf-8")
        )
        split_root = root / "experiments/data_splits/main/medvidbench"
        train = json.loads((split_root / "train.json").read_text())
        test = json.loads((split_root / "test.json").read_text())
        audit = medvid_splits.audit_split(items, train, test)
        self.assertEqual(audit["samples"], {"train": 5146, "test": 1124})
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["groups"]["overlap"], 0)
        test_fraction = len(test) / len(items)
        for field in ("dataset_name", "qa_type"):
            total = Counter(item[field] for item in items)
            observed = Counter(items[index][field] for index in test)
            for key, count in total.items():
                expected = count * test_fraction
                self.assertLessEqual(abs(observed[key] - expected) / expected, 0.2)


def fake_chat(self, messages, **kwargs):
    """Exercise offline prompts/parsing in subprocess CLI tests without a model server."""
    prompt=messages[0]['content']
    data=json.JSONDecoder().raw_decode(messages[1 if messages[0]['role'] == 'system' else 0]['content'].split('\nInput:\n',1)[1])[0]
    if prompt.startswith('Evaluate one completed'):
        result={'reason':'evidence and logical efficiency',
                'answer_status':data.get('answer_status', 'correct'), 'evidence_support':'sufficient',
                'score':4 if data.get('answer_status', 'correct') == 'correct' else 2, 'reasoning_sound':True}
    elif prompt.startswith('Reflect on the reasoning'):
        refs = {s['sample_id']: {'sample_id': s['sample_id'], 'message_indices': s['message_indices']}
                for s in (data['current_skill'] or {}).get('sources', [])}
        refs.update({i['sample_id']: {'sample_id': i['sample_id'], 'message_indices': [1]}
                     for i in data['trajectories']})
        result = {'instruction': 'Inspect decisive observations for ' + data['task_type'] + '.',
                  'sources': list(refs.values())}
    elif prompt.startswith('Review this complete Skill'):
        result = {'approved': True, 'feedback': 'Supported methods.', 'instruction_excerpt': ''}
    elif prompt.startswith('Synthesize a concise'):
        result = {'instruction': 'Maintain person and temporal scope.', 'source_tasks': list(data['task_skills'])}
    elif prompt.startswith('Extract ONE useful reasoning case'):
        result = None
    elif prompt.startswith('Extract at most ONE'):
        observation=data['observations'][0]
        result={'context':'What event was observed?', 'event':'opening a door',
                'time':{'start_seconds':1.,'end_seconds':1.,'label':''},
                'evidence_ids':[observation['id']]}
    else:
        raise AssertionError('Unexpected model request')
    return {'content':json.dumps(result)}


class V19BoundaryTest(unittest.TestCase):
    def test_skill_update_failure_keeps_previous_skill_and_continues_next_group(self):
        class InterruptedSkill(FakeReflector):
            def update_skill(self, task, group, current):
                seen=sum(stage=='skill' and key==task for stage,key in self.calls)
                if seen==1:
                    self.calls.append(('skill',task))
                    raise RuntimeError('one failed group')
                return super().update_skill(task,group,current)
        with TemporaryDirectory() as directory:
            root=Path(directory);records=write_source(root/'source')
            reflector=InterruptedSkill()
            report,_=ReflectionRunner(reflector,base_config(),workers=1).run(root/'reflection',root/'source',records,{})
            self.assertEqual(len(report.skills['retrieval']['sources']),6)
            self.assertEqual(len(report.skills['reasoning']['sources']),6)
            self.assertIn('__domain__',report.skills)

    def test_unverified_outcomes_keep_memory_and_can_be_reflected(self):
        reflector = FakeReflector()
        reflector.classify = lambda _: {'reason': 'Uncertain result', 'answer_status': 'unverified', 'evidence_support': 'sufficient', 'score': 3, 'reasoning_sound': True}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = write_source(root / 'source')
            for record in records:
                record['answer'] = 'Open-ended reference'
            report, _ = ReflectionRunner(reflector, base_config()).run(root / 'reflection', root / 'source', records, {})
            self.assertEqual(len(report.memories), 8)
            self.assertEqual(len(report.skills), 3)
            result = json.loads((root / 'reflection/classification.json').read_text())
            self.assertTrue(all(row['outcome'] == 'unverified' for row in result['samples']))

    def test_unexplained_null_and_missing_skill_response_are_rejected(self):
        model=Mock();reflector=Reflector(base_config(),model)
        model.chat.return_value={'content':'null'}
        with self.assertRaises(ValueError):reflector.update_skill('task',[],None)
        model.chat.return_value={'content':None}
        with self.assertRaises(ValueError):reflector.update_skill('task',[],None)
