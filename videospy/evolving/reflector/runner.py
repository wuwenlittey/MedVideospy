"""Run classification, grouping, Skill reflection, and independent Memory extraction."""
import json
import logging
import math
import random
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from threading import Lock
from urllib.parse import quote

from tqdm import tqdm
from videospy.model import collect_model_calls

from ..storage.artifacts import config_identity, hash_json, write_json
from ..common.training_data import load_egolife_timeline, load_medvidbench_sources, load_trajectory
from ..common.schemas import ReflectionReport, empty_stage, validate_skill_document
from ..common.memory_scope import source_context, bind_event_time
from ..optimizer.skill_optimizer import DOMAIN_TASK, SkillOptimizer
from ..common.task_types import record_task_types
from .evidence import build_evidence, repeated_frame_observations, trajectory_for_reflection


def group_samples(selected, group_size):
    remaining = deque(sorted(selected, key=lambda item: item['score']))
    groups = [[] for _ in range(math.ceil(len(selected) / group_size))]
    while remaining:
        for group in groups:
            if remaining and len(group) < group_size:
                group.append(remaining.pop())
            if remaining and len(group) < group_size:
                group.append(remaining.popleft())
    return groups


def classify_by_task(items, assessments):
    rows = []
    for item in items:
        result = assessments.get(item['sample_id'])
        if result is None:
            outcome = 'failed'
        elif result['answer_status'] == 'correct' and result['evidence_support'] == 'sufficient':
            outcome = 'success'
        elif result['answer_status'] == 'incorrect' or result['evidence_support'] != 'sufficient':
            outcome = 'problem'
        else:
            outcome = 'unverified'
        for task in item['task_types']:
            rows.append({'sample_id': item['sample_id'], 'task_type': task, 'outcome': outcome,
                         **(result or {'reason': 'assessment failed'})})
    return {'samples': rows, 'task_types': sorted({row['task_type'] for row in rows})}


def select_groups(items, classification, group_size, max_samples, seed, max_samples_per_task=16):
    by_id = {item['sample_id']: item for item in items}
    eligible = sorted({row['sample_id'] for row in classification['samples'] if row['outcome'] != 'failed'})
    if max_samples and len(eligible) > max_samples:
        eligible = random.Random(seed).sample(eligible, max_samples)
    allowed, buckets = set(eligible), defaultdict(list)
    for row in classification['samples']:
        if row['sample_id'] in allowed and row['outcome'] != 'failed':
            buckets[row['task_type']].append({**by_id[row['sample_id']], **row})
    randomizer = random.Random(seed)
    groups = {}
    for task, members in sorted(buckets.items()):
        randomizer.shuffle(members)
        levels = {score: deque(item for item in members if item['score'] == score)
                  for score in range(5, -1, -1)}
        selected = []
        while len(selected) < max_samples_per_task and any(levels.values()):
            for bucket in levels.values():
                if bucket and len(selected) < max_samples_per_task:
                    selected.append(bucket.popleft())
        groups[task] = group_samples(selected, group_size)
    return groups


class ReflectionRunner:
    def __init__(self, reflector, config, logger=None, workers=2, reuse_reflection_dir=None):
        initial_skills = config.get('TASK_SKILLS') or {}
        if not isinstance(initial_skills, dict):
            raise ValueError('TASK_SKILLS must map task types to document Skills')
        for document in initial_skills.values():
            validate_skill_document(document)
        self.reflector = reflector
        self.logger = logger or logging.getLogger(__name__)
        sampling = config.get("evolving", {}).get("reflector", {}).get("sampling", {})
        self.group_size = int(sampling.get("group_size", 4))
        self.max_samples = int(sampling.get("max_samples", 0))
        self.max_samples_per_task = sampling.get("max_samples_per_task", 16)
        if (not isinstance(self.max_samples_per_task, int) or isinstance(self.max_samples_per_task, bool)
                or self.max_samples_per_task < 1):
            raise ValueError("max_samples_per_task must be a positive integer")
        if "bad_ratio" in sampling:
            self.logger.warning("sampling.bad_ratio is deprecated and ignored; Skill sampling now balances score levels")
        self.seed = int(sampling.get("seed", 42))
        self.workers = int(workers)
        if self.group_size < 1 or self.max_samples < 0 or self.workers < 1:
            raise ValueError("group_size/workers must be positive and max_samples nonnegative")
        self.reuse_dir = Path(reuse_reflection_dir) if reuse_reflection_dir else None
        from . import evidence, model, reasoning_case
        from videospy.core import evidence as core_evidence
        from ..common import memory_scope, training_data, schemas, review
        from ..optimizer import skill_optimizer
        self.identity = {"version": "26-explained-skill-no-update", "model": config_identity(reflector.model_config),
                         "base_instructions": config.get("SYSTEM_PROMPT", ""),
                         "implementation": hash_json([Path(path).read_text() for path in
                             (__file__, model.__file__, evidence.__file__, core_evidence.__file__, reasoning_case.__file__, skill_optimizer.__file__,
                              training_data.__file__, memory_scope.__file__, schemas.__file__, review.__file__)])}

    def _record(self, stage, node, result=None, error=None, reused=False):
        with self.lock:
            counts = self.stages[stage]
            if error is not None:
                counts["failed"] += 1
                self.failures.append({"stage": stage, "node": node, "reason": str(error)[:1000]})
            else:
                counts["succeeded"] += 1
                counts["empty"] += int(result is None or isinstance(result, dict) and result.get('instruction') == '')
                counts["reused"] += int(reused)
                if stage == 'skill_review' and result:
                    counts['approved' if result['approved'] else 'rejected'] = counts.get(
                        'approved' if result['approved'] else 'rejected', 0) + 1
            progress = self.progress.get(stage)
            if progress is not None:
                progress.set_postfix(failed=counts["failed"], reused=counts["reused"], refresh=False)
                progress.update(1)

    def _checkpoint(self, stage, relative, inputs, producer):
        path = self.directory / relative
        identity = hash_json({**self.identity, "input": inputs})
        for source in (path, self.reuse_dir / relative if self.reuse_dir else None):
            if source is None or not source.is_file():
                continue
            try:
                stored = json.loads(source.read_text())
                if isinstance(stored, dict) and stored.get("identity") == identity and "result" in stored:
                    if source != path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        write_json(path, stored)
                    self._record(stage, relative, stored["result"], reused=True)
                    self.logger.info("Reflection checkpoint reused: %s", relative)
                    return stored["result"]
            except (OSError, ValueError) as error:
                self.logger.warning("Ignoring unreadable checkpoint %s: %s", source, error)
        started = perf_counter()
        calls = []
        try:
            with collect_model_calls() as calls:
                try:
                    result = producer()
                finally:
                    for index, call in enumerate(calls):
                        self.logger.info("Reflection call: stage=%s checkpoint=%s call=%d metrics=%s",
                                         stage, relative, index + 1, json.dumps(call.get('metrics', {})))
        except Exception as error:
            self.logger.warning("Reflection failed; skipping %s: %s", relative, error)
            self._record(stage, relative, error=error)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {"identity": identity, "error": str(error)[:1000],
                              **({'working': inputs['document']} if stage == 'skill_review' else {}),
                              "responses": [call['output'] for call in calls],
                              "elapsed_seconds": perf_counter() - started,
                              "call_metrics": [call.get('metrics', {}) for call in calls]})
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"identity": identity, "result": result,
                          **({'working': inputs['document']} if stage == 'skill_review' else {}),
                          'responses': [call['output'] for call in calls],
                          'elapsed_seconds': perf_counter() - started,
                          'call_metrics': [call.get('metrics', {}) for call in calls]})
        self._record(stage, relative, result)
        self.logger.info("Reflection checkpoint written: %s elapsed_seconds=%.3f outcome=%s", relative, perf_counter() - started,
                         'empty' if result is None else 'nonempty')
        return result

    def run(self, reflection_dir, run_dir, records, current_state):
        for document in (current_state.get('skills') or {}).values():
            validate_skill_document(document)
        self.directory = Path(reflection_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.stages = {name: empty_stage() for name in ("classification", "task_skill", "skill_review", "domain_skill", "skill_revision", "memory", "reasoning_case")}
        self.failures = []
        self.progress = {}
        source_run = json.loads((run_dir / "run.json").read_text())
        benchmark = source_run["dataset"]
        medvid_sources = load_medvidbench_sources(run_dir, source_run, records) if benchmark == "medvidbench" else {}
        timeline_mapping = source_run.get("timeline_mapping")
        if benchmark == 'egolife':
            timeline_mapping, timeline_audit = load_egolife_timeline(run_dir, source_run)
            write_json(self.directory / 'timeline_mapping.json', {'mapping': timeline_mapping, **timeline_audit})
            self.logger.info('EgoLife source timeline: %s', json.dumps(timeline_audit, ensure_ascii=False))
        items, evidence = [], {}
        for record in records:
            sid = str(record["sample_key"])
            try:
                trajectory = load_trajectory(run_dir, record)
                readable = trajectory_for_reflection(trajectory)
                original = build_evidence(trajectory, hash_json(readable))
                video = (source_run.get("video_root") if benchmark == "egolife" else
                         record.get("video_id") or record.get("video_path"))
                if not video:
                    self.logger.warning("Source video unknown for sample %s", sid)
                reference = record.get("answer")
                if reference is None or not str(reference).strip():
                    reference = record.get("ground_truth")
                item = {"sample_id": sid, "task_types": record_task_types(record),
                        "question": str(trajectory.get("question") or record.get("question") or ""),
                        "prediction": str(record.get("prediction") or record.get("raw_prediction") or ""),
                        "reference_answer": str(reference).strip() if reference is not None else "",
                        "trajectory": readable,
                        "source": {**(medvid_sources[sid] if benchmark == "medvidbench" else
                                       source_context(benchmark, video, record.get("query_time"))),
                                   "sample_id": sid,
                                   "trajectory": str((run_dir / record["trajectory"]).resolve())}}
                if benchmark == 'egolife' and timeline_mapping:
                    item['source']['timeline_id'] = timeline_mapping['id']
                if benchmark in {'lvbench', 'egolife'} and item['reference_answer'] in {'A', 'B', 'C', 'D'}:
                    item['answer_status'] = 'correct' if item['prediction'] == item['reference_answer'] else 'incorrect'
                    item['answer_status_source'] = 'exact_option'
                else:
                    item['answer_status_source'] = 'reflector'
                observations = original['messages']
                item['cost'] = {'rounds': record.get('agent_rounds'), 'token_usage': record.get('token_usage'),
                                'finish_reason': readable.get('finish_reason'),
                                'successful_tools': sum(not m['failed'] for m in observations),
                                'failed_tools': sum(m['failed'] for m in observations),
                                'duplicate_tools': sum(m['content'].startswith('Duplicate `') for m in observations),
                                'repeated_frame_observations': repeated_frame_observations(readable)}
                items.append(item)
                evidence[sid] = original
            except Exception as error:
                self.logger.warning("Skipping invalid trajectory %s: %s", sid, error)
                self._record("classification", sid, error=error)
                self._record("memory", sid, error=error)

        def assess(item):
            def classify():
                if not item["reference_answer"]:
                    raise ValueError("Missing reference answer; classification unavailable")
                result = self.reflector.classify(item)
                result['answer_status_source'] = item['answer_status_source']
                if item['answer_status_source'] == 'exact_option':
                    result['answer_status'] = item['answer_status']
                return result
            return self._checkpoint("classification", f"classification/{hash_json(item['sample_id'])[:24]}.json", item, classify)

        def memory(item):
            sid = item["sample_id"]
            # Outcome assessments and groups are deliberately absent from this identity.
            inputs = {"source": item["source"], "question": item["question"], "evidence": evidence[sid]}
            def extract():
                result = self.reflector.memory(item, evidence[sid])
                if result is not None:
                    result['time'] = bind_event_time(result['time'], item['source'], timeline_mapping)
                    frames = result.get('evidence_frames') or []
                    if frames:
                        # Validate source coordinates without treating the inspection as the event time.
                        bind_event_time({'start_seconds': min(frames), 'end_seconds': max(frames)},
                                        item['source'], timeline_mapping)
                return result
            result = self._checkpoint("memory", f"memories/{hash_json(sid)[:24]}.json", inputs,
                                      extract)
            return {**result, "source": item["source"], "task_types": item["task_types"]} if result else None

        def reasoning_case(item):
            sid = item['sample_id']
            return self._checkpoint('reasoning_case', f'cases/{hash_json(sid)[:24]}.json', item,
                                    lambda: self.reflector.reasoning_case(item))

        inherited = current_state.get("skills") or {}
        # Only this run's successfully reviewed Task Skills may reach publication.
        skills = {}
        skill_optimizer = SkillOptimizer(self.reflector, self._checkpoint)
        with ExitStack() as progress_stack, ThreadPoolExecutor(max_workers=self.workers) as executor:
            for position, (stage, label) in enumerate((("classification", "Classification"), ("memory", "Memory extraction"))):
                self.progress[stage] = progress_stack.enter_context(tqdm(
                    total=len(records), initial=len(records) - len(items), desc=label,
                    unit="trajectory", position=position, dynamic_ncols=True))
                self.progress[stage].set_postfix(failed=self.stages[stage]["failed"], reused=0, refresh=False)
            # Interleave independent requests so neither route depends on the other succeeding.
            assessing, extracting = {}, []
            for item in items:
                assessing[executor.submit(assess, item)] = item["sample_id"]
                extracting.append(executor.submit(memory, item))
            assessments = {assessing[future]: future.result() for future in as_completed(assessing)}
            case_items = [dict(item, **{k: assessments[item['sample_id']][k] for k in
                                      ('answer_status', 'evidence_support')}) for item in items
                          if (assessments.get(item['sample_id']) or {}).get('answer_status') == 'correct'
                          and (assessments.get(item['sample_id']) or {}).get('evidence_support') == 'sufficient']
            self.stages['reasoning_case']['skipped'] = len(items) - len(case_items)
            self.progress['reasoning_case'] = progress_stack.enter_context(tqdm(
                total=len(case_items), desc='Reasoning cases', unit='trajectory', position=2, dynamic_ncols=True))
            case_futures = {executor.submit(reasoning_case, item): item['sample_id'] for item in case_items}
            cases = {case_futures[f]: result for f in as_completed(case_futures) if (result := f.result()) is not None}
            classification = classify_by_task(items, assessments)
            write_json(self.directory / "classification.json", classification)
            groups = select_groups(items, classification, self.group_size, self.max_samples, self.seed, self.max_samples_per_task)
            selected = sorted({item["sample_id"] for batches in groups.values() for group in batches for item in group})
            selection = {"selected_sample_ids": selected, "memory_sample_ids": [item["sample_id"] for item in items],
                         "case_sample_ids": sorted(cases),
                         "skill_strategy": "document",
                         "group_count": sum(map(len, groups.values())),
                         "max_samples_per_task": self.max_samples_per_task, "group_size": self.group_size,
                         "score_distributions": {task: {
                             "candidate": dict(sorted(Counter(row['score'] for row in classification['samples']
                                 if row['task_type'] == task and row['outcome'] != 'failed').items())),
                             "selected": dict(sorted(Counter(i['score'] for g in batches for i in g).items()))}
                             for task, batches in groups.items()},
                         "groups": {task: [[{"sample_id": i["sample_id"], "outcome": i["outcome"],
                                            "answer_status": i["answer_status"], "evidence_support": i["evidence_support"],
                                            "score": i["score"], "reasoning_sound": i["reasoning_sound"], "reason": i["reason"]} for i in g]
                                           for g in batches] for task, batches in groups.items()}}
            write_json(self.directory / "selection.json", selection)
            self.stages["task_skill"]["skipped"] = sum(task not in groups for task in classification["task_types"])
            self.logger.info("Classification completed: trajectories=%d selected=%d groups=%d workers=%d",
                             len(items), len(selected), selection["group_count"], self.workers)
            self.progress["task_skill"] = progress_stack.enter_context(tqdm(
                total=selection["group_count"], desc="Task Skill", unit="group", position=2, dynamic_ncols=True))
            self.progress['skill_review'] = progress_stack.enter_context(tqdm(
                total=None, desc='Skill review', unit='document', position=3, dynamic_ncols=True))
            task_futures = {executor.submit(skill_optimizer.optimize_task, task, batches,
                                                  inherited.get(task)): task
                            for task, batches in groups.items()}
            for future in as_completed(task_futures):
                result = future.result()
                task = task_futures[future]
                if result is not None and result.get('instruction'):
                    skills[task] = result
            # Preserve source order regardless of completion order.
            memories = [result for future in extracting if (result := future.result()) is not None]
            memories.extend({**cases[item['sample_id']], 'source': item['source'], 'task_types': item['task_types']}
                            for item in items if item['sample_id'] in cases)

        task_skills = {task: skills[task] for task in sorted(skills) if task != DOMAIN_TASK}
        if len(task_skills) >= 2:
            domain = skill_optimizer.optimize_domain(benchmark, task_skills)
            if domain is not None and domain.get('instruction'):
                skills[DOMAIN_TASK] = domain
        else:
            self.stages['domain_skill']['skipped'] = 1
        write_json(self.directory / "selection.json", selection)
        # Human-readable final Skills; group checkpoints belong to the resume cache.
        expected = {"domain.json" if task == DOMAIN_TASK else f"tasks/{quote(task, safe=' ')}.json" for task in skills}
        for old in (self.directory / 'skills').glob('**/*.json'):
            if old.relative_to(self.directory / 'skills').as_posix() not in expected:
                old.unlink()
        for task, document in skills.items():
            relative = "domain.json" if task == DOMAIN_TASK else f"tasks/{quote(task, safe=' ')}.json"
            path = self.directory / "skills" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, document)
        self.failures.sort(key=lambda item: (item["stage"], item["node"]))
        report = ReflectionReport(selection, skills, memories, self.stages, self.failures)
        write_json(self.directory / "report.json", report.to_dict())
        return report, hash_json({"classification": classification, "selection": selection})
