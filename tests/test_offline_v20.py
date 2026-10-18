"""Source scope, factual Memory, and offline recovery regression tests."""
import copy
import json
from unittest.mock import Mock

import pytest

from tests.test_evolving import base_config
from videospy.evolved_config import format_skills
from videospy.evolving.common.memory_scope import (
    bind_event_time, memory_exclusion, medvidbench_source_context, source_context,
    timeline_wall_seconds,
)
from videospy.evolving.optimizer.memory_optimizer import _overlap
from videospy.evolving.reflector import Reflector
from videospy.evolving.reflector.evidence import build_evidence, observation_spans, resolve_evidence_ids
from videospy.evolving.storage.memory_store import EvolutionStateStore


def sample(sid, video=None, status='correct'):
    return {'sample_id': sid, 'answer_status': status, 'evidence_support': 'sufficient',
            'source': source_context('lvbench', video or sid), 'cost': {'rounds': 3},
            'trajectory': {'messages': [{'role': 'assistant', 'content': 'Inspect the distinguishing detail before answering.'}]}}


@pytest.mark.parametrize('invalid', [None, 'duration', 'missing_duration', 'unknown_source'])
def test_legacy_egolife_mapping_requires_all_recorded_query_boundaries(tmp_path, monkeypatch, invalid):
    from tests.test_evolving import write_source
    from tests.test_egolife import egolife
    from videospy.evolving.common.training_data import load_egolife_timeline
    records = write_source(tmp_path, 'egolife', count=2)
    manifest = json.loads((tmp_path / 'run.json').read_text())
    for index, record in enumerate(records):
        path = tmp_path / record['trajectory']
        trajectory = json.loads(path.read_text())
        duration = (index + 1) * 10
        if invalid == 'duration' and index == 1:
            duration += 1
        if invalid != 'missing_duration' or index == 0:
            trajectory['messages'].insert(0, {'role': 'user', 'content': f'Video Duration: {duration:.1f}s\n'})
        path.write_text(json.dumps(trajectory))
    if invalid == 'unknown_source':
        manifest['video_root'] = '/another/person'
    timeline = Mock()
    timeline.time_mapping = {'id': 'checked-timeline', 'segments': []}
    def set_time(query):
        timeline.duration = 10 if query == records[0]['query_time'] else 20
    timeline.set_query_time.side_effect = set_time
    monkeypatch.setattr(egolife, 'DEFAULT_VIDEO_ROOT', tmp_path / 'A1_JAKE')
    monkeypatch.setattr('benchmark.egolife.timeline.EgoLifeTimelineReader', Mock(return_value=timeline))
    original = (tmp_path / 'run.json').read_bytes()
    mapping, audit = load_egolife_timeline(tmp_path, manifest)
    assert (tmp_path / 'run.json').read_bytes() == original
    if invalid:
        assert mapping is None and audit['status'] == 'unavailable'
    else:
        assert mapping['id'] == 'checked-timeline' and audit['status'] == 'reconstructed'
        assert audit['checked_samples'] == 2 and timeline.set_query_time.call_count == 2


def test_recorded_egolife_mapping_is_used_without_local_reconstruction(tmp_path):
    from videospy.evolving.common.training_data import load_egolife_timeline
    mapping = {'id': 'original', 'segments': []}
    result, audit = load_egolife_timeline(tmp_path, {'timeline_mapping': mapping})
    assert result == mapping and audit['status'] == 'recorded'


def test_program_correctness_overrides_model_and_keeps_cost_information():
    model = Mock()
    model.chat.return_value = {'content': json.dumps({'reason': 'Apparently supported answer',
                                                     'answer_status': 'correct', 'evidence_support': 'sufficient', 'score': 2, 'reasoning_sound': True})}
    item = {'trajectory': {'messages': []}, 'reference_answer': 'A', 'prediction': 'B',
            'answer_status': 'incorrect', 'answer_status_source': 'exact_option',
            'cost': {'rounds': 16, 'failed_tools': 2, 'token_usage': {'total_tokens': 5000}}}
    result = Reflector(base_config(), model).classify(item)
    assert result['answer_status'] == 'incorrect'
    payload = model.chat.call_args.args[0][0]['content']
    assert '5000' in payload and 'failed_tools' in payload


def test_observation_ids_cover_original_without_copying_long_quotes():
    text = 'Visible fact.\n' + '超长观察' * 601 + '\nFinal observation.'
    source = build_evidence({'messages': [
        {'role': 'tool', 'tool_call_id': 'ok', 'name': 'overview', 'content': text},
        {'role': 'tool', 'tool_call_id': 'bad', 'name': 'overview', 'content': 'Tool `overview` execution failed.'},
    ]}, 'trajectory')
    spans = observation_spans(source)
    assert ''.join(s['text'] for s in spans) == text
    assert all(0 < len(s['text']) <= 400 for s in spans)
    refs = resolve_evidence_ids(source, [spans[0]['id']])
    assert all(r['quote'] in text and r['tool_call_id'] == 'ok' for r in refs)
    with pytest.raises(ValueError):
        resolve_evidence_ids(source, ['invalid'])
    with pytest.raises(ValueError):
        resolve_evidence_ids(source, [spans[0]['id']] * 2)


def memory_row(mid, source, vector, time=None):
    return {'version_id': 'final', 'memory_id': mid, 'source_sample_id': mid, 'content': 'A local event',
            'embedding': json.dumps(vector), 'metadata': json.dumps({'source': source, 'time': time or {}})}


def test_source_filter_runs_before_query_embedding_and_top_k():
    embed = Mock(return_value=[1., 0.])
    store = EvolutionStateStore('/tmp/unused-v20-test.sqlite3', embedding_function=embed)
    early = source_context('egolife', '/train/A1_JAKE', {'date': 'DAY1', 'time': '08000000'})
    future = source_context('egolife', '/train/A1_JAKE', {'date': 'DAY1', 'time': '10000000'})
    now = source_context('egolife', '/eval/A1_JAKE', {'date': 'DAY1', 'time': '09000000'})
    store._memory_rows = Mock(return_value=[memory_row('future', future, [1., 0.]), memory_row('early', early, [.8, .2])])
    audit = {}
    assert store.search('final', 'question', None, 1, .5, now, audit)[0]['memory_id'] == 'early'
    assert audit['excluded'] == {'future_source': 1} and audit['eligible_count'] == 1
    embed.reset_mock()
    for context in (None, source_context('lvbench', 'elsewhere')):
        assert store.search('final', 'question', None, context=context) == []
    embed.assert_not_called()
    store._memory_rows.return_value = [memory_row('same', source_context('lvbench', 'v1'), [1., 0.])]
    assert store.search('final', 'question', None, context=source_context('lvbench', 'v1'))
    assert store.search('final', 'question', None, context=source_context('lvbench', 'v2')) == []


def test_medvidbench_complete_sequence_and_fps_define_host_independent_identity():
    item = {'video': ['/root/data/dataset/v/001.jpg', '/root/data/dataset/v/003.jpg'],
            'data_source': 'dataset', 'metadata': {'video_id': 'v', 'fps': .5}}
    source = medvidbench_source_context(item)
    moved = copy.deepcopy(item)
    moved['video'] = [f.replace('/root/data', '/mnt/data') for f in moved['video']]
    assert medvidbench_source_context(moved)['source_id'] == source['source_id']
    for changed in [item | {'video': list(reversed(item['video']))},
                    item | {'video': [item['video'][0], '/root/data/dataset/v/002.jpg', item['video'][1]]},
                    item | {'metadata': {'video_id': 'v', 'fps': 1.}}]:
        assert medvidbench_source_context(changed)['source_id'] != source['source_id']


def test_timeline_gap_boundary_and_availability_are_separate():
    mapping = {'id': 'map', 'segments': [{'start': 0., 'wall_start': 86370., 'duration': 30.},
                                       {'start': 30., 'wall_start': 90000., 'duration': 30.}]}
    source = source_context('egolife', '/data/A1_JAKE', {'date': 'DAY2', 'time': '02000000'}) | {'timeline_id': 'map'}
    assert timeline_wall_seconds(mapping, 30.) == 90000.
    assert timeline_wall_seconds(mapping, 60.1) is None
    assert timeline_wall_seconds(mapping, True) is None
    interval = {'start_seconds': 20., 'end_seconds': 40., 'label': 'untrusted label'}
    bound = bind_event_time(interval, source, mapping)
    assert bound['calendar']['continuous'] is False
    assert bound['label'] == 'DAY1 23595000 - DAY2 01001000'
    assert source['available_at'] == 93600.
    assert bind_event_time(interval, source, None)['label'] == ''
    assert memory_exclusion({'source': source, 'time': bound}, source | {'timeline_id': 'other'}) == 'incomparable_timeline'
    assert _overlap({'source': source, 'time': bound}, {'source': source | {'timeline_id': 'other'}, 'time': bound}) is None
    with pytest.raises(ValueError, match='later'):
        bind_event_time(interval, source | {'available_at': 86400.}, mapping)


def test_empty_memory_does_not_change_agent_inputs_and_budget_keeps_whole_entries():
    from tests.test_evolved_memory_injection import agent_config
    from videospy.agent import VideoSpyAgent

    agent = VideoSpyAgent.__new__(VideoSpyAgent)
    agent.config = agent_config()
    agent.memory_context = source_context('lvbench', 'v') | {'query': 'Only the semantic question', 'sample_id': 'q1'}
    agent.memory_store = Mock()
    agent.memory_store.search.return_value = []
    assert agent._retrieve_memory_prompt('formatted question with output requirements') == ''
    assert agent.memory_store.search.call_args.kwargs['query'] == 'Only the semantic question'
    agent.memory_store.search.return_value = [
        {'memory_id': f'm{i}', 'content': str(i) + 'x' * 499,
         'metadata': {'source': source_context('lvbench', 'v'), 'uncertainty': 'unsure'}} for i in range(3)]
    prompt = agent._retrieve_memory_prompt('formatted')
    assert len(prompt) <= 1600
    assert len(agent.memory_audit['ids']) == 2 and agent.memory_audit['budget_skipped'] == ['m2']
    for i in range(2):
        assert str(i) + 'x' * 499 in prompt
    agent.memory_store = None
    assert agent._retrieve_memory_prompt('formatted') == ''
    assert agent.memory_audit['ids'] == []


@pytest.mark.parametrize('corrected', ['valid', 'null', 'invalid'])
def test_corrective_conversation_and_raw_responses_are_preserved(tmp_path, corrected):
    from threading import Lock
    from types import SimpleNamespace
    from unittest.mock import patch
    from videospy.evolving.common.schemas import empty_stage
    from videospy.evolving.reflector.runner import ReflectionRunner
    from videospy.model import ModelClient

    valid = {'instruction': 'Inspect a distinguishing detail.', 'sources': [
        {'sample_id': sid, 'message_indices': [0]} for sid in ('a', 'b')]}
    invalid = {'skill': valid}
    outputs = [invalid, valid if corrected == 'valid' else None if corrected == 'null' else invalid]
    responses = [SimpleNamespace(choices=[SimpleNamespace(message={'content': json.dumps(value)})]) for value in outputs]
    reflector = Reflector(base_config(), ModelClient(base_config()['agent']['model']))
    runner = ReflectionRunner(reflector, base_config())
    runner.directory, runner.lock, runner.progress = tmp_path, Lock(), {}
    runner.stages, runner.failures = {'task_skill': empty_stage()}, []
    with patch('videospy.model.completion', side_effect=responses) as completion, patch('videospy.model.record_llm_response'):
        result = runner._checkpoint('task_skill', 'group.json', {},
                                    lambda: reflector.update_skill('task', [sample('a'), sample('b')], None))
    assert completion.call_count == 2
    retry = completion.call_args_list[1].kwargs['messages']
    assert [m['role'] for m in retry] == ['system', 'user', 'assistant', 'user']
    assert retry[2]['content'] == json.dumps(invalid)
    assert 'top-level JSON' in retry[3]['content'] and 'not as a shortcut' in retry[3]['content']
    checkpoint = json.loads((tmp_path / 'group.json').read_text())
    assert [json.loads(response['content']) for response in checkpoint['responses']] == outputs
    assert ('error' in checkpoint) == (corrected == 'invalid')
    if corrected == 'valid':
        assert len(result['sources']) == 2
    else:
        assert result is None


def test_failed_and_null_skill_groups_block_automatic_evaluation_but_keep_memories(tmp_path):
    from unittest.mock import patch
    from tests.test_evolving import FakeReflector, write_source
    from videospy.evolving.pipeline import EvolutionPipeline
    from videospy.evolving.cli import main

    source, manifest = tmp_path / 'source', tmp_path / 'train.json'
    write_source(source, count=4)
    manifest.write_text('[0,1,2,3]')
    reflector = FakeReflector()
    def update(task, group, current):
        if task == 'retrieval':
            raise ValueError('invalid model output')
        return None
    reflector.update_skill = update
    pipeline = EvolutionPipeline('lvbench', manifest, tmp_path / 'out', base_config(), source, reflector=reflector)
    run = pipeline.run()
    result = pipeline.result
    assert result['status'] == 'partial'
    assert result['skill_characters'] == 0 and result['stages']['task_skill']['nonempty'] == 0
    assert result['evaluation_blockers'] and result['evaluation_version'] is None
    assert result['memory_count'] == 4 and len(json.loads((run / 'memories.json').read_text())) == 4
    with patch('videospy.evolving.cli.EvolutionPipeline', return_value=pipeline), patch.object(pipeline, 'run', return_value=run):
        marker = tmp_path / 'run-path'
        status = main(['run', '--benchmark', 'lvbench', '--train-manifest', str(manifest),
                       '--inference-run-dir', str(source), '--run-dir-file', str(marker)])
    assert status == 1 and not marker.exists()
    reflector.update_skill = Mock(return_value=None)
    clean = EvolutionPipeline('lvbench', manifest, tmp_path / 'clean', base_config(), source, reflector=reflector)
    clean.run()
    assert clean.result['status'] == 'complete' and not clean.result['evaluation_blockers']
    assert clean.result['evaluation_version'] is not None


def test_memory_labels_are_program_bound_and_prompt_excludes_answer_conclusions():
    from videospy.evolving.reflector.model import validate_memory
    evidence = build_evidence({'messages': [{'role': 'tool', 'name': 'overview', 'tool_call_id': 't',
                                           'content': 'A presenter demonstrates a tablet on a stage.'}]}, 'source')
    memory = {'context': 'What is on the stage?', 'content': 'A presenter demonstrates a tablet.',
              'event': 'Tablet demonstration', 'time': {'start_seconds': None, 'end_seconds': None, 'label': 'invented date'},
              'evidence_ids': ['o0:0']}
    assert validate_memory(memory, evidence)['time']['label'] == ''
    del memory['time']['label']
    assert validate_memory(memory, evidence)['time']['label'] == ''
    model = Mock()
    model.chat.return_value = {'content': 'null'}
    Reflector(base_config(), model).memory({'source': {}, 'question': 'Which option fits?'}, evidence)
    prompt = model.chat.call_args.args[0][0]['content']
    assert 'option comparisons and final-answer conclusions' in prompt
    assert 'evidence_ids must be a JSON list' in prompt


def test_skill_request_exposes_message_indices_and_assessments_without_extra_calls():
    from videospy.evolving.optimizer.skill_optimizer import SkillOptimizer
    model = Mock()
    model.chat.return_value = {'content': 'null'}
    optimizer = SkillOptimizer(Reflector(base_config(), model), lambda stage, path, inputs, produce: produce())
    optimizer.optimize_task('task', [[sample('a'), sample('b', status='incorrect')]])
    assert model.chat.call_count == 1
    payload = json.loads(model.chat.call_args.args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    good, bad = payload['trajectories']
    assert good['answer_status'] == 'correct' and bad['answer_status'] == 'incorrect'
    assert good['trajectory']['messages'][0]['message_index'] == 0


def test_memory_keeps_original_qualification_instead_of_generated_claim():
    from videospy.evolving.reflector.model import validate_memory
    text = 'The athlete prepares for a routine, but the number is not explicitly visible or confirmed by the UI text.'
    source = build_evidence({'messages': [{'role': 'tool', 'tool_call_id': 't', 'content': text}]}, 'source')
    output = {'context': 'What is visible on the uniform?', 'event': 'Athlete preparing',
              'content': 'The UI confirms the athlete wears number 365.', 'uncertainty': 'Confirmed',
              'time': {'start_seconds': None, 'end_seconds': None}, 'evidence_ids': ['o0:0']}
    result = validate_memory(output, source)
    assert result['content'] == text and result['evidence_refs'][0]['quote'] == text
    assert result['uncertainty'].startswith('Unverified')


@pytest.mark.parametrize('tool,header,expected', [
    ('frame_inspect', 'Frame inspection at 12.000s (requested 12.100s):\n', [12.0]),
    ('frame_inspect', 'The description mentions a frame at 12.0s.\n', []),
    ('overview', 'Frame inspection at 12.000s (requested 12.100s):\n', []),
])
def test_memory_preserves_source_frames_without_inventing_event_time(tool, header, expected):
    from videospy.evolving.reflector.model import validate_memory
    text = 'The person wears a white oven mitt.'
    source = build_evidence({'messages': [{'role': 'tool', 'name': tool, 'tool_call_id': 't',
                                          'content': header + text}]}, 'source')
    output = {'context': 'What is the person wearing?', 'event': 'Oven mitt',
              'time': {'start_seconds': None, 'end_seconds': None},
              'evidence_ids': ['o0:0'], 'evidence_frames': [999.]}
    result = validate_memory(output, source)
    assert result['evidence_frames'] == expected
    assert result['time'] == {'start_seconds': None, 'end_seconds': None, 'label': ''}


def test_inspection_coordinates_require_matching_timeline_even_without_event_time():
    source = source_context('egolife', '/data/A1_JAKE', {'date': 'DAY2', 'time': '02000000'})
    metadata = {'source': source | {'timeline_id': 'map'}, 'evidence_frames': [12.]}
    assert memory_exclusion(metadata, source | {'timeline_id': 'map'}) is None
    for context in (source, source | {'timeline_id': 'other'}):
        assert memory_exclusion(metadata, context) == 'incomparable_timeline'


@pytest.mark.parametrize('frame', [5., 15., 21.])
def test_memory_extraction_checks_inspection_frames_against_source_boundary(tmp_path, frame):
    from tests.test_evolving import FakeReflector, write_source
    from videospy.evolving.reflector.runner import ReflectionRunner

    source = tmp_path / 'source'
    records = write_source(source, 'egolife', count=1)
    records[0]['query_time'] = {'date': 'DAY1', 'time': '00001000'}
    manifest = json.loads((source / 'run.json').read_text())
    manifest['timeline_mapping'] = {'id': 'map', 'segments': [
        {'start': 0., 'wall_start': 0., 'duration': 20.}]}
    (source / 'run.json').write_text(json.dumps(manifest))
    reflector = FakeReflector()
    original = reflector.memory
    reflector.memory = lambda item, evidence: original(item, evidence) | {
        'time': {'start_seconds': None, 'end_seconds': None}, 'evidence_frames': [frame]}
    report, _ = ReflectionRunner(reflector, base_config()).run(tmp_path / 'reflection', source, records, {})
    assert bool(report.memories) == (frame == 5.)
    assert report.stages['memory']['failed'] == (frame != 5.)
    if report.memories:
        assert report.memories[0]['time']['start_seconds'] is None


def test_answer_analysis_is_visible_but_cannot_be_selected_as_memory():
    from videospy.evolving.reflector.model import validate_memory
    text = ('**Observation:**\nThe screen displays Microsoft Build.\n\n'
            '**Categorization Analysis:**\nNews conference is the best fit among the choices.\n\n'
            '**Conclusion:**\nThe answer is News conference.\n')
    source = build_evidence({'messages': [{'role': 'tool', 'tool_call_id': 't', 'content': text}]}, 'source')
    spans = observation_spans(source)
    assert ''.join(s['text'] for s in spans) == text
    good = next(s for s in spans if 'screen displays' in s['text'])
    assert good['memory_eligible']
    output = {'context': 'Which event is shown?', 'event': 'Event signage',
              'time': {'start_seconds': None, 'end_seconds': None}, 'evidence_ids': [good['id']]}
    assert 'Microsoft Build' in validate_memory(output, source)['content']
    for span in spans:
        if 'best fit' in span['text'] or 'The answer' in span['text']:
            assert not span['memory_eligible']
            with pytest.raises(ValueError, match='memory_eligible'):
                validate_memory(output | {'evidence_ids': [span['id']]}, source)


def test_memory_never_cuts_a_long_sentence_or_silently_truncates_selected_spans():
    from videospy.evolving.reflector.model import validate_memory
    text = 'The identity ' + 'x' * 600 + ' is not confirmed.'
    source = build_evidence({'messages': [{'role': 'tool', 'tool_call_id': 't', 'content': text}]}, 'source')
    assert not any(s['memory_eligible'] for s in observation_spans(source))
    text = 'x' * 290 + '.\n' + 'y' * 290 + '.'
    source = build_evidence({'messages': [{'role': 'tool', 'tool_call_id': 't', 'content': text}]}, 'source')
    spans = observation_spans(source)
    output = {'context': 'Observation', 'event': 'Event', 'time': {'start_seconds': None, 'end_seconds': None},
              'evidence_ids': [s['id'] for s in spans]}
    with pytest.raises(ValueError, match='exactly one'):
        validate_memory(output, source)
    for span in spans:
        result = validate_memory(output | {'evidence_ids': [span['id']]}, source)
        assert result['content'] == span['text'].strip()
        assert len(result['content']) <= 400


@pytest.mark.parametrize('has_task_skill', [False, True])
def test_only_domain_failure_with_usable_task_rules_is_partial(has_task_skill):
    from videospy.evolving.common.schemas import empty_stage, training_status
    stages = {name: empty_stage() | {'succeeded': 2} for name in
              ('classification', 'task_skill', 'memory', 'embedding')}
    stages['domain_skill'] = empty_stage() | {'failed': 1}
    failures = [{'stage': 'domain_skill', 'reason': 'Invalid JSON'}]
    assert training_status(stages, failures, has_task_skill=has_task_skill) == ('partial' if has_task_skill else 'failed')
    for name in ('classification', 'task_skill', 'memory', 'embedding'):
        failed = stages | {name: empty_stage() | {'failed': 1}}
        assert training_status(failed, failures, has_task_skill=has_task_skill) == 'failed'


def test_domain_failure_keeps_task_memory_and_evaluation_then_resumes_only_failed_node(tmp_path):
    from tests.test_evolving import FakeReflector, write_source
    from videospy.evolving.pipeline import EvolutionPipeline
    source, manifest = tmp_path / 'source', tmp_path / 'train.json'
    write_source(source, count=4)
    manifest.write_text('[0,1,2,3]')
    reflector = FakeReflector(failures={('domain', 'lvbench')})
    pipeline = EvolutionPipeline('lvbench', manifest, tmp_path / 'out', base_config(), source, reflector=reflector)
    run = pipeline.run()
    result = pipeline.result
    assert result['status'] == 'partial' and result['evaluation_version'] and not result['evaluation_blockers']
    assert result['domain_skill_available'] is False
    assert result['memory_count'] == 4 and result['skill_characters'] > 0
    assert result['stages']['domain_skill']['failed'] == 1
    assert any(f['stage'] == 'domain_skill' for f in result['failures'])
    assert '__domain__' not in json.loads((run / 'skills.json').read_text())
    assert json.loads((run / 'final_version.json').read_text())['version_id'] == result['evaluation_version']
    reflector.calls.clear()
    reflector.failures.clear()
    pipeline = EvolutionPipeline('lvbench', manifest, tmp_path / 'out', base_config(), source,
                                 reflector=reflector, run_dir=run)
    pipeline.run()
    assert reflector.calls == [('domain', 'lvbench'), ('review', '__domain__')]
    assert pipeline.result['status'] == 'complete' and pipeline.result['domain_skill_available'] is True
    assert pipeline.result['stages']['task_skill']['reused'] == result['stages']['task_skill']['succeeded']
