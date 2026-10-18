"""Score sampling, bounded evidence and request accounting regressions."""
import copy
import json
from collections import Counter
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_evolving import FakeReflector, base_config
from tests.test_skill_review import sample, draft, model, pipeline, no_update
from videospy.evolving.reflector.model import Reflector, evidence_view, validate_assessment
from videospy.evolving.reflector.runner import ReflectionRunner, select_groups
from videospy.evolving.optimizer.skill_optimizer import validate_task_document, skill_input
from videospy.model import ModelClient, collect_model_calls


def assessment(score=5, **overrides):
    return dict(reason='Recorded evidence and reasoning support this score.', answer_status='correct',
                evidence_support='sufficient', reasoning_sound=True, score=score) | overrides


@pytest.mark.parametrize('score', range(6))
def test_score_levels(score):
    assert validate_assessment(assessment(score))['score'] == score


@pytest.mark.parametrize('score', [-1, 6, True, 2.5, '4', None])
def test_score_must_be_integer_in_range(score):
    with pytest.raises(ValueError, match='score'):
        validate_assessment(assessment(score))


@pytest.mark.parametrize('overrides', [dict(answer_status='incorrect'), dict(answer_status='unverified'),
                                     dict(evidence_support='partial'), dict(reasoning_sound=False)])
def test_high_score_requires_all_conditions(overrides):
    with pytest.raises(ValueError, match='Scores 4-5'):
        validate_assessment(assessment(**overrides))


def test_exact_answer_conflict_is_corrected_not_clamped():
    client = model(assessment(), assessment(2))
    result = Reflector(base_config(), client).classify(
        dict(answer_status='incorrect', answer_status_source='exact_option'))
    assert result['answer_status'] == 'incorrect' and result['score'] == 2
    assert client.chat.call_count == 2
    feedback = client.chat.call_args.args[0][-1]['content']
    assert 'answer_status=incorrect' in feedback and 'score=5' in feedback
    assert 'score 0-3' in feedback


def selection(scores, cap=16, global_cap=0):
    items = [dict(sample_id=str(i), task_types=['a', 'b']) for i in range(len(scores))]
    rows = [dict(sample_id=str(i), task_type=task, score=score, outcome='success')
            for i, score in enumerate(scores) for task in ['a', 'b']]
    return select_groups(items, {'samples': rows}, 4, global_cap, 42, cap)


@pytest.mark.parametrize('scores', [[0] * 100 + [5] * 20 + [2] * 30, [3] * 30,
                                   list(range(6)) * 10, [0, 5, 2]])
def test_balanced_sampling_unique_reproducible_and_mixed(scores):
    groups = selection(scores)
    assert groups == selection(scores)
    for batches in groups.values():
        chosen = [item for group in batches for item in group]
        assert len(chosen) == min(16, len(scores))
        assert len({item['sample_id'] for item in chosen}) == len(chosen)
        if len(scores) >= 16:
            counts = Counter(item['score'] for item in chosen)
            assert max(counts.values()) - min(counts.values()) <= 1
        if len(set(scores)) > 1:
            assert all(len({item['score'] for item in group}) > 1 for group in batches)
        assert all(len(group) <= 4 for group in batches)
    capped = selection(scores, global_cap=2)
    assert len({item['sample_id'] for batches in capped.values() for group in batches for item in group}) <= 2


@pytest.mark.parametrize('cap', [0, -1, True, '16', 1.5])
def test_invalid_per_task_cap(cap):
    config = base_config()
    config['evolving']['reflector']['sampling']['max_samples_per_task'] = cap
    with pytest.raises(ValueError, match='max_samples_per_task'):
        ReflectionRunner(FakeReflector(), config)


def test_evidence_excerpts_preserve_indices_pairs_and_original_storage():
    item = sample()
    messages = [{'role': 'assistant', 'content': f'Full message {i}'} for i in range(20)]
    messages[2]['tool_calls'] = [{'id': 't', 'function': {'name': 'overview'}}]
    messages[10] = {'role': 'tool', 'tool_call_id': 't', 'content': 'Complete evidence, with uncertainty.'}
    item['trajectory']['messages'] = messages
    item.update(score=4, reasoning_sound=True)
    doc = validate_task_document(draft(index=10), [item])
    before = copy.deepcopy(doc)
    view = evidence_view([doc, doc])[0]
    assert [m['message_index'] for m in view['trajectory']['messages']] == [2, 9, 10, 11]
    assert view['allowed_message_indices'] == [10]
    assert view['score'] == 4 and view['reasoning_sound']
    assert doc == before and len(doc['sources'][0]['trajectory']['messages']) == 20
    client = model(no_update())
    Reflector(base_config(), client).update_skill('task', [sample('new')], doc)
    payload = json.loads(client.chat.call_args.args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert payload['trajectories'][0] == skill_input(sample('new'))
    assert len(payload['readonly_evidence'][0]['trajectory']['messages']) == 4


def test_unchanged_update_does_not_accumulate_a_new_document():
    current = validate_task_document(draft(), [sample()])
    assert Reflector(base_config(), model(draft())).update_skill('task', [sample('new')], current) is None


def test_reordering_sources_alone_is_no_update():
    value = draft()
    value['sources'].append({'sample_id': 'b', 'message_indices': [1]})
    current = validate_task_document(value, [sample(), sample('b')])
    value['sources'].reverse()
    assert Reflector(base_config(), model(value)).update_skill('task', [], current) is None


def test_low_score_does_not_add_a_memory_case_gate(tmp_path):
    reflector = FakeReflector()
    reflector.classify = Mock(return_value=assessment(2, reasoning_sound=False))
    reflector.reasoning_case = Mock(return_value=None)
    run = pipeline(tmp_path, reflector)
    run.run()
    assert reflector.reasoning_case.call_count == 4
    assert run.result['memory_count'] == 4


@pytest.mark.parametrize('usage', [None, {'prompt_tokens': 20, 'completion_tokens': 5}])
def test_call_metrics_preserve_unknown_usage_and_finish_reason(monkeypatch, usage):
    response = SimpleNamespace(choices=[SimpleNamespace(message={'content': 'result'}, finish_reason='length')], usage=usage)
    monkeypatch.setattr('videospy.model.completion', Mock(return_value=response))
    with collect_model_calls() as calls:
        result = ModelClient(base_config()['agent']['model']).chat([{'role': 'user', 'content': 'hello'}])
    assert result == {'content': 'result'}
    metrics = calls[0]['metrics']
    assert metrics['finish_reason'] == 'length' and metrics['input_characters'] > 0
    assert metrics['elapsed_seconds'] >= 0 and metrics['error'] is None
    assert metrics['input_tokens'] == (20 if usage else None)
    assert metrics['output_tokens'] == (5 if usage else None)


def test_call_failure_records_timing_without_fabricating_usage(monkeypatch):
    monkeypatch.setattr('videospy.model.completion', Mock(side_effect=ValueError('context window exceeded')))
    with collect_model_calls() as calls, pytest.raises(ValueError):
        ModelClient(base_config()['agent']['model']).chat([{'role': 'user', 'content': 'hello'}])
    assert len(calls) == 1
    assert calls[0]['metrics']['input_tokens'] is None
    assert calls[0]['metrics']['error'] == 'context window exceeded'


def test_checkpoint_saves_metrics_and_standard_chat_correction(tmp_path, monkeypatch):
    from videospy.evolving.common.schemas import empty_stage

    outputs = ['   ', json.dumps(draft())]
    responses = [SimpleNamespace(choices=[SimpleNamespace(message={'content': value}, finish_reason=finish)],
                                 usage={'prompt_tokens': 10, 'completion_tokens': 20})
                 for value, finish in zip(outputs, ['length', 'stop'])]
    completion = Mock(side_effect=responses)
    monkeypatch.setattr('videospy.model.completion', completion)
    reflector = Reflector(base_config(), ModelClient(base_config()['agent']['model']))
    runner = ReflectionRunner(reflector, base_config())
    runner.directory = tmp_path
    runner.lock = Lock()
    runner.stages, runner.failures, runner.progress = {'task_skill': empty_stage()}, [], {}
    result = runner._checkpoint('task_skill', 'group.json', {},
                                lambda: reflector.update_skill('task', [sample()], None))
    assert result is not None
    checkpoint = json.loads((tmp_path / 'group.json').read_text())
    assert [call['finish_reason'] for call in checkpoint['call_metrics']] == ['length', 'stop']
    assert checkpoint['elapsed_seconds'] >= 0
    assert all(call.kwargs['response_format'] is None for call in completion.call_args_list)
    assert [m['role'] for m in completion.call_args_list[1].kwargs['messages']] == ['system', 'user', 'assistant', 'user']


def test_explained_empty_is_saved_and_reused_without_another_model_call(tmp_path, monkeypatch):
    from videospy.evolving.common.schemas import empty_stage

    value = no_update('no_reusable_experience', 'The tool failed before any useful observation or decision.')
    completion = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message={'content': json.dumps(value)}, finish_reason='stop')], usage=None))
    monkeypatch.setattr('videospy.model.completion', completion)
    reflector = Reflector(base_config(), ModelClient(base_config()['agent']['model']))
    runner = ReflectionRunner(reflector, base_config())
    runner.directory, runner.lock, runner.progress = tmp_path, Lock(), {}
    runner.stages, runner.failures = {'task_skill': empty_stage()}, []
    producer = lambda: reflector.update_skill('task', [sample()], None)
    assert runner._checkpoint('task_skill', 'group.json', {}, producer) is None
    saved = json.loads((tmp_path / 'group.json').read_text())
    assert json.loads(saved['responses'][0]['content'])['no_update'] == value['no_update']
    assert saved['result'] is None and 'error' not in saved
    assert runner._checkpoint('task_skill', 'group.json', {}, producer) is None
    assert completion.call_count == 1 and runner.stages['task_skill']['reused'] == 1
