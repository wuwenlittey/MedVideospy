"""Decision provenance, cross-video retrieval, and case/fact separation."""
import json
from unittest.mock import Mock

import pytest

from tests.test_evolving import FakeReflector, base_config, write_source
from tests.test_evolved_memory_injection import FakeVideoReader, agent_config
from videospy.agent import VideoSpyAgent
from videospy.evolving.common.memory_scope import memory_exclusion, source_context
from videospy.evolving.pipeline import EvolutionPipeline
from videospy.evolving.storage.memory_store import EvolutionStateStore, hash_embedding
from videospy.evolving.reflector import Reflector
from videospy.evolving.reflector.reasoning_case import case_question, decision_pairs, validate_reasoning_case


def trajectory():
    return {'messages': [
        {'role': 'assistant', 'tool_calls': [{'id': 't1', 'function': {
            'name': 'frame_inspect', 'arguments': '{"timestamps":[12],"query":"read the sign"}'}}]},
        {'role': 'tool', 'name': 'frame_inspect', 'tool_call_id': 't1',
         'content': 'Frame inspection at 12s: The sign reads EXIT.'}]}


def output():
    return {'problem': 'Read a sign at a specified time.', 'result': 'The sign was legible.', 'when': 'A question specifies when a sign is visible',
            'action': 'Inspect the specified time to transcribe the sign', 'stop': 'The relevant text is legible',
            'evidence_indices': [1]}


def test_cases_bind_original_actions_and_results_and_hide_answers_from_index():
    pairs = decision_pairs(trajectory())
    case = validate_reasoning_case(output(), pairs, 'What does the sign say?\n(A) EXIT\n(B) STOP')
    assert case['context'] == 'What does the sign say?'
    assert 'frame_inspect({"timestamps":[12],"query":"read the sign"})' in case['content']
    assert case['evidence_refs'][0]['arguments']['timestamps'] == [12]
    assert case['evidence_refs'][0]['observation'] == trajectory()['messages'][1]['content']
    assert case['time']['start_seconds'] is None
    pairs[0]['arguments']['timestamps'][0] = 999
    assert case['evidence_refs'][0]['arguments']['timestamps'] == [12]
    for indices in ([True], [0], [1, 1], [2], []):
        with pytest.raises(ValueError):
            validate_reasoning_case(output() | {'evidence_indices': indices}, pairs, 'Question')
    failed = trajectory()
    failed['messages'][1]['content'] = 'Tool execution failed.'
    assert decision_pairs(failed) == []
    failed['messages'].reverse()
    assert decision_pairs(failed) == []


@pytest.mark.parametrize('question', [
    'Who did it?\n\nA. Alice\nB. Bob\nCurrent query time: DAY1',
    'Who did it?\n(A) Alice\n(B) Bob', 'Who did it?\nOutput requirement: Answer briefly.'])
def test_case_index_omits_options_and_runtime_boilerplate(question):
    assert case_question(question) == 'Who did it?'


def test_wrong_answers_cannot_become_positive_cases_and_labels_are_not_sent():
    model = Mock()
    reflector = Reflector(base_config(), model)
    item = {'answer_status': 'incorrect', 'trajectory': trajectory(),
            'question': 'Read the sign?\n(A) OPTION_ONLY_A\n(B) OPTION_ONLY_B'}
    assert reflector.reasoning_case(item) is None
    model.chat.assert_not_called()
    model.chat.side_effect = [{'content': json.dumps(output())},
                              {'content': '{"keep":true,"reason":"Original result resolved the sign text.",'
                                          '"checks":{"action":true,"scope":true,"conclusion":true}}'}]
    assert reflector.reasoning_case(item | {'answer_status': 'correct', 'reference_answer': 'SECRET'})
    payload = model.chat.call_args.args[0][0]['content']
    assert 'SECRET' not in payload and 'answer_status' not in payload
    assert 'OPTION_ONLY' not in payload and 'Read the sign?' in payload
    assert 'OPTION_ONLY_A' in model.chat.call_args_list[0].args[0][0]['content']


def test_correct_final_answer_does_not_override_rejected_case():
    model = Mock()
    model.chat.side_effect = [{'content': json.dumps(output())},
                             {'content': '{"keep":false,"reason":"The reported result remains ambiguous."}'}]
    item = {'answer_status': 'correct', 'trajectory': trajectory(), 'question': 'Read the sign?'}
    assert Reflector(base_config(), model).reasoning_case(item) is None


@pytest.mark.parametrize('check_name', ['action', 'scope', 'conclusion'])
@pytest.mark.parametrize('verdict', [False, None])
def test_case_keep_cannot_override_missing_support(check_name, verdict):
    checks = {'action': True, 'scope': True, 'conclusion': True, check_name: verdict}
    model = Mock()
    model.chat.side_effect = [{'content': json.dumps(output())}, {'content': json.dumps(
        {'keep': True, 'reason': 'Candidate claim exceeds the recorded observation.', 'checks': checks})}]
    item = {'answer_status': 'correct', 'trajectory': trajectory(), 'question': 'Read the sign?'}
    assert Reflector(base_config(), model).reasoning_case(item) is None
    assert model.chat.call_count == 2


def test_cases_cross_videos_but_facts_remain_local_and_future_ego_cases_are_excluded():
    source = source_context('lvbench', 'a') | {'sample_id': 'old'}
    context = source_context('lvbench', 'b') | {'sample_id': 'new'}
    metadata = {'source': source}
    assert memory_exclusion(metadata, context) == 'different_source'
    metadata['kind'] = 'reasoning_case'
    assert memory_exclusion(metadata, context) is None
    assert memory_exclusion(metadata, context | {'sample_id': 'old'}) == 'same_sample'
    metadata['source'] = source_context('egolife', '/data/A1_JAKE', {'date': 'DAY2', 'time': '10000000'})
    ego = source_context('egolife', '/other/A1_JAKE', {'date': 'DAY1', 'time': '10000000'})
    assert memory_exclusion(metadata, ego) == 'future_source'
    ego['available_at'] += 86400
    assert memory_exclusion(metadata, ego) is None


def test_pipeline_publishes_case_and_same_sample_fact_separately(tmp_path):
    source = tmp_path / 'source'
    write_source(source, count=2)
    manifest = tmp_path / 'train.json'
    manifest.write_text('[0,1]')
    reflector = FakeReflector()
    reflector.reasoning_case = lambda item: validate_reasoning_case(output(), decision_pairs(item['trajectory']), item['question']) | {'review': {'keep': True, 'reason': 'Fixture review'}}
    pipeline = EvolutionPipeline('lvbench', manifest, tmp_path / 'out', base_config(), source,
                                 reflector=reflector, workers=1)
    pipeline.run()
    store = EvolutionStateStore(pipeline.database_path)
    version = json.loads((pipeline.run_dir / 'final_version.json').read_text())['snapshot_version']
    memories = store.load_memories(version)
    assert len(memories) == 4
    assert {m['metadata']['kind'] for m in memories} == {'reasoning_case', 'observation'}
    cases = [m for m in memories if m['metadata']['kind'] == 'reasoning_case']
    assert all(c['metadata']['evidence_refs'][0]['message_index'] == 1 for c in cases)
    context = source_context('lvbench', 'new-video') | {'sample_id': 'new'}
    hits = store.search(version, 'Question', None, 3, -1, context)
    assert len(hits) == 1 and hits[0]['metadata']['kind'] == 'reasoning_case'
    queries = []
    store.embedding_function = lambda text: queries.append(text) or hash_embedding(text)
    hits = store.search(version, 'Question\n(A) EXIT\n(B) STOP', None, 3, -1,
                        source_context('lvbench', 'video0') | {'sample_id': 'new'})
    assert queries == ['Question\n(A) EXIT\n(B) STOP', 'Question']
    assert sum(h['metadata']['kind'] == 'reasoning_case' for h in hits) == 1
    assert any(h['metadata']['kind'] == 'observation' for h in hits)


def test_injection_labels_cases_as_historical_without_source_coordinates(monkeypatch):
    case = validate_reasoning_case(output(), decision_pairs(trajectory()), 'Read a sign?')
    store = Mock()
    store.search.return_value = [{'memory_id': 'case1', 'content': case['content'],
                                 'metadata': {'kind': 'reasoning_case', 'source': {'query_time': 'DO_NOT_TRANSFER'}}}]
    monkeypatch.setattr('videospy.evolving.storage.memory_store.MemoryStore', Mock(return_value=store))
    monkeypatch.setattr('videospy.agent.ModelClient', Mock())
    agent = VideoSpyAgent(agent_config(), 'video', None, '/tmp', [], video_reader=FakeVideoReader())
    prompt = agent._retrieve_memory_prompt('Current question')
    assert 'Historical reasoning case:' in prompt and 'DO_NOT_TRANSFER' not in prompt
    assert agent.memory_audit['ids'] == ['case1']
    assert len(prompt) <= 1600


@pytest.mark.parametrize('injected_client', [False, True])
def test_skill_review_uses_the_same_reflector_model(monkeypatch, injected_client):
    from config.videospy import init_config
    from argparse import Namespace
    monkeypatch.setenv('TEST_REFLECTOR_BASE', 'https://reflector.example/v1')
    monkeypatch.setenv('TEST_REFLECTOR_KEY', 'test-reflector-key')
    config = base_config()
    config['evolving']['reflector']['model'] = {
        'model_name': 'reflector-model', 'api_base': 'TEST_REFLECTOR_BASE', 'api_key': 'TEST_REFLECTOR_KEY'}
    config = init_config(config, Namespace())
    model = Mock()
    model.chat.side_effect = [{'content': 'null'}, {'content': '{"approved":true,"feedback":"Supported","instruction_excerpt":""}'}]
    factory = Mock(return_value=model)
    monkeypatch.setattr('videospy.evolving.reflector.model.ModelClient', factory)
    reflector = Reflector(config, model if injected_client else None)
    assert reflector.model_config['model_name'] == 'reflector-model'
    assert reflector.model_config['api_base'] == 'https://reflector.example/v1'
    assert reflector.model_config['api_key'] == 'test-reflector-key'
    reflector.memory({'source': {}, 'question': 'Question'}, {'messages': []})
    reflector.review_skill('task', {'instruction': '', 'sources': []})
    assert model.chat.call_count == 2
    if injected_client:
        factory.assert_not_called()
    else:
        factory.assert_called_once_with(reflector.model_config)
