"""Document accumulation, evidence review and immutable offline publication."""
import copy
import json
import sqlite3
from unittest.mock import Mock

import pytest

from tests.test_evolving import FakeReflector, base_config, write_source
from videospy.evolved_config import format_skills
from videospy.evolving.optimizer.skill_optimizer import (
    SkillOptimizer, validate_task_document, validate_domain_document, validate_document_review, skill_input,
)
from videospy.evolving.pipeline import EvolutionPipeline
from videospy.evolving.reflector import Reflector, ReflectionRunner
from videospy.evolving.storage.memory_store import EvolutionStateStore


def sample(sid='a', status='correct'):
    return {'sample_id': sid, 'question': 'Which detail distinguishes the options?',
            'prediction': 'A', 'answer_status': status, 'evidence_support': 'sufficient',
            'trajectory': {'messages': [
                {'role': 'assistant', 'content': 'Inspect the unresolved detail.'},
                {'role': 'tool', 'content': 'The label reads EXIT.', 'tool_call_id': 't'},
                {'role': 'assistant', 'content': 'A'}]}}


def draft(text='Locate the relevant event, then inspect the detail distinguishing the options.', sid='a', index=1):
    return {'instruction': text, 'sources': [{'sample_id': sid, 'message_indices': [index]}] if text else []}


def document(text='Inspect the distinguishing detail.'):
    return {'format': 'document', 'instruction': text, 'sources': [], 'source_tasks': [], 'review': []}


def approve():
    return {'approved': True, 'feedback': 'The supplied observations support these methods.', 'instruction_excerpt': ''}


def no_update(kind='covered_by_current', reason='The current label-inspection method covers the same decision in this group.'):
    return {'instruction': None, 'sources': [], 'no_update': {'kind': kind, 'reason': reason}}


def model(*values):
    client = Mock()
    client.chat.side_effect = [value if isinstance(value, Exception) else {'content': json.dumps(value)} for value in values]
    return client


def direct(stage, path, inputs, producer):
    return producer()


def pipeline(tmp_path, reflector=None, config=None):
    source = tmp_path / 'source'
    write_source(source, count=4)
    manifest = tmp_path / 'train.json'
    manifest.write_text('[0,1,2,3]')
    return EvolutionPipeline('lvbench', manifest, tmp_path / 'out', config or base_config(), source,
                             reflector=reflector or FakeReflector(), run_dir=tmp_path / 'run')


@pytest.mark.parametrize('index', [-1, 9, True, '1'])
def test_sources_must_identify_original_messages(index):
    with pytest.raises(ValueError):
        validate_task_document(draft(index=index), [sample()])


def test_sources_bind_complete_original_evidence_without_model_quotes():
    item = sample()
    item['trajectory']['messages'][0]['reasoning_content'] = 'The query needs a local distinction.'
    doc = validate_task_document(draft(index=0), [item])
    assert doc['sources'][0]['trajectory'] == skill_input(item)['trajectory']
    assert doc['sources'][0]['trajectory'] is not item['trajectory']
    with pytest.raises(ValueError):
        validate_task_document(draft(sid='invented'), [item])
    proposed = draft()
    proposed['sources'][0]['quote'] = 'invented'
    with pytest.raises(ValueError):
        validate_task_document(proposed, [item])
    item['trajectory']['messages'][1]['role'] = 'user'
    with pytest.raises(ValueError):
        validate_task_document(draft(), [item])


def test_revisions_cannot_introduce_new_sources_or_message_indices():
    doc = validate_task_document(draft(), [sample()])
    for invalid in (draft(sid='new'), draft(index=0)):
        client = model(invalid, invalid)
        with pytest.raises(ValueError):
            Reflector(base_config(), client).revise_skill('task', doc, 'Narrow the claim')
        assert client.chat.call_count == 2


def test_correct_and_wrong_trajectories_remain_in_input_even_with_case():
    positive, negative = sample(), sample('b', 'incorrect')
    positive['reasoning_case'] = {'content': 'CASE_ONLY_MARKER'}
    client = model(no_update('no_reusable_experience', 'The isolated label observation gives no additional reusable decision.'))
    reflector = Reflector(base_config(), client)
    SkillOptimizer(reflector, direct).optimize_task('task', [[positive, negative]])
    prompt = client.chat.call_args.args[0][1]['content']
    payload = json.loads(prompt.split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert [item['answer_status'] for item in payload['trajectories']] == ['correct', 'incorrect']
    assert all('trajectory' in item for item in payload['trajectories'])
    assert 'CASE_ONLY_MARKER' not in prompt
    assert base_config()['SYSTEM_PROMPT'] in prompt


def test_complete_document_updates_support_rewrite_deletion_and_no_update():
    first = validate_task_document(draft('Earlier useful method.'), [sample()])
    assert validate_task_document(None, [], first) is None
    revised = validate_task_document(draft('Corrected method with a narrower scope.'), [], first)
    assert revised['instruction'] != first['instruction']
    assert first['instruction'] == 'Earlier useful method.'
    empty = validate_task_document(draft(''), [], first)
    assert empty['instruction'] == '' and empty['sources'] == []
    client = model(draft(), draft(''))
    assert SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()], [sample('b')]]) is None
    assert client.chat.call_count == 2


@pytest.mark.parametrize('revisions,approved', [(1, True), (2, True), (3, True), (3, False)])
def test_review_allows_up_to_three_revisions_and_requires_reapproval(revisions, approved):
    texts = ['Broad unsupported advice.'] + [f'Inspect the visible label with scope {i}.' for i in range(revisions)]
    responses = [draft(texts[0])]
    for index, text in enumerate(texts):
        passed = approved and index == revisions
        responses.append({'approved': passed, 'feedback': 'Check the scope against observations.',
                          'instruction_excerpt': '' if passed else text})
        if index < revisions:
            responses.append(draft(texts[index + 1]))
    client = model(*responses)
    paths = []

    def checkpoint(stage, path, inputs, producer):
        paths.append(path)
        return producer()

    result = SkillOptimizer(Reflector(base_config(), client), checkpoint).optimize_task('task', [[sample()]])
    assert client.chat.call_count == 2 + 2 * revisions
    assert len(paths) == len(set(paths))
    if approved:
        assert result['instruction'] == texts[-1]
        assert [r['approved'] for r in result['review']] == [False] * revisions + [True]
    else:
        assert result is None
    payload = json.loads(client.chat.call_args.args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert payload['candidate']['instruction'] == texts[-1]
    assert 'The label reads EXIT.' in json.dumps(payload)


def test_unchanged_revision_stops_without_repeating_review():
    original = draft()
    original['sources'][0]['message_indices'] = [0, 1]
    unchanged = copy.deepcopy(original)
    unchanged['sources'][0]['message_indices'].reverse()
    client = model(original, {'approved': False, 'feedback': 'Unsupported method.',
                             'instruction_excerpt': original['instruction']}, unchanged)
    assert SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()]]) is None
    assert client.chat.call_count == 3


def test_changed_references_require_review_even_when_text_is_unchanged():
    original = draft()
    original['sources'][0]['message_indices'] = [0, 1]
    client = model(original, {'approved': False, 'feedback': 'Keep only the observation reference.',
                             'instruction_excerpt': original['instruction']}, draft(), approve())
    result = SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()]])
    assert result['sources'][0]['message_indices'] == [1]
    assert client.chat.call_count == 4


def test_local_revision_preserves_supported_content():
    supported = 'Inspect a located label when text distinguishes the options.'
    original = supported + '\n\nAlways stop after one overview.'
    client = model(draft(original), {'approved': False, 'feedback': 'Remove the unsupported stopping claim.',
                                    'instruction_excerpt': 'Always stop after one overview.'},
                   draft(supported), approve())
    result = SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()]])
    assert result['instruction'] == supported
    assert result['review'][-1]['approved']


def test_empty_revision_stops_without_further_review():
    original = draft()
    client = model(original, {'approved': False, 'feedback': 'No supported methods.',
                             'instruction_excerpt': original['instruction']}, draft(''))
    assert SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()]]) is None
    assert client.chat.call_count == 3


def test_compaction_happens_before_review_without_cutting_text():
    client = model(draft('Useful context. ' * 150), {'instruction': 'Inspect the distinguishing detail.'}, approve())
    result = SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()]])
    assert result['instruction'] == 'Inspect the distinguishing detail.'
    assert client.chat.call_count == 3
    compact_input = json.loads(client.chat.call_args_list[1].args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert set(compact_input) == {'task_type', 'instruction', 'budget'}
    assert result['sources'][0]['message_indices'] == [1]
    with pytest.raises(ValueError, match='3600'):
        validate_task_document(draft('x' * 3601), [sample()])


def test_long_revision_is_compacted_and_rechecked_before_publication():
    long_revision = 'Inspect the distinguishing detail. ' * 70
    client = model(draft(), {'approved': False, 'feedback': 'Narrow this method.',
                             'instruction_excerpt': draft()['instruction']},
                   draft(long_revision), {'instruction': 'Inspect the distinguishing detail.'}, approve())
    result = SkillOptimizer(Reflector(base_config(), client), direct).optimize_task('task', [[sample()]])
    assert result['instruction'] == 'Inspect the distinguishing detail.'
    assert client.chat.call_count == 5
    assert [decision['approved'] for decision in result['review']] == [False, True]
    review_payload = json.loads(client.chat.call_args.args[0][1]['content'].split('\nInput:\n')[1]
                                .split('\n\nEnd of immutable input.')[0])
    assert review_payload['candidate']['instruction'] == result['instruction']


@pytest.mark.parametrize('value', [None, {}, {'approved': 'true', 'feedback': 'bad'}, {'approved': True, 'feedback': ''}])
def test_review_requires_an_explicit_typed_decision(value):
    with pytest.raises(ValueError):
        validate_document_review(value, 'Candidate methods.')


def test_review_format_retries_once_but_context_overflow_does_not():
    doc = validate_task_document(draft(), [sample()])
    client = model({}, approve())
    assert Reflector(base_config(), client).review_skill('task', doc)['approved']
    assert client.chat.call_count == 2
    client = model(RuntimeError('context window exceeded'))
    with pytest.raises(RuntimeError):
        Reflector(base_config(), client).review_skill('task', doc)
    assert client.chat.call_count == 1


def test_domain_requires_approved_cross_task_sources_and_is_reviewed():
    task = validate_task_document(draft(), [sample()]) | {'review': [approve()]}
    skills = {'retrieval': task, 'reasoning': copy.deepcopy(task)}
    with pytest.raises(ValueError, match='two source tasks'):
        validate_domain_document({'instruction': 'Common method.', 'source_tasks': ['retrieval']}, skills)
    with pytest.raises(ValueError, match='approved'):
        validate_domain_document({'instruction': 'Common method.', 'source_tasks': list(skills)},
                                 skills | {'reasoning': task | {'review': []}})
    candidate = {'instruction': 'Inspect a located label when text matters across these tasks.', 'source_tasks': list(skills)}
    client = model(candidate, {'approved': False, 'feedback': 'The shared scope is unsupported.',
                                'instruction_excerpt': 'Inspect a located label'},
                   {'instruction': '', 'source_tasks': []})
    assert SkillOptimizer(Reflector(base_config(), client), direct).optimize_domain('lvbench', skills) is None
    assert client.chat.call_count == 3
    review_input = json.loads(client.chat.call_args_list[1].args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert set(review_input['task_skills']) == set(skills)
    assert review_input['readonly_evidence'][0]['trajectory'] == skill_input(sample())['trajectory']


def test_injection_round_robins_whole_paragraphs_and_hides_sources():
    skills = {'first': document('A' * 1000 + '\n\n' + 'B' * 1000),
              'second': document('C' * 1000 + '\n\n' + 'D' * 1000),
              '__domain__': document('# Shared method\n\nUse a matching method.')}
    skills['first']['sources'] = validate_task_document(draft(), [sample()])['sources']
    skills['first']['sources'][0]['private_source_marker'] = 'SECRET'
    text, audit = format_skills({'TASK_SKILLS': skills}, ['first', 'second', 'first'])
    assert 'A' * 1000 in text and 'C' * 1000 in text
    assert 'B' not in text and 'D' * 1000 not in text
    assert '# Shared method\n\nUse a matching method.' in text
    assert 'SECRET' not in text and len(text) <= 3000
    assert len(audit['budget_skipped']) == 2
    assert len(audit['paragraphs']) == 3


@pytest.mark.parametrize('legacy', ['old text', [{'rules': []}], {'instruction': 'old instruction'},
                                    document() | {'execution': {'stop_after_duplicates': 2}}])
def test_legacy_skills_are_rejected_instead_of_silently_ignored(legacy):
    with pytest.raises(ValueError, match='Legacy Skill'):
        format_skills({'TASK_SKILLS': {'task': legacy}}, ['task'])


def test_schema_seven_and_old_database_rejection_without_mutation(tmp_path):
    path = tmp_path / 'old.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=6')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Regenerate'):
        EvolutionStateStore(path, read_only=True).validate()
    assert path.read_bytes() == before
    run = pipeline(tmp_path)
    run.run()
    with sqlite3.connect(run.database_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 7
        columns = {r[1] for r in db.execute('PRAGMA table_info(skills)')}
    assert {'sources', 'source_tasks', 'review', 'instruction'} <= columns
    assert not {'rules', 'selection', 'updates'} & columns


def test_review_failure_preserves_memory_and_resume_retries_only_failed_review(tmp_path):
    reflector = FakeReflector({('review', 'retrieval'), ('review', 'reasoning')})
    run = pipeline(tmp_path, reflector)
    run.run()
    assert run.result['memory_count'] == 4
    assert run.result['evaluation_version'] is None
    assert run.result['evaluation_blockers']
    assert json.loads((run.run_dir / 'skills.json').read_text()) == {}
    reflector.calls.clear()
    reflector.failures.clear()
    run.run()
    assert set(reflector.calls) == {('review', 'retrieval'), ('review', 'reasoning'),
                                    ('domain', 'lvbench'), ('review', '__domain__')}
    version = run.result['version_id']
    reflector.calls.clear()
    run.run()
    assert not reflector.calls and run.result['version_id'] == version


def test_rejected_documents_are_valid_empty_results_not_request_failures(tmp_path):
    reflector = FakeReflector()
    reflector.review_skill = Mock(return_value={'approved': False, 'feedback': 'Unsupported generalization.'})
    run = pipeline(tmp_path, reflector)
    run.run()
    assert run.result['status'] == 'complete' and run.result['skill_count'] == 0
    assert not run.result['evaluation_blockers']
    assert run.result['memory_count'] == 4
    assert reflector.review_skill.call_count == 2
    assert run.result['stages']['skill_review']['rejected'] == 2


def test_changed_base_prompt_invalidates_review_checkpoints(tmp_path):
    source = tmp_path / 'source'
    records = write_source(source, count=2)
    reflector = FakeReflector()
    root = tmp_path / 'reflection'
    ReflectionRunner(reflector, base_config()).run(root, source, records, {})
    reflector.calls.clear()
    config = base_config() | {'SYSTEM_PROMPT': 'New base instructions.'}
    ReflectionRunner(reflector, config).run(root, source, records, {})
    assert ('review', 'retrieval') in reflector.calls


def test_json_duplicate_fields_are_rejected_and_corrected():
    client = Mock()
    client.chat.side_effect = [{'content': '{"instruction":"one","instruction":"two","sources":[]}'},
                              {'content': json.dumps(draft())}]
    result = Reflector(base_config(), client).update_skill('task', [sample()], None)
    assert result['instruction'] == draft()['instruction'] and client.chat.call_count == 2


def test_old_run_rejected_before_any_artifact_is_written(tmp_path):
    run = pipeline(tmp_path)
    run.run_dir.mkdir()
    with sqlite3.connect(run.database_path) as db:
        db.execute('PRAGMA user_version=6')
    (run.run_dir / 'evolution.log').write_text('Historical log must remain unchanged.')
    before = {p.name: p.read_bytes() for p in run.run_dir.iterdir()}
    with pytest.raises(ValueError, match='Regenerate'):
        run.run()
    assert {p.name: p.read_bytes() for p in run.run_dir.iterdir()} == before
    assert not run.reflector.calls


def test_legacy_initial_skill_rejected_before_run_creation(tmp_path):
    config = base_config() | {'TASK_SKILLS': {'task': [{'rules': []}]}}
    run = pipeline(tmp_path, config=config)
    with pytest.raises(ValueError, match='Legacy Skill'):
        run.run()
    assert not run.run_dir.exists() and not run.reflector.calls


def test_old_reflection_checkpoints_are_not_reused_or_modified(tmp_path):
    run = pipeline(tmp_path)
    old = tmp_path / 'old-reflection'
    path = old / 'checkpoints/skills/tasks/retrieval/group_000.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'identity': 'old-v20', 'result': {'rules': []}}))
    before = path.read_bytes()
    run.reuse_reflection_dir = old
    run.run()
    assert path.read_bytes() == before
    assert ('skill', 'retrieval') in run.reflector.calls
    assert run.result['stages']['task_skill']['reused'] == 0


def test_case_failure_does_not_change_task_learning_or_factual_memory(tmp_path):
    reflector = FakeReflector()
    reflector.reasoning_case = Mock(side_effect=RuntimeError('Case extraction unavailable'))
    run = pipeline(tmp_path, reflector)
    run.run()
    assert run.result['skill_count'] == 3 and run.result['memory_count'] == 4
    assert run.result['stages']['reasoning_case']['failed'] == 4
    assert run.result['stages']['task_skill']['failed'] == 0


@pytest.mark.parametrize('raw', ['# Useful methods\nInspect the distinguishing detail.',
                                '{"instruction":"unfinished',
                                json.dumps(draft('Markdown, <=3600 characters')),
                                json.dumps(draft('x' * 3601)),
                                json.dumps({'instruction': 'Useful method.', 'sources': [sample()]}),
                                json.dumps(draft(index=99))])
def test_real_output_failure_patterns_are_corrected_without_losing_methods(raw):
    client = Mock()
    client.chat.side_effect = [{'content': raw}, {'content': json.dumps(draft())}]
    result = Reflector(base_config(), client).update_skill('task', [sample()], None)
    assert result['instruction'] == draft()['instruction']
    assert [call.kwargs['return_json'] for call in client.chat.call_args_list] == [False, False]
    correction = client.chat.call_args.args[0]
    assert correction[2]['content'] == raw
    assert 'actual guide in instruction' in correction[3]['content']


def test_placeholder_after_format_correction_fails_and_retains_previous_draft():
    current = validate_task_document(draft('Earlier supported method.'), [sample()])
    client = Mock()
    client.chat.side_effect = [{'content': '# New method'},
                              {'content': json.dumps(draft('Markdown, <=3600 characters'))},
                              {'content': json.dumps(approve())}]
    errors = []
    def checkpoint(stage, path, inputs, producer):
        try:
            return producer()
        except ValueError as error:
            errors.append((stage, str(error)))
            return None
    result = SkillOptimizer(Reflector(base_config(), client), checkpoint).optimize_task('task', [[sample('b')]], current)
    assert result['instruction'] == current['instruction']
    assert errors[0][0] == 'task_skill' and 'placeholder' in errors[0][1]


def test_json_no_update_and_revision_cannot_return_no_update():
    current = validate_task_document(draft(), [sample()])
    unchanged = no_update()
    assert Reflector(base_config(), model(unchanged)).update_skill('task', [sample()], current) is None
    with pytest.raises(ValueError, match='Revision must'):
        Reflector(base_config(), model(unchanged, unchanged)).revise_skill('task', current, 'Condense')


def test_invalid_reference_feedback_contains_sample_index_and_allowed_indices():
    item = sample()
    item['trajectory']['messages'].append({'role': 'user', 'content': 'Now answer.'})
    with pytest.raises(ValueError, match=r'Sample a, index 3:.*allowed indices: \[0, 1, 2\]'):
        validate_task_document(draft(index=3), [item])


def test_review_rejects_source_excerpt_and_keeps_evidence_immutable():
    doc = validate_task_document(draft('Check the label before choosing an option.'), [sample()])
    original = copy.deepcopy(doc)
    invalid = {'approved': False, 'feedback': 'Change the observation.', 'instruction_excerpt': 'The label reads EXIT.'}
    client = model(invalid, approve())
    assert Reflector(base_config(), client).review_skill('task', doc)['approved']
    payload = json.loads(client.chat.call_args.args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert payload['candidate'] == {'instruction': doc['instruction'], 'sources': draft()['sources']}
    assert 'trajectory' not in json.dumps(payload['candidate'])
    assert payload['readonly_evidence'][0]['trajectory'] == doc['sources'][0]['trajectory']
    assert doc == original


def test_group_input_retains_prior_methods_and_deduplicates_evidence():
    first = validate_task_document(draft(), [sample()])
    client = model(no_update())
    reflector = Reflector(base_config(), client)
    reflector.update_skill('task', [sample('b', 'incorrect')], first)
    payload = json.loads(client.chat.call_args.args[0][1]['content'].split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert payload['current_skill']['instruction'] == first['instruction']
    assert payload['trajectories'][0]['allowed_message_indices'] == [0, 1, 2]
    assert payload['readonly_evidence'][0]['allowed_message_indices'] == [1]
    assert payload['trajectories'][0]['answer_status'] == 'incorrect'


def test_json_endpoint_missing_prefilled_opening_brace_is_losslessly_recovered():
    client = Mock()
    original = draft()
    client.chat.return_value = {'content': json.dumps(original)[1:]}
    result = Reflector(base_config(), client).update_skill('task', [sample()], None)
    assert result['instruction'] == original['instruction']
    assert result['sources'][0]['message_indices'] == original['sources'][0]['message_indices']
    assert client.chat.call_count == 1


def test_missing_opening_brace_does_not_hide_incomplete_or_duplicate_fields():
    from videospy.evolving.reflector.model import _skill_json
    for text in ['"instruction":"unfinished', '"instruction":"one","instruction":"two","sources":[]}']:
        with pytest.raises(ValueError):
            _skill_json(text)


def test_empty_json_mode_response_uses_the_existing_correction_without_weakening_validation():
    client = Mock()
    client.chat.side_effect = [{'role': 'assistant'}, {'content': json.dumps(draft())}]
    result = Reflector(base_config(), client).update_skill('task', [sample()], None)
    assert result['instruction'] == draft()['instruction']
    assert [call.kwargs['return_json'] for call in client.chat.call_args_list] == [False, False]
    initial = client.chat.call_args_list[0].args[0]
    assert [m['role'] for m in initial] == ['system', 'user']
    assert 'The label reads EXIT.' not in initial[0]['content']
    assert 'The label reads EXIT.' in initial[1]['content']


def test_review_keeps_candidate_after_evidence_and_repairs_wrong_excerpt():
    item = sample(status='incorrect')
    item['trajectory']['messages'][0]['content'] = 'Guess from option length.'
    instruction = 'Do not treat option length as observed evidence.'
    doc = validate_task_document(draft(instruction, index=0), [item])
    client = model({'approved': False, 'feedback': 'Historical guess was wrong.',
                    'instruction_excerpt': 'Guess from option length.'}, approve())
    result = Reflector(base_config(), client).review_skill('task', doc)
    assert result['approved']
    initial = client.chat.call_args_list[0].args[0][1]['content']
    payload = json.loads(initial.split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert list(payload)[-1] == 'candidate'
    assert payload['readonly_evidence'][0]['answer_status'] == 'incorrect'
    assert payload['candidate']['instruction'] == instruction
    correction = client.chat.call_args.args[0][-1]['content']
    assert instruction in correction
    assert doc['sources'][0]['trajectory']['messages'][0]['content'] == 'Guess from option length.'


def test_groups_revise_both_parts_without_losing_unrelated_supported_experience():
    first = ('**Recommended approach:** Inspect the located label.\n\n'
             '**Pitfalls to avoid:** Do not infer text from the object shape.')
    second = ('**Recommended approach:** Inspect a sharper neighboring frame when the label is blurred.\n\n'
              '**Pitfalls to avoid:** Do not infer text from the object shape or guess from blurred letters.')
    revised = draft(second)
    revised['sources'].append({'sample_id': 'b', 'message_indices': [1]})
    client = model(draft(first), revised, no_update(), approve())
    result = SkillOptimizer(Reflector(base_config(), client), direct).optimize_task(
        'task', [[sample()], [sample('b')], [sample('c')]])
    assert result['instruction'] == second
    assert {s['sample_id'] for s in result['sources']} == {'a', 'b'}
    for index, expected in [(1, first), (2, second)]:
        text = client.chat.call_args_list[index].args[0][1]['content']
        payload = json.loads(text.split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
        assert payload['current_skill']['instruction'] == expected
        assert payload['readonly_evidence']
    assert result['review'][-1]['approved']


@pytest.mark.parametrize('empty', [None, {'instruction': None, 'sources': []},
                                 {'instruction': 'null', 'sources': []}, draft(''), draft('   '),
                                 no_update()])
def test_initial_unexplained_empty_uses_one_correction_to_create_skill(empty):
    client = model(empty, draft())
    result = Reflector(base_config(), client).update_skill('task', [sample()], None)
    assert result['instruction'] == draft()['instruction']
    assert client.chat.call_count == 2
    initial = client.chat.call_args_list[0].args[0][1]['content']
    payload = json.loads(initial.split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
    assert payload['learning_mode'] == 'create' and payload['current_skill'] is None
    assert 'No working Skill exists' in initial


@pytest.mark.parametrize('decision', [None, {}, {'kind': 'covered_by_current', 'reason': ''},
                                    {'kind': 'unknown', 'reason': 'A reason'},
                                    {'kind': 'no_reusable_experience', 'reason': '   '}])
def test_malformed_no_update_is_not_accepted(decision):
    invalid = {'instruction': None, 'sources': [], 'no_update': decision}
    client = model(invalid, invalid)
    with pytest.raises(ValueError, match='no_update requires'):
        Reflector(base_config(), client).update_skill('task', [sample()], None)
    assert client.chat.call_count == 2


def test_no_reusable_experience_is_allowed_with_auditable_reason(caplog):
    reason = 'Sample a contains only a failed tool observation and no interpretable decision.'
    client = model(no_update('no_reusable_experience', reason))
    with caplog.at_level('INFO'):
        assert Reflector(base_config(), client).update_skill('task', [sample()], None) is None
    assert client.chat.call_count == 1
    assert 'mode=create kind=no_reusable_experience' in caplog.text
    assert reason in caplog.text


def test_repeated_unexplained_no_update_fails_without_overwriting_prior_draft(tmp_path):
    from threading import Lock
    from videospy.evolving.common.schemas import empty_stage

    current = validate_task_document(draft(), [sample()])
    invalid = {'instruction': None, 'sources': []}
    client = model(invalid, invalid, approve())
    reflector = Reflector(base_config(), client)
    runner = ReflectionRunner(reflector, base_config())
    runner.directory, runner.lock, runner.progress = tmp_path, Lock(), {}
    runner.stages = {name: empty_stage() for name in ('task_skill', 'skill_review')}
    runner.failures = []
    result = SkillOptimizer(reflector, runner._checkpoint).optimize_task('task', [[sample('b')]], current)
    assert result['instruction'] == current['instruction']
    assert result['review'][-1]['approved']
    assert runner.stages['task_skill']['failed'] == 1
    assert runner.stages['task_skill']['empty'] == 0
    assert len(runner.failures) == 1
    assert client.chat.call_count == 3


def test_explained_no_update_does_not_prevent_later_creation_and_accumulation():
    updated = draft('Verify the label and distinguish it from neighboring objects.')
    updated['sources'].append({'sample_id': 'b', 'message_indices': [1]})
    client = model(no_update('no_reusable_experience', 'This group contains no supported reusable decision.'),
                   draft(), updated, no_update(), approve())
    result = SkillOptimizer(Reflector(base_config(), client), direct).optimize_task(
        'task', [[sample('empty')], [sample()], [sample('b')], [sample('c')]])
    assert result['instruction'] == updated['instruction']
    assert {s['sample_id'] for s in result['sources']} == {'a', 'b'}
    assert result['review'][-1]['approved']
    for i, mode in enumerate(['create', 'create', 'revise', 'revise']):
        content = client.chat.call_args_list[i].args[0][1]['content']
        payload = json.loads(content.split('\nInput:\n')[1].split('\n\nEnd of immutable input.')[0])
        assert payload['learning_mode'] == mode
