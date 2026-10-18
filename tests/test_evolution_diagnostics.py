import json
from types import SimpleNamespace

import pytest

from benchmark.evolution_diagnostics import attach_evolution_usage, aggregate_evolution_usage, evolution_usage_notes


def trajectory(ids=(), reason='no_applicable_source', skipped=()):
    return {'finish_reason': 'reach_max_steps', 'diagnostics': {
        'skill': {'paragraphs': [{'task': 'retrieval', 'paragraph': 0}], 'characters': 123, 'budget_skipped': []},
        'memory': {'ids': list(ids), 'characters': 200 if ids else 0, 'budget_skipped': list(skipped),
                   'retrieval': {'empty_reason': reason, 'excluded': {'different_source': 77}}},
        'visual_calls': {'executed': 3, 'failed': 0, 'duplicates': 2}}}


CONFIG = {'_evolution': {'components': {'memory': True}}}


def test_zero_injection_reports_invalid_memory_ablation_and_does_not_hide_missing():
    record = {}
    attach_evolution_usage(record, trajectory(), CONFIG)
    summary = aggregate_evolution_usage(['1'], {'1': record})
    assert summary['memory']['injected_questions'] == 0
    assert summary['skill']['injected_questions'] == 1
    assert summary['reach_max_steps'] == 1 and summary['duplicate_visual_calls'] == 2
    assert summary['memory']['excluded_candidates'] == {'different_source': 77}
    assert '本次 Memory 未实际参与推理' in '\n'.join(evolution_usage_notes(summary))
    incomplete = aggregate_evolution_usage(['1', '2'], {'1': record, '2': {'status': 'error'}})
    assert incomplete['memory']['missing_questions'] == 1
    text = '\n'.join(evolution_usage_notes(incomplete))
    assert '诊断不完整' in text and '本次 Memory 未实际参与推理' not in text


def test_actual_injection_budget_rejection_and_retrieval_failure_are_distinct():
    records = {}
    for sid, tr in [('hit', trajectory(['m'], None)), ('budget', trajectory([], None, ['m'])),
                    ('failure', trajectory([], 'retrieval_failed'))]:
        records[sid] = {}
        attach_evolution_usage(records[sid], tr, CONFIG)
    summary = aggregate_evolution_usage(list(records), records)
    assert summary['memory']['injected_questions'] == 1
    assert summary['memory']['characters'] == 200
    assert summary['memory']['empty_reasons'] == {'prompt_budget': 1, 'retrieval_failed': 1}
    assert '本次 Memory 未实际参与推理' not in '\n'.join(evolution_usage_notes(summary))


@pytest.mark.parametrize('benchmark', ['lvbench', 'egolife', 'medvidbench'])
def test_real_benchmark_record_fields_and_reports_preserve_task_metrics(tmp_path, benchmark):
    # Import the existing benchmark fixtures, including optional dependency stubs.
    from tests.test_egolife import egolife
    from tests.test_benchmarks import lvbench, medvidbench
    from tests.test_evolving import write_source
    from benchmark.lvbench.utils import attach_task_metrics
    from benchmark.medvidbench.utils import attach_task_metrics as attach_medical_metrics
    records = write_source(tmp_path / 'source', benchmark, count=2)
    agent = SimpleNamespace(config=CONFIG, last_run_rounds=3,
                            last_run_token_usage={'total_tokens': 40, 'input_tokens': 30, 'output_tokens': 10})
    attach = attach_medical_metrics if benchmark == 'medvidbench' else attach_task_metrics
    for r in records:
        attach(r, agent, trajectory(), 2.)
    indexed = {r['sample_key']: r for r in records}
    output = tmp_path / 'reports'
    output.mkdir()
    if benchmark == 'lvbench':
        samples = [{'uid': r['sample_key'], 'question_types': r['question_types'], 'answer': r['answer']} for r in records]
        metrics = lvbench.compute_metrics(samples, indexed)
        lvbench.write_report(output, 'videospy', 'debug', 'test-model', samples, indexed, metrics)
    elif benchmark == 'egolife':
        metrics = egolife.compute_metrics(records, indexed)
        egolife.write_report(output, 'videospy', 'debug', 'test-model', records, indexed, metrics)
    else:
        metrics = {'nap_exact_acc': .5}
        medvidbench.write_report(output, 'videospy', 'debug', 'test-model', list(indexed), records, indexed, metrics)
        saved = json.loads((output / 'metrics.json').read_text())
        assert saved['task_metrics'] == metrics
        assert saved['evolution_usage']['memory']['injected_questions'] == 0
    report = (output / 'report.md').read_text()
    assert '| skill | 2/2 |' in report and '| memory | 0/2 |' in report
    assert '本次 Memory 未实际参与推理' in report
    assert all(r['agent_rounds'] == 3 and r['token_usage']['total_tokens'] == 40 for r in records)
