"""Summarize actual component injection without changing inference or task metrics."""
from collections import Counter


def attach_evolution_usage(record, trajectory, config):
    diagnostics = (trajectory or {}).get('diagnostics') or {}
    requested = (config.get('_evolution') or {}).get('components', {}).get('memory')
    if requested is None:
        requested = config.get('tools', {}).get('memory_search', {}).get('enabled', False)
    usage = {'memory_requested': bool(requested), 'skill': None, 'memory': None,
             'finish_reason': (trajectory or {}).get('finish_reason'),
             'visual_calls': diagnostics.get('visual_calls')}
    for component, key in (('skill', 'paragraphs'), ('memory', 'ids')):
        item = diagnostics.get(component)
        if isinstance(item, dict) and isinstance(item.get(key), list) and isinstance(item.get('characters'), int):
            usage[component] = {'count': len(item[key]), 'characters': item['characters'],
                                'budget_skipped': len(item.get('budget_skipped', []))}
            if component == 'memory':
                retrieval = item.get('retrieval') or {}
                reason = retrieval.get('empty_reason')
                if not item[key] and item.get('budget_skipped'):
                    reason = 'prompt_budget'
                elif not item[key] and reason == 'disabled' and requested:
                    reason = 'no_memories'
                usage[component].update(empty_reason=(reason or 'unknown') if not item[key] else None,
                                        excluded=retrieval.get('excluded', {}))
    record['evolution_usage'] = usage


def aggregate_evolution_usage(keys, records):
    rows = [(records.get(str(key)) or {}).get('evolution_usage') or {} for key in keys]
    summary = {'questions': len(keys)}
    for component in ('skill', 'memory'):
        measured = [r[component] for r in rows if isinstance(r.get(component), dict)]
        summary[component] = {
            'measured_questions': len(measured), 'missing_questions': len(rows) - len(measured),
            'injected_questions': sum(r['count'] > 0 for r in measured),
            'injected_items': sum(r['count'] for r in measured),
            'characters': sum(r['characters'] for r in measured),
            'budget_skipped': sum(r['budget_skipped'] for r in measured),
        }
        if component == 'memory':
            excluded = Counter()
            for r in measured:
                excluded.update(r.get('excluded', {}))
            summary[component].update(
                requested_questions=sum(r.get('memory_requested') is True for r in rows),
                empty_reasons=dict(Counter(r['empty_reason'] for r in measured if not r['count'])),
                excluded_candidates=dict(excluded))
    summary['reach_max_steps'] = sum(r.get('finish_reason') == 'reach_max_steps' for r in rows)
    summary['missing_finish_reason'] = sum(not r.get('finish_reason') for r in rows)
    visual = [r['visual_calls'] for r in rows if isinstance(r.get('visual_calls'), dict)
              and isinstance(r['visual_calls'].get('duplicates'), int)]
    summary['duplicate_visual_calls'] = sum(r.get('duplicates', 0) for r in visual)
    summary['missing_visual_diagnostics'] = len(rows) - len(visual)
    return summary


def evolution_usage_notes(summary):
    lines = ['## Actual component usage', '',
             '| Component | Injected questions | Skill paragraphs / memories injected | Characters | Missing diagnostics |',
             '| --- | --- | --- | --- | --- |']
    for component in ('skill', 'memory'):
        value = summary[component]
        lines.append(f"| {component} | {value['injected_questions']}/{summary['questions']} | "
                     f"{value['injected_items']} | {value['characters']} | {value['missing_questions']} |")
    memory = summary['memory']
    lines += ['', f"Memory not injected: {memory['empty_reasons']}",
              f"Excluded memory candidates: {memory['excluded_candidates']}",
              f"Memory items skipped by prompt budget: {memory['budget_skipped']}",
              f"Reached max steps: {summary['reach_max_steps']}; repeated visual calls: {summary['duplicate_visual_calls']}",
              f"Missing finish reason: {summary['missing_finish_reason']}; missing visual diagnostics: {summary['missing_visual_diagnostics']}"]
    if not summary['skill']['injected_questions'] and not summary['skill']['missing_questions']:
        lines += ['', '本次 Skill 未实际注入任何问题。']
    if memory['requested_questions'] and not memory['injected_questions']:
        lines += ['', ('Memory 诊断不完整，尚不能确认是否全部未注入。' if memory['missing_questions'] else
                       '本次 Memory 未实际参与推理，不能据此判断 Memory 收益或与 Skill 的交互效果。')]
    return lines
