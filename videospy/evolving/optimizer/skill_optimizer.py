"""Accumulate task documents and publish only evidence-reviewed methods."""
import copy
import re
from urllib.parse import quote

from ..common.schemas import (
    WORK_SKILL_CHARS, TASK_SKILL_CHARS, DOMAIN_SKILL_CHARS, validate_skill_document,
)

DOMAIN_TASK = '__domain__'
MAX_SKILL_REVISIONS = 3


def allowed_indices(messages):
    return [i for i, message in enumerate(messages)
            if message.get('role') in {'assistant', 'tool'}
            and any(message.get(k) for k in ('content', 'reasoning_content', 'reasoning', 'tool_calls'))]


def skill_input(item):
    # Cases and reference answers do not replace the original execution evidence.
    result = {key: copy.deepcopy(item[key]) for key in (
        'sample_id', 'question', 'prediction', 'answer_status', 'answer_status_source',
        'evidence_support', 'score', 'reasoning_sound', 'reason', 'outcome', 'cost', 'source', 'trajectory',
    ) if key in item}
    result['trajectory']['messages'] = [dict(message, message_index=index)
        for index, message in enumerate(result['trajectory']['messages'])]
    result['allowed_message_indices'] = allowed_indices(result['trajectory']['messages'])
    return result


def _instruction(value, budget):
    if not isinstance(value, dict) or not isinstance(value.get('instruction'), str):
        raise ValueError('Return a document object with an instruction string, or null for no update')
    text = value['instruction'].strip()
    if text and (re.fullmatch(r'Markdown(?:\s*,?\s*<=?\s*\d+\s*characters)?', text, re.I)
                 or text.casefold() in {'...', '<instruction>', 'your instruction here', 'todo'}):
        raise ValueError('instruction contains a placeholder, not methods. Put the actual supported Markdown guide in instruction.')
    if len(text) > budget:
        raise ValueError(f'Skill instruction exceeds {budget} characters (actual {len(text)}); condense whole ideas, never cut sentences')
    return text


def validate_task_document(value, group, current=None, budget=WORK_SKILL_CHARS):
    if value is None or value == {'instruction': None, 'sources': []}:
        return None
    text = _instruction(value, budget)
    if set(value) != {'instruction', 'sources'}:
        raise ValueError('Task document must contain instruction and sources only')
    refs = value.get('sources')
    if not isinstance(refs, list) or (bool(text) != bool(refs)):
        raise ValueError('Nonempty Skill text requires sources; an explicitly empty document must have no sources')
    previous = {s['sample_id']: s for s in (current or {}).get('sources', [])}
    samples = {item['sample_id']: skill_input(item) for item in group}
    sources, seen = [], set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {'sample_id', 'message_indices'}:
            raise ValueError('Each source requires sample_id and message_indices only')
        sid, indices = ref['sample_id'], ref['message_indices']
        if not isinstance(sid, str) or sid in seen or sid not in samples.keys() | previous.keys():
            raise ValueError('Source must identify a distinct supplied sample')
        if (not isinstance(indices, list) or not indices
                or any(not isinstance(i, int) or isinstance(i, bool) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('message_indices must be distinct integer indices')
        item = samples.get(sid, previous.get(sid))
        messages = item['trajectory']['messages']
        allowed = allowed_indices(messages) if sid in samples else previous[sid]['message_indices']
        for index in indices:
            if (not 0 <= index < len(messages) or messages[index].get('role') not in {'assistant', 'tool'}
                    or not any(messages[index].get(k) for k in ('content', 'reasoning_content', 'reasoning', 'tool_calls'))):
                raise ValueError(f'Sample {sid}, index {index}: cite an original readable assistant decision or tool observation; allowed indices: {allowed}')
            if sid not in samples and index not in previous[sid]['message_indices']:
                raise ValueError(f'Sample {sid}, index {index}: revision cannot introduce evidence outside the supplied source references; allowed indices: {allowed}')
        sources.append({**copy.deepcopy(item), 'message_indices': sorted(indices)})
        seen.add(sid)
    return {'format': 'document', 'instruction': text, 'sources': sources,
            'source_tasks': [], 'review': []}


def validate_domain_document(value, skills):
    if value is None or value == {'instruction': None, 'source_tasks': []}:
        return None
    text = _instruction(value, DOMAIN_SKILL_CHARS)
    if set(value) != {'instruction', 'source_tasks'}:
        raise ValueError('Domain document must contain instruction and source_tasks only')
    tasks = value.get('source_tasks')
    if (not isinstance(tasks, list) or any(not isinstance(t, str) or t not in skills for t in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError('Domain source_tasks must reference distinct supplied Task Skills')
    if text and len(tasks) < 2:
        raise ValueError('Domain Skill requires at least two source tasks')
    if not text and tasks:
        raise ValueError('Empty Domain Skill must not claim sources')
    sources = {}
    for task in tasks:
        skill = skills[task]
        if not skill.get('review') or not skill['review'][-1]['approved']:
            raise ValueError('Domain Skill may use only approved Task Skills')
        for source in skill['sources']:
            sid = source['sample_id']
            if sid not in sources:
                sources[sid] = copy.deepcopy(source)
            else:
                sources[sid]['message_indices'] = sorted(set(sources[sid]['message_indices']) | set(source['message_indices']))
    return {'format': 'document', 'instruction': text, 'sources': list(sources.values()),
            'source_tasks': tasks, 'review': []}


def validate_document_review(value, instruction):
    if (not isinstance(value, dict) or set(value) != {'approved', 'feedback', 'instruction_excerpt'}
            or not isinstance(value['approved'], bool)
            or not isinstance(value['feedback'], str) or not value['feedback'].strip()
            or not isinstance(value['instruction_excerpt'], str)):
        raise ValueError('Review requires approved (boolean), nonempty feedback and instruction_excerpt (string)')
    excerpt = value['instruction_excerpt']
    if value['approved'] and excerpt != '':
        raise ValueError('Approved review must have an empty instruction_excerpt')
    if not value['approved'] and (not excerpt.strip() or excerpt not in instruction):
        raise ValueError('Rejected review must quote an exact nonempty passage from candidate.instruction, not from evidence. '
                         f'The only text you may quote is: {instruction}')
    feedback = value['feedback'].strip()
    if excerpt:
        feedback = f'Instruction passage: {excerpt}\n{feedback}'
    return {'approved': value['approved'], 'feedback': feedback}


class SkillOptimizer:
    def __init__(self, reflector, checkpoint):
        self.reflector = reflector
        self.checkpoint = checkpoint

    def _publish(self, task, working, path, task_skills=None):
        if not working or not working['instruction'] or not working['sources']:
            return None
        budget = DOMAIN_SKILL_CHARS if task == DOMAIN_TASK else TASK_SKILL_CHARS
        reviews = []
        for attempt in range(MAX_SKILL_REVISIONS + 1):
            if len(working['instruction']) > budget:
                compact_path = '/compact.json' if attempt == 0 else f'/compact_{attempt}.json'
                working = self.checkpoint('skill_revision', path + compact_path,
                                         {'document': working, 'budget': budget},
                                         lambda: self.reflector.compact_skill(task, working, budget))
                if not working or not working['instruction']:
                    return None
            decision = self.checkpoint('skill_review', path + f'/review_{attempt}.json',
                                       {'task': task, 'document': working, 'task_skills': task_skills},
                                       lambda: self.reflector.review_skill(task, working, task_skills))
            if decision is None:  # Request failure is recorded separately from an empty/rejected document.
                return None
            reviews.append(decision)
            if decision['approved']:
                return {**working, 'review': reviews}
            if attempt < MAX_SKILL_REVISIONS:
                previous = working
                working = self.checkpoint('skill_revision', path + f'/revision_{attempt}.json',
                                         {'task': task, 'document': working, 'feedback': decision['feedback'],
                                          'task_skills': task_skills},
                                         lambda: self.reflector.revise_skill(task, working, decision['feedback'], task_skills))
                if not working or not working['instruction']:
                    return None
                if (working['instruction'] == previous['instruction']
                        and set(working['source_tasks']) == set(previous['source_tasks'])
                        and {s['sample_id']: set(s['message_indices']) for s in working['sources']}
                        == {s['sample_id']: set(s['message_indices']) for s in previous['sources']}):
                    return None  # The rejected claims and their evidence have not changed.
        return None

    def optimize_task(self, task, groups, current=None):
        if current is not None:
            validate_skill_document(current)
        working = copy.deepcopy(current)
        path = f'checkpoints/skills/tasks/{quote(task, safe=" ")}'
        for index, group in enumerate(groups):
            inputs = {'task': task, 'group': [skill_input(item) for item in group], 'current': working}
            result = self.checkpoint('task_skill', path + f'/group_{index:03d}.json', inputs,
                                     lambda: self.reflector.update_skill(task, inputs['group'], working))
            if result is not None:
                working = result
        return self._publish(task, working, path)

    def optimize_domain(self, benchmark, task_skills):
        if len(task_skills) < 2:
            return None
        path = 'checkpoints/skills/domain'
        document = self.checkpoint('domain_skill', path + '/draft.json',
                                   {'benchmark': benchmark, 'task_skills': task_skills},
                                   lambda: self.reflector.domain_skill(benchmark, task_skills))
        return self._publish(DOMAIN_TASK, document, path, task_skills)
