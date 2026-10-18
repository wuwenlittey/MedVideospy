"""Stored reflection outputs and shared Skill budgets."""
from dataclasses import asdict, dataclass, field

WORK_SKILL_CHARS = 3600
TASK_SKILL_CHARS = 1800
DOMAIN_SKILL_CHARS = 600
SKILL_PROMPT_CHARS = 3000
MEMORY_PROMPT_CHARS = 1600


def validate_skill_document(value):
    """Reject old rule artifacts rather than silently injecting a different method."""
    if (not isinstance(value, dict) or value.get('format') != 'document'
            or any(key in value for key in ('rules', 'selection', 'updates', 'execution'))):
        raise ValueError('Legacy Skill format is unsupported; regenerate from base trajectories using document Skills.')
    if not isinstance(value.get('instruction'), str):
        raise ValueError('Skill instruction must be Markdown text')
    for key in ('sources', 'source_tasks', 'review'):
        if not isinstance(value.get(key), list):
            raise ValueError(f'Skill {key} must be a list')
    seen = set()
    for source in value['sources']:
        if (not isinstance(source, dict) or not isinstance(source.get('sample_id'), str)
                or source['sample_id'] in seen or not isinstance(source.get('trajectory'), dict)
                or not isinstance(source['trajectory'].get('messages'), list)):
            raise ValueError('Skill sources must identify distinct original trajectories')
        indices = source.get('message_indices')
        messages = source['trajectory']['messages']
        if (not isinstance(indices, list) or not indices
                or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(messages)
                       for i in indices) or len(set(indices)) != len(indices)):
            raise ValueError('Skill sources require valid original message indices')
        seen.add(source['sample_id'])
    if (any(not isinstance(task, str) for task in value['source_tasks'])
            or len(set(value['source_tasks'])) != len(value['source_tasks'])):
        raise ValueError('Skill source_tasks must be distinct task names')
    for review in value['review']:
        if (not isinstance(review, dict) or not isinstance(review.get('approved'), bool)
                or not isinstance(review.get('feedback'), str) or not review['feedback'].strip()):
            raise ValueError('Skill review requires an approval decision and feedback')
    return value


@dataclass
class ReflectionReport:
    selection: dict
    skills: dict[str, dict]
    memories: list[dict]
    stages: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryEntry:
    id: str
    task_type: str
    context: str
    content: str
    source_sample_id: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EvolutionUpdate:
    skills: dict[str, dict]
    memories: list[MemoryEntry]
    hyperparameters: dict
    summary: dict


def empty_stage():
    return {"succeeded": 0, "failed": 0, "empty": 0, "skipped": 0, "reused": 0}


def training_status(stages, failures, *, has_task_skill=False):
    if any(stage["failed"] and not stage["succeeded"]
           and not (name == 'domain_skill' and has_task_skill)
           for name, stage in stages.items()):
        return "failed"
    return "partial" if failures else "complete"
