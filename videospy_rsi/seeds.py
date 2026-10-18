"""Snapshot the base tools without changing any base module on disk."""

from .config import ROOT, baseline_config

RUNTIME_POLICY = '''def strategy(context):
    return ""

def select_tools(names, context):
    return names
'''
EVIDENCE_POLICY = '''import copy

def prepare_messages(messages):
    return copy.deepcopy(messages)
'''
DIAGNOSER = '''def diagnose(payload, request):
    return request("diagnosis", "Diagnose video evidence and complete-task cost failures. "
        "Return JSON with hypotheses, evidence_refs, counterexamples, unknowns, and investigations. "
        "Each investigation has sample_id, tool, parameters. Queries must not contain gold answers. "
        "Do not force a single cause or assume a wrong answer means too few frames.", payload)
'''
TASK_PROPOSER = '''def propose(payload, request):
    return request("task_proposal", "Generate executable Task candidates grounded in diagnosis "
        "and counterexamples. Return JSON {candidates: [...]}. Each candidate needs layer=task, "
        "parent_id, hypothesis, files (complete allowed Python source), parameters (dotted paths), "
        "evidence_refs, risks. Change one mechanism and necessary dependencies. "
        "No evaluator/model/data access changes. Empty candidates are valid. "
        "For a uniform-overview efficiency candidate, consider TARGET_NUM_FRAMES=32 first.", payload)
'''


def task_seed(config):
    files = {}
    for name in ("overview", "clip_skim", "frame_inspect"):
        source = (ROOT / "videospy" / "tools" / f"{name}.py").read_text(encoding="utf-8")
        source = source.replace("from videospy.model import ModelClient",
                                "from videospy_rsi.runtime import ModelClient")
        source = source.replace("from .inspection import", "from videospy.tools.inspection import")
        files[f"{name}.py"] = source
    files.update({"runtime_policy.py": RUNTIME_POLICY, "evidence_policy.py": EVIDENCE_POLICY})
    return {"layer": "task", "parent_id": None, "files": files,
            "config": baseline_config(config), "skills": {}}


def meta_seed():
    return {"layer": "meta", "parent_id": None,
            "files": {"diagnoser.py": DIAGNOSER, "task_proposer.py": TASK_PROPOSER},
            "config": {}, "skills": {}}
