"""Fixed candidate boundary, interfaces and execution-parameter checks."""

import ast
import copy
import math

TASK_FILES = {"overview.py", "clip_skim.py", "frame_inspect.py",
              "runtime_policy.py", "evidence_policy.py"}
META_FILES = {"diagnoser.py", "task_proposer.py"}
PARAMETERS = {
    "agent.max_steps": (1, "max_steps"),
    "tools.overview.num_frames": (1, "max_frames"),
    "tools.clip_skim.num_frames": (1, "max_frames"),
    "tools.frame_inspect.max_frames": (1, "max_frames"),
    "tools.overview.subtitle_context_seconds": (0, None),
    "tools.clip_skim.subtitle_context_seconds": (0, None),
    "tools.frame_inspect.subtitle_context_seconds": (0, None),
    "tools.clip_skim.max_duration": (0.000001, None),
}
ALLOWED_IMPORTS = {"copy", "math", "json", "base64", "io", "numpy", "PIL",
                   "videospy.tools.inspection", "videospy.utils", "videospy_rsi.runtime"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "globals", "locals",
                   "setattr", "delattr", "vars", "__import__", "breakpoint"}
FORBIDDEN_ATTRS = {"read", "read_text", "read_bytes", "write_text", "write_bytes",
                  "load", "loads", "loadtxt", "fromfile", "tofile", "memmap", "ctypeslib",
                  "system", "popen", "getenv", "environ", "modules"}


def check_source(name, source, layer, limits):
    if name not in (TASK_FILES if layer == "task" else META_FILES):
        raise ValueError(f"File is outside the {layer} boundary: {name}")
    if not isinstance(source, str) or len(source) > limits["max_source_characters"]:
        raise ValueError("Invalid or oversized candidate source")
    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(module not in ALLOWED_IMPORTS for module in names) or getattr(node, "level", 0):
                raise ValueError(f"Disallowed import in {name}: {names}")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id in FORBIDDEN_ATTRS or node.id.startswith("__")):
            raise ValueError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str) or node.args[1].value.startswith("_"):
                raise ValueError("getattr only accepts literal public attribute names")
        if isinstance(node, ast.ImportFrom) and node.module == "videospy_rsi.runtime" and any(alias.name != "ModelClient" for alias in node.names):
            raise ValueError("Candidates may only import the fixed ModelClient capability")
        if isinstance(node, ast.ImportFrom) and node.module == "videospy.utils" and any(alias.name not in {
            "parse_video_timestamp", "sampled_subtitles_text", "validate_time_range"} for alias in node.names):
            raise ValueError("Candidates may only import video timestamp/subtitle helpers")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_")
                                               or node.attr in FORBIDDEN_ATTRS):
            raise ValueError(f"Disallowed attribute: {node.attr}")
    compile(tree, name, "exec")


def apply_candidate(parent, candidate, layer, limits):
    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be an object")
    if candidate.get("layer") != layer or candidate.get("parent_id") != parent["id"]:
        raise ValueError("Candidate layer or exact parent does not match")
    if not isinstance(candidate.get("hypothesis"), str) or not candidate["hypothesis"].strip():
        raise ValueError("Candidate must explain its hypothesis")
    allowed = {"layer", "parent_id", "hypothesis", "files", "parameters", "skills",
               "evidence_refs", "risks", "expected_effect", "local_check_plan"}
    if set(candidate) - allowed:
        raise ValueError(f"Unknown candidate fields: {sorted(set(candidate) - allowed)}")
    result = copy.deepcopy(parent)
    files = candidate.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("files must be a filename-to-source mapping")
    for name, source in files.items():
        check_source(name, source, layer, limits)
        result["files"][name] = source
    parameters = candidate.get("parameters", {})
    if not isinstance(parameters, dict) or (layer == "meta" and parameters):
        raise ValueError("Invalid execution parameters")
    for path, value in parameters.items():
        if path not in PARAMETERS:
            raise ValueError(f"Parameter is not evolvable: {path}")
        minimum, cap = PARAMETERS[path]
        if path.endswith("max_duration") and value is None:
            pass
        elif type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"Invalid parameter: {path}")
        elif path.endswith(("num_frames", "max_frames", "max_steps")) and type(value) is not int:
            raise ValueError(f"Integer required: {path}")
        elif cap and value > limits[cap]:
            raise ValueError(f"Parameter exceeds fixed limit: {path}")
        target = result["config"]
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    if "skills" in candidate:
        skills = candidate["skills"]
        if layer != "task" or not isinstance(skills, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or len(v) > 1800 for k, v in skills.items()
        ) or sum(map(len, skills.values())) > 12000:
            raise ValueError("Invalid Task skill asset")
        result["skills"] = skills
    if all(result.get(key) == parent.get(key) for key in ("files", "config", "skills")):
        raise ValueError("Candidate contains no change")
    result.pop("id", None)
    result["parent_id"] = parent["id"]
    return result
