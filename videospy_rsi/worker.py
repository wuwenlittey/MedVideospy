"""Fresh-process worker. Labels and other branch histories are not Task inputs."""

import contextlib
import json
import sys
import types


def demo_dependencies():
    # Explicitly scoped to the offline demo worker; never patch a live process.
    for name, attr in (("decord", "VideoReader"), ("litellm", "completion")):
        module = types.ModuleType(name)
        setattr(module, attr, object if name == "decord" else lambda **kwargs: None)
        sys.modules[name] = module


def execute(job):
    from .model import Gateway
    gateway = Gateway(job["config"], job["ledger"], job["context"])
    try:
        if job["config"]["backend"] == "demo":
            demo_dependencies()
        operation = job["operation"]
        if operation == "task":
            from .runtime import run_task
            result = run_task(job["snapshot"], job["sample"], gateway)
        elif operation == "investigate":
            from .runtime import investigate
            result = investigate(job["snapshot"], job["sample"], job["tool"], job["parameters"], gateway)
        elif operation in {"diagnose", "propose"}:
            from .runtime import load_modules
            modules = load_modules(job["snapshot"], job["config"]["limits"])
            module, function = ("diagnoser.py", "diagnose") if operation == "diagnose" else ("task_proposer.py", "propose")
            result = getattr(modules[module], function)(job["payload"], gateway.request)
            if not isinstance(result, dict):
                raise ValueError("Meta module must return a JSON object")
        elif operation == "request":
            result = gateway.request(job["kind"], job["instruction"], job["payload"])
        elif operation == "check":
            from .runtime import load_modules
            modules = load_modules(job["snapshot"], job["config"]["limits"])
            expected = {"diagnoser.py": "diagnose", "task_proposer.py": "propose"} if job["snapshot"]["layer"] == "meta" else {
                "overview.py": "execute_overview", "clip_skim.py": "execute_clip_skim",
                "frame_inspect.py": "execute_frame_inspect", "runtime_policy.py": "strategy",
                "evidence_policy.py": "prepare_messages"}
            if any(not callable(getattr(modules[name], function, None)) for name, function in expected.items()):
                raise ValueError("Missing required plugin function")
            result = {"checked": True}
        else:
            raise ValueError("Unknown worker operation")
        return {"status": "ok", "result": result, "calls": gateway.calls}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}", "calls": gateway.calls}


def main():
    job = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = execute(job)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
