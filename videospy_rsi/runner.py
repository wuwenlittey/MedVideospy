"""Fixed subprocess launcher with phase-level recovery and label-free inputs."""

import json
import os
import subprocess
import sys

from .config import ROOT
from .data import runtime_view
from .io import public_config
from .records import Ledger


class Runner:
    def __init__(self, archive, config):
        self.archive, self.config = archive, config
        self.ledger = Ledger(archive.root / "calls.jsonl", config["limits"])

    def job(self, key, operation, context, **kwargs):
        inputs = {"operation": operation, "context": context, "config": public_config(self.config), **kwargs}
        def produce():
            if context["scope"] != "final_test":
                self.ledger.check(context["scope"], context.get("branch") if context.get("stage") != "report" else None)
            job = {**inputs, "config": self.config, "ledger": str(self.ledger.path)}
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            cwd = self.archive.root / "workers" / key
            cwd.mkdir(parents=True, exist_ok=True)
            try:
                process = subprocess.run([sys.executable, "-m", "videospy_rsi.worker"],
                    input=json.dumps(job, ensure_ascii=False), capture_output=True, text=True,
                    encoding="utf-8", cwd=cwd, env=environment,
                    timeout=self.config["limits"]["worker_timeout"])
                if process.returncode:
                    # Do not persist arbitrary stderr, which can contain credentials.
                    return {"status": "error", "error": f"Worker exited with code {process.returncode}", "calls": []}
                result = json.loads(process.stdout)
                if result.get("status") not in {"ok", "error"}:
                    raise ValueError("Invalid worker response")
                return result
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                return {"status": "error", "error": type(error).__name__, "calls": []}
        return self.archive.phase(key, inputs, produce)

    def batch(self, task, samples, key, context):
        records = []
        for index, sample in enumerate(samples):
            subkey = f"{key}-sample-{index}"
            sample_context = {**context, "sample_id": sample["id"], "task_id": task["id"], "job_id": subkey}
            result = self.job(subkey, "task", sample_context, snapshot=task, sample=runtime_view(sample))
            # Journaled partial calls survive worker timeout or interruption.
            calls = [row for row in self.ledger.events() if row.get("job_id") == subkey]
            summary = self.ledger.summary(calls)
            if result["status"] != "ok" and not calls:
                summary["complete"] = False
            row = {"sample_id": sample["id"], "task_id": task["id"], "status": result["status"],
                   "prediction": result.get("result", {}).get("prediction", ""),
                   "trajectory": result.get("result", {}).get("trajectory", {}),
                   "visual_observations": result.get("result", {}).get("visual_observations", []),
                   "tokens": summary["tokens"], "usage_complete": summary["complete"],
                   "calls": [call["id"] for call in calls], "error": result.get("error")}
            records.append(row)
        self.archive.write(f"records/{key}.json", records)
        return records
