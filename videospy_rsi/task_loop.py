"""Fixed Task loop; evolving modules only diagnose and propose executable edits."""

import copy

from .contracts import apply_candidate
from .data import correct, runtime_view
from .io import digest
from .records import aggregate, accepts, BudgetExceeded

ASSESSMENT = ("Assess this completed video QA trajectory using gold only as outcome feedback. "
    "Return JSON with evidence_support (sufficient/partial/insufficient), score (integer 0-5), reason. "
    "A correct answer is not proof of sufficient evidence. 4-5 require correct with sufficient evidence. "
    "Efficiency only counts when relevant evidence supports the answer. Unknown costs are unknown.")


def readable(record):
    row = copy.deepcopy(record)
    for message in row.get("trajectory", {}).get("messages", []):
        for key in ("model_call", "model_calls"):
            message.pop(key, None)
    return row


def attempt_view(attempt):
    """Bounded historical view; full inputs stay in the immutable phase archive."""
    reports = []
    for report in attempt.get("validation", []):
        proposal = report.get("proposal")
        proposal = proposal if isinstance(proposal, dict) else {}
        files = proposal.get("files")
        files = files if isinstance(files, dict) else {}
        reports.append({"candidate_id": report.get("candidate_id"), "accepted": report.get("accepted"),
                        "reason": report.get("reason"), "gate": report.get("gate"),
                        "parent_gate": report.get("parent_gate"),
                        "proposal": {k: v for k, v in proposal.items() if k != "files"},
                        "changed_source_hashes": {k: digest(v) for k, v in files.items()}})
    return {k: attempt.get(k) for k in ("round_id", "task_id", "meta_id", "task_after", "outcome")} | {
        "diagnosis": attempt.get("diagnosis"), "validation": reports,
        "input_ref": attempt.get("round_id") + "-attempt",
        "cases": attempt.get("proposal_input", {}).get("cases", [])}


class TaskLoop:
    def __init__(self, archive, runner, config):
        self.archive, self.runner, self.config = archive, runner, config

    def step(self, task_id, meta_id, samples, gate, key, context, history):
        task, meta = self.archive.get(task_id), self.archive.get(meta_id)
        identity = {"task": task_id, "meta": meta_id, "samples": samples, "gate": gate,
                    "context": context, "history": history}
        return self.archive.phase(key + "-attempt", identity,
                                  lambda: self._step(task, meta, samples, gate, key, context, history))

    def _step(self, task, meta, samples, gate, key, context, history):
        context = {**context, "meta_id": meta["id"], "round_id": key}
        records = self.runner.batch(task, samples, key + "-diag", {**context, "stage": "diagnostic"})
        if not any(record["status"] == "ok" for record in records):
            raise RuntimeError("All diagnostic executions failed: " + str(records[0].get("error")))
        cases = []
        for index, (record, sample) in enumerate(zip(records, samples)):
            payload = {"question": sample["question"], "reference_answer": sample["answer"],
                       "correct": correct(record, sample), "record": readable(record)}
            result = self.runner.job(key + f"-assess-{index}", "request", {**context, "stage": "assessment"},
                                     kind="assessment", instruction=ASSESSMENT, payload=payload)
            assessment = result.get("result", {})
            if not isinstance(assessment, dict):
                assessment = {}
            if assessment.get("evidence_support") not in {"sufficient", "partial", "insufficient"} or type(assessment.get("score")) is not int or not 0 <= assessment["score"] <= 5:
                assessment = {"evidence_support": "unknown", "score": 0,
                              "reason": result.get("error", "Malformed assessment")}
            if assessment["score"] >= 4 and (not correct(record, sample) or assessment["evidence_support"] != "sufficient"):
                assessment["score"] = 3
            cases.append({**readable(record), "question": sample["question"], "reference_answer": sample["answer"],
                          "task_type": sample.get("task_type", "general"), "assessment": assessment,
                          "correct": correct(record, sample)})
        # Fixed, balanced selector: alternate low/high scores within task groups.
        groups = {}
        for case in cases:
            groups.setdefault(case["task_type"], []).append(case)
        selected = []
        for task_type in sorted(groups):
            bucket = sorted(groups[task_type], key=lambda row: (row["assessment"]["score"], row["sample_id"]))
            while bucket and len(selected) < 8:
                selected.append(bucket.pop(0))
                if bucket and len(selected) < 8:
                    selected.append(bucket.pop())
        history = [attempt_view(attempt) for attempt in history[-6:]]
        diag_payload = {"task": task, "cases": selected, "history": history}
        diagnosis_job = self.runner.job(key + "-diagnose", "diagnose", {**context, "stage": "diagnosis"},
                                        snapshot=meta, payload=diag_payload)
        diagnosis = diagnosis_job.get("result")
        if not isinstance(diagnosis, dict):
            diagnosis = {"unknowns": [diagnosis_job.get("error", "Malformed diagnosis")]}
        investigations = []
        indexed = {sample["id"]: sample for sample in samples}
        requests = diagnosis.get("investigations", [])
        if not isinstance(requests, list):
            requests = []
        for index, request in enumerate(requests[:self.config["investigations"]]):
            if not isinstance(request, dict) or request.get("sample_id") not in indexed:
                investigations.append({"status": "error", "error": "Unknown diagnostic sample"})
                continue
            sample = indexed[request["sample_id"]]
            parameters = request.get("parameters", {})
            query = str(parameters.get("query", "")) if isinstance(parameters, dict) else ""
            # Short option letters cannot be meaningfully screened as substrings.
            # Longer exact gold text must never be copied into a visual query.
            if len(sample["answer"].strip()) > 3 and sample["answer"].casefold() in query.casefold():
                investigations.append({"status": "error", "error": "Gold text in visual investigation query"})
                continue
            result = self.runner.job(key + f"-investigate-{index}", "investigate",
                {**context, "stage": "investigation", "sample_id": sample["id"]}, snapshot=task,
                sample=runtime_view(sample), tool=request.get("tool"), parameters=parameters)
            investigations.append({"request": request, **result})
        payload = {"task": task, "meta_id": meta["id"], "cases": selected,
                   "diagnosis": diagnosis, "investigations": investigations,
                   "history": history, "allowed_files": sorted(task["files"]),
                   "limits": self.config["limits"]}
        proposed = self.runner.job(key + "-propose", "propose", {**context, "stage": "task_proposal"},
                                   snapshot=meta, payload=payload)
        value = proposed.get("result")
        proposals = value.get("candidates", []) if isinstance(value, dict) else []
        if not isinstance(proposals, list):
            proposals = []
        reports = []
        baseline = None
        for index, proposal in enumerate(proposals[:self.config["task_candidates"]]):
            report, candidate, baseline = self.validate(task, proposal, samples, gate,
                f"{key}-candidate-{index}", context, baseline)
            if not report["accepted"] and report.get("repairable") and self.config["repair_attempts"]:
                # Repairs remain children of the exact same Task parent.
                for revision in range(self.config["repair_attempts"]):
                    repair_payload = {**payload, "repair": {"candidate": proposal, "error": report.get("reason")}}
                    repaired = self.runner.job(f"{key}-repair-{index}-{revision}", "propose",
                        {**context, "stage": "local_repair"}, snapshot=meta, payload=repair_payload)
                    value = repaired.get("result")
                    repair_proposals = value.get("candidates", []) if isinstance(value, dict) else []
                    if not isinstance(repair_proposals, list) or not repair_proposals:
                        break
                    reports.append(report)
                    proposal = repair_proposals[0]
                    report, candidate, baseline = self.validate(task, proposal, samples, gate,
                        f"{key}-candidate-{index}-revision-{revision}", context, baseline)
                    if report["accepted"] or not report.get("repairable"):
                        break
            reports.append(report)
        self.runner.ledger.check(context["scope"], context.get("branch"), after_spend=True)
        eligible = [r for r in reports if r["accepted"]]
        winner = min(eligible, key=lambda r: (-r["gate"]["quality"], r["gate"]["cost"], r["candidate_id"])) if eligible else None
        attempt = {"round_id": key, "task_id": task["id"], "meta_id": meta["id"],
                   "scope": context["scope"], "branch": context.get("branch"),
                   "proposal_input": payload, "diagnosis": diagnosis, "investigations": investigations,
                   "validation": reports, "task_after": winner["candidate_id"] if winner else task["id"],
                   "outcome": "accepted" if winner else "rejected" if proposals else "no_candidate"}
        return attempt

    def validate(self, parent, proposal, local_samples, gate, key, context, baseline=None):
        report = {"candidate_id": None, "accepted": False, "proposal": proposal,
                  "repairable": True, "reason": ""}
        try:
            material = apply_candidate(parent, proposal, "task", self.config["limits"])
            candidate = self.archive.snapshot(material)
            report["candidate_id"] = candidate["id"]
            self.archive.patch(parent, proposal, f"candidates/{key}")
            check = self.runner.job(key + "-check", "check", {**context, "stage": "interface"}, snapshot=candidate)
            if check["status"] != "ok":
                raise ValueError(check.get("error", "Interface check failed"))
            local = self.runner.batch(candidate, local_samples, key + "-local", {**context, "stage": "local_validation"})
            if any(r["status"] != "ok" for r in local):
                raise ValueError("Local candidate execution failed")
            if baseline is None:
                records = self.runner.batch(parent, gate, context["round_id"] + "-parent-gate",
                                            {**context, "stage": "gate"})
                baseline = aggregate(records, gate)
            records = self.runner.batch(candidate, gate, key + "-gate", {**context, "stage": "gate"})
            report["gate"] = aggregate(records, gate)
            report["parent_gate"] = baseline
            report["accepted"], report["reason"] = accepts(baseline, report["gate"], self.config["selection"])
            report["repairable"] = False
            return report, candidate, baseline
        except BudgetExceeded:
            raise
        except (ValueError, TypeError, KeyError, SyntaxError) as error:
            report["reason"] = str(error)
            return report, None, baseline
