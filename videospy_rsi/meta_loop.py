"""Fixed outer updater and paired, fresh inner-search validation of Meta code."""

from .contracts import apply_candidate
from .records import aggregate, accepts, BudgetExceeded
from .task_loop import attempt_view

OUTER_INSTRUCTION = ("Improve the VideoSpy evolution procedure from its complete attempts, "
    "not from one wrong QA answer. Return JSON {candidates: [...]}. Each candidate has layer=meta, "
    "parent_id, hypothesis, files (complete source for diagnoser.py and/or task_proposer.py), "
    "evidence_refs and risks. Keep function interfaces, model configuration, evaluation, "
    "permissions and budget fixed. Prioritize one recurring diagnostic/proposal failure. "
    "Empty candidates are valid. Candidate utility is measured by new paired inner searches.")


class MetaLoop:
    def __init__(self, archive, runner, task_loop, config):
        self.archive, self.runner, self.task_loop, self.config = archive, runner, task_loop, config

    def step(self, current_task, initial_task, meta_id, history, episodes, key, previous=None):
        inputs = {"current_task": current_task, "initial_task": initial_task,
                  "meta": meta_id, "history": history, "episodes": episodes, "previous": previous}
        return self.archive.phase(key + "-attempt", inputs,
            lambda: self._step(current_task, initial_task, meta_id, history, episodes, key, previous))

    def _step(self, current_task, initial_task, meta_id, history, episodes, key, previous):
        meta = self.archive.get(meta_id)
        context = {"scope": "outer_meta_validation", "stage": "meta_proposal", "meta_id": meta_id,
                   "round_id": key, "seed": self.config["seed"]}
        # Never supply episode samples, labels or report trajectories to G.
        payload = {"meta": meta, "current_task": self.archive.get(current_task),
                   "initial_task_id": initial_task, "attempts": [attempt_view(a) for a in history[-6:]],
                   "limits": self.config["limits"], "previous_meta_feedback": (previous or [])[-6:]}
        proposed = self.runner.job(key + "-propose", "request", context,
                                   kind="meta_proposal", instruction=OUTER_INSTRUCTION, payload=payload)
        value = proposed.get("result")
        proposals = value.get("candidates", []) if isinstance(value, dict) else []
        if not isinstance(proposals, list):
            proposals = []
        reports = []
        for index, proposal in enumerate(proposals[:self.config["meta_candidates"]]):
            candidate_key = key + f"-candidate-{index}"
            report = {"accepted": False, "proposal": proposal, "candidate_id": None}
            try:
                candidate = self.archive.snapshot(apply_candidate(meta, proposal, "meta", self.config["limits"]))
                report["candidate_id"] = candidate["id"]
                self.archive.patch(meta, proposal, f"candidates/{candidate_key}")
                checked = self.runner.job(candidate_key + "-check", "check", {**context, "stage": "interface"}, snapshot=candidate)
                if checked["status"] != "ok":
                    raise ValueError(checked.get("error", "Meta interface check failed"))
                paired = []
                for episode_index, episode in enumerate(episodes):
                    start = initial_task if episode.get("start", "initial") == "initial" else current_task
                    pair = {}
                    for label, tested_meta in (("old", meta_id), ("new", candidate["id"])):
                        branch = f"{candidate_key}-episode-{episode_index}-{label}"
                        pair[label] = self.branch(start, tested_meta, episode, branch)
                    pair["episode_id"] = episode["id"]
                    paired.append(pair)
                old = self.combine(paired, "old")
                new = self.combine(paired, "new")
                report.update(episodes=paired, parent=old, candidate=new)
                report["accepted"], report["reason"] = accepts(old, new, self.config["selection"])
            except BudgetExceeded:
                raise
            except (ValueError, TypeError, KeyError, SyntaxError) as error:
                report["reason"] = str(error)
            reports.append(report)
        self.runner.ledger.check(context["scope"], after_spend=True)
        accepted = [r for r in reports if r["accepted"]]
        winner = min(accepted, key=lambda r: (-r["candidate"]["quality"], r["candidate"]["cost"], r["candidate_id"])) if accepted else None
        return {"round_id": key, "meta_id": meta_id, "meta_after": winner["candidate_id"] if winner else meta_id,
                "task_unchanged": current_task, "proposal_input": payload, "validation": reports,
                "outcome": "accepted" if winner else "rejected" if proposals else "no_candidate"}

    def branch(self, start_task, meta_id, episode, branch):
        history, current, failure = [], start_task, None
        context = {"scope": "outer_meta_validation", "branch": branch,
                   "seed": episode.get("seed", self.config["seed"]), "meta_id": meta_id}
        for round_index in range(self.config["episode_rounds"]):
            try:
                self.runner.ledger.check(context["scope"], branch)
                attempt = self.task_loop.step(current, meta_id, episode["probe"], episode["inner_gate"],
                    branch + f"-round-{round_index}", context, history)
                history.append(attempt)
                current = attempt["task_after"]
            except BudgetExceeded as error:
                # Local quota ends a branch; a global/unknown budget stops the run.
                if "episode" not in str(error):
                    raise
                break
            except RuntimeError as error:
                failure = str(error)
                break
        records = self.runner.batch(self.archive.get(current), episode["report"], branch + "-report",
                                    {**context, "stage": "report"})
        report = aggregate(records, episode["report"])
        if failure:
            report["complete"] = False
        initial_records = self.runner.batch(self.archive.get(start_task), episode["report"], branch + "-initial-report",
                                            {**context, "stage": "report"})
        initial_report = aggregate(initial_records, episode["report"])
        search_calls = [r for r in self.runner.ledger.events() if r.get("branch") == branch and r.get("phase") != "report"]
        summary = self.runner.ledger.summary(search_calls)
        if summary["tokens"] > self.config["limits"]["episode_tokens"]:
            # A final API call may exceed the soft token quota. It cannot earn
            # extra search beyond the paired budget or justify acceptance.
            report["complete"] = False
        return {"start_task": start_task, "final_task": current, "meta_id": meta_id,
                "report": report, "initial_report": initial_report,
                "quality_gain": report["quality"] - initial_report["quality"],
                "cost_change": report["cost"] - initial_report["cost"] if report["cost"] is not None and initial_report["cost"] is not None else None,
                "search_cost": summary, "branch": branch, "error": failure}

    @staticmethod
    def combine(paired, label):
        reports = [pair[label]["report"] for pair in paired]
        total = sum(r["count"] for r in reports)
        complete = bool(reports) and all(r["complete"] and r["usage_complete"] for r in reports)
        return {"count": total, "complete": complete, "usage_complete": complete,
                "quality": sum(r["quality"] * r["count"] for r in reports) / total if total else 0,
                "cost": sum(r["cost"] * r["count"] for r in reports) / total if complete else None,
                "failed": sum(r["failed"] for r in reports),
                "groups": {f"{pair['episode_id']}/{task}": group for pair in paired
                           for task, group in pair[label]["report"]["groups"].items()}}
