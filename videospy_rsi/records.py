"""Append-only call ledger; unknown usage never becomes zero-cost evidence."""

import json
import uuid
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, limits):
        self.path = Path(path)
        self.limits = limits

    def events(self):
        if not self.path.exists():
            return []
        rows = []
        seen = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] not in seen:
                rows.append(row)
                seen.add(row["id"])
        return rows

    def append(self, event):
        event = {**event, "id": event.get("id") or uuid.uuid4().hex}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        return event

    def summary(self, rows=None):
        rows = self.events() if rows is None else rows
        unknown = sum(not row.get("usage_complete", False) for row in rows)
        return {"calls": len(rows), "tokens": sum(row.get("total_tokens") or 0 for row in rows),
                "unknown_calls": unknown, "complete": unknown == 0}

    def check(self, scope, branch=None, after_spend=False):
        rows = [row for row in self.events() if row.get("scope") != "final_test"]
        summary = self.summary(rows)
        if summary["unknown_calls"]:
            raise BudgetExceeded("Usage is unknown; stop search rather than assume zero cost")
        def exhausted(value, limit):
            return value > limit if after_spend else value >= limit
        if exhausted(summary["calls"], self.limits["max_calls"]) or exhausted(summary["tokens"], self.limits["total_tokens"]):
            raise BudgetExceeded("Global RSI search budget exhausted")
        if branch:
            local = self.summary([r for r in rows if r.get("branch") == branch and r.get("phase") != "report"])
            if exhausted(local["tokens"], self.limits["episode_tokens"]):
                raise BudgetExceeded("Meta episode search budget exhausted")


def aggregate(records, samples):
    from .data import correct
    indexed = {row["sample_id"]: row for row in records}
    complete = len(indexed) == len(samples) and all(s["id"] in indexed for s in samples)
    results = [indexed.get(s["id"], {"status": "missing", "usage_complete": False}) for s in samples]
    count = len(samples)
    groups = {}
    for sample, row in zip(samples, results):
        task = sample.get("task_type", "general")
        group = groups.setdefault(task, {"count": 0, "correct": 0})
        group["count"] += 1
        group["correct"] += int(correct(row, sample))
    usage_complete = complete and all(row.get("usage_complete", False) for row in results)
    return {"count": count, "complete": complete, "usage_complete": usage_complete,
            "quality": sum(correct(row, s) for s, row in zip(samples, results)) / count if count else 0,
            "cost": sum(row.get("tokens") or 0 for row in results) / count if usage_complete and count else None,
            "failed": sum(row.get("status") != "ok" for row in results),
            "groups": {key: {**value, "quality": value["correct"] / value["count"]}
                       for key, value in groups.items()}}


def accepts(parent, candidate, rule):
    if not all(report.get("complete") and report.get("usage_complete") for report in (parent, candidate)):
        return False, "Incomplete evaluation or usage"
    if candidate.get("failed", 0):
        return False, "Candidate execution failed"
    q0, q1 = parent["quality"], candidate["quality"]
    c0, c1 = parent["cost"], candidate["cost"]
    if c0 is None or c1 is None:
        return False, "Unknown cost"
    floor = q0 - rule["quality_tolerance"]
    for task, previous in parent.get("groups", {}).items():
        current = candidate.get("groups", {}).get(task)
        if current is None or current["quality"] < previous["quality"] - rule["quality_tolerance"]:
            return False, "Task category regressed"
    if q1 - q0 >= rule["min_quality_gain"] and c1 <= c0 * (1 + rule["max_cost_growth"]):
        return True, "Quality improved within cost bound"
    if q1 >= floor and c0 > 0 and c1 <= c0 * (1 - rule["min_cost_reduction"]):
        return True, "Cost reduced within quality bound"
    return False, "No qualifying quality/cost improvement"
