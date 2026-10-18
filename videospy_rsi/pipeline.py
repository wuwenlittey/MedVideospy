"""Official pointers and scheduling; probe branches never commit official state."""

from .archive import Archive
from .config import ROOT
from .data import validate_dataset
from .io import digest, public_config, read_json
from .meta_loop import MetaLoop
from .records import BudgetExceeded, aggregate
from .runner import Runner
from .seeds import task_seed, meta_seed
from .task_loop import TaskLoop


def implementation_identity():
    paths = list((ROOT / "videospy_rsi").glob("*.py")) + list((ROOT / "videospy" / "tools").glob("*.py"))
    paths += [ROOT / "videospy" / name for name in ("agent.py", "model.py", "metrics.py", "utils.py")]
    paths += list((ROOT / "videospy" / "core").glob("*.py"))
    paths += [ROOT / "benchmark" / "medvidbench" / "image_sequence.py"]
    return digest({str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in sorted(paths)})


class EvolutionPipeline:
    def __init__(self, config, dataset, output):
        validate_dataset(dataset)
        self.config, self.dataset = config, dataset
        identity = {"config": digest(public_config(config)), "dataset": digest(dataset),
                    "initial_task": digest(task_seed(config)),
                    "implementation": implementation_identity()}
        self.archive = Archive(output, identity)
        self.runner = Runner(self.archive, config)
        self.task_loop = TaskLoop(self.archive, self.runner, config)
        self.meta_loop = MetaLoop(self.archive, self.runner, self.task_loop, config)

    def run(self):
        state = self.archive.state()
        if state is None:
            task = self.archive.snapshot(task_seed(self.config))
            meta = self.archive.snapshot(meta_seed())
            state = {"task_id": task["id"], "meta_id": meta["id"], "initial_task": task["id"],
                     "task_round": 0, "meta_round": 0, "since_meta": 0, "status": "running"}
            self.archive.save_state(state)
        if state["status"] in {"complete", "budget_exhausted"}:
            self.archive.write("final/versions.json", {"task_id": state["task_id"], "meta_id": state["meta_id"],
                                                       "status": state["status"]})
            return state
        state["status"] = "running"
        self.archive.save_state(state)
        try:
            while state["task_round"] < self.config["max_task_rounds"]:
                self.runner.ledger.check("formal_task_search")
                index = state["task_round"]
                key = f"task-{index:04d}"
                history = [attempt for attempt in self.archive.attempts() if attempt["round_id"] < key]
                if state["since_meta"] >= self.config["meta_every"] and state["meta_round"] < self.config["max_meta_rounds"]:
                    self._meta_step(state, history)
                    continue
                samples = self.dataset["diagnostic_batches"][index % len(self.dataset["diagnostic_batches"])]
                print(f"Task round {index + 1}: {state['task_id']} / {state['meta_id']}", flush=True)
                attempt = self.task_loop.step(state["task_id"], state["meta_id"], samples, self.dataset["gate"], key,
                    {"scope": "formal_task_search", "seed": self.config["seed"]}, history)
                self.archive.write(f"task_rounds/{key}/attempt.json", attempt)
                self.runner.ledger.check("formal_task_search", after_spend=True)
                state.update(task_id=attempt["task_after"], task_round=index + 1, since_meta=state["since_meta"] + 1)
                self.archive.save_state(state)
            state["status"] = "complete"
            self.archive.save_state(state)
        except BudgetExceeded as error:
            state.update(status="budget_exhausted", stop_reason=str(error))
            self.archive.save_state(state)
        except Exception as error:
            state.update(status="failed", stop_reason=f"{type(error).__name__}: {error}")
            self.archive.save_state(state)
            raise
        self.archive.write("final/versions.json", {"task_id": state["task_id"], "meta_id": state["meta_id"],
                                                   "status": state["status"]})
        return state

    def _meta_step(self, state, history):
        key = f"meta-{state['meta_round']:04d}"
        print(f"Meta round {state['meta_round'] + 1}: paired inner searches", flush=True)
        previous = []
        for path in sorted((self.archive.root / "meta_rounds").glob("*/attempt.json")):
            if path.parent.name < key:
                attempt = read_json(path)
                previous.append({"round_id": attempt["round_id"], "outcome": attempt["outcome"],
                                 "validation": [{k: row.get(k) for k in
                                     ("candidate_id", "accepted", "reason", "parent", "candidate")}
                                     for row in attempt["validation"]]})
        result = self.meta_loop.step(state["task_id"], state["initial_task"], state["meta_id"], history,
                                     self.dataset["episodes"], key, previous)
        self.archive.write(f"meta_rounds/{key}/attempt.json", result)
        self.runner.ledger.check("outer_meta_validation", after_spend=True)
        state.update(meta_id=result["meta_after"], meta_round=state["meta_round"] + 1, since_meta=0)
        self.archive.save_state(state)

    def final_test(self):
        state = self.archive.state()
        if not state or state["status"] not in {"complete", "budget_exhausted"}:
            raise ValueError("Freeze a completed or budget-limited run before final testing")
        samples = self.dataset.get("test", [])
        if not samples:
            raise ValueError("No final test samples")
        records = self.runner.batch(self.archive.get(state["task_id"]), samples, "final-test",
                                    {"scope": "final_test", "stage": "report", "seed": self.config["seed"]})
        report = aggregate(records, samples)
        self.archive.write("final/test.json", report)
        return report
