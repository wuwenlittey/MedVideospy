"""Behavior tests for the isolated dual loop; no paid model/video calls."""

import copy
import json
import tempfile
import unittest
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from videospy_rsi.archive import Archive
from videospy_rsi.config import load_config
from videospy_rsi.contracts import apply_candidate, check_source
from videospy_rsi.data import demo_dataset, runtime_view, validate_dataset
from videospy_rsi.io import read_json, write_json, public_config
from videospy_rsi.pipeline import EvolutionPipeline
from videospy_rsi.records import Ledger, BudgetExceeded, accepts
from videospy_rsi.seeds import task_seed, meta_seed


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(backend="demo")
        self.parent = {**task_seed(self.config), "id": "task-parent"}

    def candidate(self, **kwargs):
        return {"layer": "task", "parent_id": "task-parent", "hypothesis": "Inspect fewer repeated frames", **kwargs}

    def test_parameter_edit_keeps_parent_and_models_immutable(self):
        original = copy.deepcopy(self.parent)
        child = apply_candidate(self.parent, self.candidate(parameters={"tools.overview.num_frames": 8}), "task", self.config["limits"])
        self.assertEqual(self.parent, original)
        self.assertEqual(child["config"]["tools"]["overview"]["num_frames"], 8)
        self.assertNotIn("id", child)
        self.assertEqual(child["config"]["agent"]["model"], original["config"]["agent"]["model"])

    def test_reject_model_evaluator_path_and_resource_changes(self):
        for proposal in (
            self.candidate(parameters={"agent.model.model_name": "cheater"}),
            self.candidate(files={"../agent.py": "pass"}),
            self.candidate(parameters={"tools.overview.num_frames": 129}),
            self.candidate(parameters={"tools.clip_skim.max_duration": 0}),
            self.candidate(parent_id="different", parameters={"agent.max_steps": 2}),
        ):
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                apply_candidate(self.parent, proposal, "task", self.config["limits"])

    def test_source_restrictions_and_noop(self):
        for source in ("import os", "x = open('labels')", "x = getattr(a, '__dict__')",
                       "from videospy_rsi.runtime import _ACTIVE", "x = reader._reader"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                check_source("runtime_policy.py", source, "task", self.config["limits"])
        with self.assertRaises(ValueError):
            apply_candidate(self.parent, self.candidate(files={}), "task", self.config["limits"])

    def test_seed_code_passes_same_contract(self):
        for snapshot in (task_seed(self.config), meta_seed()):
            for name, source in snapshot["files"].items():
                check_source(name, source, snapshot["layer"], self.config["limits"])

    def test_meta_cannot_change_task_parameters(self):
        parent = {**meta_seed(), "id": "meta-parent"}
        proposal = {"layer": "meta", "parent_id": parent["id"], "hypothesis": "Invalid",
                    "parameters": {"agent.max_steps": 1}}
        with self.assertRaises(ValueError):
            apply_candidate(parent, proposal, "meta", self.config["limits"])


class DataAndCostTest(unittest.TestCase):
    def test_final_call_overspend_blocks_promotion(self):
        config = load_config(backend="demo")
        limits = {**config["limits"], "total_tokens": 100}
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(Path(root) / "calls.jsonl", limits)
            ledger.append({"usage_complete": True, "total_tokens": 100})
            ledger.check("formal_task_search", after_spend=True)
            with self.assertRaises(BudgetExceeded):
                ledger.check("formal_task_search")
            ledger.append({"usage_complete": True, "total_tokens": 1})
            with self.assertRaises(BudgetExceeded):
                ledger.check("formal_task_search", after_spend=True)

    def test_all_retry_attempts_are_journaled(self):
        from videospy_rsi.model import Gateway
        config = load_config(backend="demo")
        config["backend"] = "live"
        calls = [{"metrics": {"input_tokens": None, "output_tokens": None, "api_attempt": 1, "error": "retry"}},
                 {"metrics": {"input_tokens": 10, "output_tokens": 5, "api_attempt": 2}, "output": {"content": "A"}}]
        @contextmanager
        def collector():
            yield calls
        fake = types.ModuleType("videospy.model")
        fake.collect_model_calls = collector
        fake.ModelClient = lambda _: types.SimpleNamespace(chat=lambda *a, **k: {"content": "A"})
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {"videospy.model": fake}):
            gateway = Gateway(config, Path(root) / "calls.jsonl", {"scope": "formal_task_search"})
            gateway.chat("task", [{"role": "user", "content": "Q"}])
            self.assertEqual([row["attempt"] for row in gateway.calls], [1, 2])
            self.assertEqual(gateway.ledger.summary()["tokens"], 15)
            self.assertEqual(gateway.ledger.summary()["unknown_calls"], 1)
            with self.assertRaises(BudgetExceeded):
                gateway.ledger.check("formal_task_search", after_spend=True)

    def test_labels_and_unknown_metadata_not_in_runtime_view(self):
        sample = {**demo_dataset()["gate"][0], "ground_truth": "secret", "struc_info": {"gold": 1}}
        view = runtime_view(sample)
        self.assertNotIn("answer", view)
        self.assertNotIn("ground_truth", view)
        self.assertNotIn("struc_info", view)

    def test_video_leakage_and_report_overlap_rejected(self):
        for mode in ("gate", "report"):
            data = demo_dataset()
            if mode == "gate":
                data["gate"][0]["source_id"] = data["diagnostic_batches"][0][0]["source_id"]
            else:
                e = data["episodes"][0]
                e["report"][0]["source_id"] = e["probe"][0]["source_id"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_dataset(data)

    def test_unknown_usage_is_not_a_cost_saving(self):
        rule = load_config(backend="demo")["selection"]
        parent = {"complete": True, "usage_complete": True, "quality": 1.0, "cost": 100, "failed": 0}
        child = {**parent, "cost": None, "usage_complete": False}
        self.assertFalse(accepts(parent, child, rule)[0])
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(Path(root) / "calls.jsonl", load_config(backend="demo")["limits"])
            ledger.append({"usage_complete": False, "total_tokens": None})
            self.assertEqual(ledger.summary()["unknown_calls"], 1)
            with self.assertRaises(BudgetExceeded):
                ledger.check("formal_task_search")

    def test_category_regression_cannot_be_hidden_in_average(self):
        rule = load_config(backend="demo")["selection"]
        p = {"complete": True, "usage_complete": True, "quality": .8, "cost": 100,
             "groups": {"action": {"quality": .8}}}
        c = {**p, "quality": .9, "cost": 90, "groups": {"action": {"quality": .7}}}
        self.assertFalse(accepts(p, c, rule)[0])


class ArchiveTest(unittest.TestCase):
    def test_modified_snapshot_and_changed_resume_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Archive(root, {"run": "same"})
            version = archive.snapshot(meta_seed())
            path = Path(root) / "versions/meta" / version["id"] / "diagnoser.py"
            path.write_text("bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                archive.get(version["id"])
            with self.assertRaises(ValueError):
                Archive(root, {"run": "changed"})

    def test_checkpoint_reuses_only_exact_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Archive(root, {})
            first = archive.phase("phase", {"x": 1}, lambda: {"value": 2})
            reused = archive.phase("phase", {"x": 1}, lambda: self.fail("Repeated producer"))
            self.assertEqual(first, reused)
            with self.assertRaises(ValueError):
                archive.phase("phase", {"x": 2}, lambda: {})

    def test_credentials_not_exported(self):
        public = public_config({"models": {"api_key": "private", "model_name": "fixed"}})
        self.assertEqual(public, {"models": {"model_name": "fixed"}})


class FullLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            import PIL
        except ImportError:
            raise unittest.SkipTest("Offline visual integration needs numpy and Pillow")
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = load_config(backend="demo")
        cls.pipeline = EvolutionPipeline(cls.config, demo_dataset(), cls.temp.name)
        cls.state = cls.pipeline.run()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def test_both_layers_accept_and_later_task_uses_new_meta(self):
        state = self.state
        self.assertEqual(state["status"], "complete")
        self.assertNotEqual(state["task_id"], state["initial_task"])
        attempts = self.pipeline.archive.attempts()
        self.assertNotEqual(attempts[0]["meta_id"], attempts[3]["meta_id"])
        self.assertEqual(attempts[0]["outcome"], "accepted")
        self.assertEqual(attempts[1]["outcome"], "rejected")
        self.assertEqual(attempts[1]["task_after"], attempts[0]["task_after"])

    def test_meta_validation_never_promotes_a_probe_task(self):
        outer = read_json(Path(self.temp.name) / "meta_rounds/meta-0000/attempt.json")
        self.assertEqual(outer["outcome"], "accepted")
        self.assertEqual(outer["task_unchanged"], self.pipeline.archive.attempts()[2]["task_after"])
        episodes = outer["validation"][0]["episodes"]
        self.assertEqual(len(episodes), 2)
        for pair in episodes:
            self.assertEqual(pair["old"]["start_task"], pair["new"]["start_task"])
            self.assertNotEqual(pair["old"]["branch"], pair["new"]["branch"])

    def test_gate_and_meta_report_labels_not_in_generator_payloads(self):
        ids = {s["id"] for s in demo_dataset()["gate"]}
        ids |= {s["id"] for e in demo_dataset()["episodes"] for s in e["report"]}
        for event in self.pipeline.runner.ledger.events():
            if event["stage"] in {"diagnosis", "task_proposal", "meta_proposal"}:
                value = json.dumps(event["input"])
                for sid in ids:
                    self.assertNotIn(f'"id": "{sid}"', value)

    def test_model_calls_observations_and_final_test_are_recorded(self):
        rows = read_json(Path(self.temp.name) / "records/task-0000-diag.json")
        self.assertTrue(rows[0]["visual_observations"])
        self.assertTrue(rows[0]["usage_complete"])
        tools = [m for m in rows[0]["trajectory"]["messages"] if m.get("role") == "tool"]
        self.assertTrue(tools[0]["model_calls"])
        before = len(self.pipeline.archive.attempts())
        report = self.pipeline.final_test()
        self.assertEqual(report["quality"], 1.0)
        self.assertEqual(len(self.pipeline.archive.attempts()), before)
        self.assertTrue(any(r["scope"] == "final_test" for r in self.pipeline.runner.ledger.events()))

    def test_completed_resume_does_not_repeat_calls(self):
        count = len(self.pipeline.runner.ledger.events())
        rerun = EvolutionPipeline(self.config, demo_dataset(), self.temp.name)
        self.assertEqual(rerun.run(), self.state)
        self.assertEqual(len(rerun.runner.ledger.events()), count)

    def test_interrupt_after_attempt_recovers_without_repeating_calls(self):
        config = copy.deepcopy(self.config)
        config.update(max_task_rounds=1, max_meta_rounds=0)
        with tempfile.TemporaryDirectory() as root:
            pipeline = EvolutionPipeline(config, demo_dataset(), root)
            original = pipeline.archive.save_state
            def interrupted(state):
                if state["task_round"] == 1:
                    raise KeyboardInterrupt("Simulate interruption before official commit")
                original(state)
            with patch.object(pipeline.archive, "save_state", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.run()
            before = len(pipeline.runner.ledger.events())
            self.assertEqual(pipeline.archive.state()["task_round"], 0)
            resumed = EvolutionPipeline(config, demo_dataset(), root)
            state = resumed.run()
            self.assertEqual(state["task_round"], 1)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(resumed.runner.ledger.events()), before)

    def test_budget_failure_preserves_official_versions(self):
        config = copy.deepcopy(self.config)
        config.update(max_task_rounds=1, max_meta_rounds=0)
        with tempfile.TemporaryDirectory() as root:
            pipeline = EvolutionPipeline(config, demo_dataset(), root)
            def overspend(*args, **kwargs):
                pipeline.runner.ledger.append({"scope": "formal_task_search", "usage_complete": True,
                                              "total_tokens": config["limits"]["total_tokens"] + 1})
                return {"task_after": "should-never-be-promoted"}
            with patch.object(pipeline.task_loop, "step", side_effect=overspend):
                state = pipeline.run()
            self.assertEqual(state["status"], "budget_exhausted")
            self.assertEqual(state["task_id"], state["initial_task"])
            self.assertEqual(state["task_round"], 0)

    def test_task_source_candidate_is_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = EvolutionPipeline(self.config, demo_dataset(), root)
            parent = pipeline.archive.snapshot(task_seed(self.config))
            proposal = {"layer": "task", "parent_id": parent["id"], "hypothesis": "Add task-aware strategy",
                        "files": {"runtime_policy.py": 'def strategy(context):\n    return "SOURCE_CANDIDATE_LOADED"\n\ndef select_tools(names, context):\n    return names\n'}}
            candidate = pipeline.archive.snapshot(apply_candidate(parent, proposal, "task", self.config["limits"]))
            rows = pipeline.runner.batch(candidate, demo_dataset()["gate"], "source-check",
                                         {"scope": "formal_task_search", "stage": "local_validation"})
            self.assertEqual(rows[0]["status"], "ok")
            self.assertIn("SOURCE_CANDIDATE_LOADED", rows[0]["trajectory"]["messages"][0]["content"])


class BaselineParityTest(unittest.TestCase):
    def test_t0_preserves_base_model_inputs_and_protocol(self):
        try:
            import yaml, decord, litellm
        except ImportError:
            self.skipTest("Base parity needs the existing VideoSpy environment")
        from tests.test_videospy import FakeVideoReader, FakeModelClient
        from videospy.agent import VideoSpyAgent
        from videospy_rsi.runtime import run_task
        config = load_config(backend="demo")
        snapshot = {**task_seed(config), "id": "task-parity"}
        responses = [{"role": "assistant", "content": "Inspect", "tool_calls": [
            {"id": "tool-1", "type": "function", "function": {"name": "overview",
             "arguments": '{"query": "Describe the scene"}'}}]},
            {"role": "assistant", "content": "Evidence collected"}, {"role": "assistant", "content": "A"}]
        base_task, base_tool = FakeModelClient(responses), FakeModelClient([{"content": "Black frames"}])
        question = "What color? A. black B. red"
        with patch("videospy.agent.ModelClient", return_value=base_task), patch("videospy.tools.overview.ModelClient", return_value=base_tool):
            agent = VideoSpyAgent(snapshot["config"], "video", None, ".", snapshot["config"]["tools"]["enabled"], video_reader=FakeVideoReader())
            baseline = agent.run(question).to_dict()
        rsi_task, rsi_tool = FakeModelClient(responses), FakeModelClient([{"content": "Black frames"}])
        class ParityGateway:
            def __init__(self):
                self.config, self.calls = config, []
            def chat(self, role, messages, stage=None, **kwargs):
                self.calls.append({"role": role, "images": [{}] if role != "task" else []})
                return (rsi_task if role == "task" else rsi_tool).chat(messages, **kwargs)
        with patch("videospy_rsi.runtime.make_reader", return_value=FakeVideoReader()):
            outcome = run_task(snapshot, {"source_id": "video", "video_path": "video", "question": question}, ParityGateway())
        self.assertEqual(outcome["prediction"], baseline["final_answer"])
        self.assertEqual(rsi_task.calls, base_task.calls)
        self.assertEqual(rsi_tool.calls, base_tool.calls)


if __name__ == "__main__":
    unittest.main()
