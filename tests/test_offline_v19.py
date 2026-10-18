"""V19 failure recovery, complete inputs, and factual-memory scope contracts."""
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import Mock, patch

from tests.test_evolving import FakeReflector, base_config, write_source
from videospy.evolving.cli import main
from videospy.evolving.common.memory_scope import egolife_seconds, medvidbench_source_context, source_context
from videospy.evolving.storage.memory_store import CachedEmbedding, EvolutionStateStore
from videospy.evolving.reflector.evidence import build_evidence, tool_observations
from videospy.evolving.reflector.runner import ReflectionRunner
from videospy.evolving.optimizer import MemoryOptimizer
from videospy.evolving.pipeline import EvolutionPipeline
from videospy.evolving.reflector import Reflector
from videospy.evolving.reflector.evidence import trajectory_for_reflection
from videospy.evolving.common.schemas import EvolutionUpdate, MemoryEntry, ReflectionReport
from videospy.model import ModelClient


class OfflineV19Test(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.records = write_source(self.source, count=4)
        self.manifest = self.root / "train.json"
        self.manifest.write_text('[0, 1, 2, 3]')

    def pipeline(self, reflector=None, config=None, name="run"):
        return EvolutionPipeline("lvbench", self.manifest, self.root / "out", config or base_config(),
                                 self.source, reflector=reflector or FakeReflector(), workers=3,
                                 run_dir=self.root / name)

    def test_complete_readable_input_and_observations_have_no_middle_truncation(self):
        subtitles = "initial subtitle " * 900
        observation = "visible " * 1000 + "unique middle fact" + " tail" * 1000
        raw = {"question": "Question", "final_answer": "A", "messages": [
            {"role": "user", "content": [{"type": "text", "text": subtitles},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}]},
            {"role": "assistant", "content": "Inspect middle", "reasoning_content": "recorded reasoning",
             "tool_calls": [{"id": "t1", "function": {"name": "frame_inspect", "arguments": '{"timestamps":[1]}'}}],
             "model_call": {"input": "duplicate request"}, "usage": {"tokens": 4}, "provider_specific_fields": {"x": 1}},
            {"role": "tool", "tool_call_id": "t1", "name": "frame_inspect", "content": observation},
            {"role": "tool", "tool_call_id": "t2", "name": "frame_inspect", "content": "Tool execution failed."},
            {"role": "assistant", "content": "A"}]}
        readable = trajectory_for_reflection(raw)
        model = Mock()
        model.chat.return_value = {"content": '{"reason":"supported","answer_status":"correct","evidence_support":"sufficient","score":4,"reasoning_sound":true}'}
        reflector = Reflector(base_config(), model)
        reflector.classify({"trajectory": readable, "reference_answer": "REFERENCE"})
        prompt = model.chat.call_args.args[0][0]["content"]
        for text in (subtitles, observation, "recorded reasoning", "REFERENCE"):
            self.assertIn(text, prompt)
        for text in ("SECRET", "duplicate request", "provider_specific_fields", '"usage"'):
            self.assertNotIn(text, prompt)
        self.assertEqual(len(readable["messages"]), len(raw["messages"]))
        evidence = build_evidence(raw, "source")
        self.assertEqual(tool_observations(evidence), [{"tool_call_id": "t1", "tool": "frame_inspect", "content": observation}])
        model.chat.return_value = {"content": "null"}
        reflector.memory({"source": {}, "question": "Question"}, evidence)
        payload = json.loads(model.chat.call_args.args[0][0]['content'].split('\nInput:\n')[1])
        self.assertEqual(''.join(span['text'] for span in payload['observations']), observation)

    def test_context_limit_is_failed_once_and_does_not_request_corrective_json(self):
        error = RuntimeError("Internal error: context_length_exceeded")
        self.assertFalse(ModelClient._is_retryable(error))
        self.assertTrue(ModelClient._is_retryable(RuntimeError("internal server error")))
        model = Mock()
        model.chat.side_effect = error
        with self.assertRaises(RuntimeError):
            Reflector(base_config(), model).classify({"trajectory": {}})
        self.assertEqual(model.chat.call_count, 1)

    def test_memory_time_instructions_are_scoped_to_the_source_benchmark(self):
        model = Mock()
        model.chat.return_value = {"content": "null"}
        reflector = Reflector(base_config(), model)
        for benchmark in ("lvbench", "medvidbench", "egolife"):
            with self.subTest(benchmark=benchmark):
                source = {"benchmark": benchmark, "video": "video"}
                reflector.memory({"source": source, "question": "What happened?"}, {"messages": []})
                prompt = model.chat.call_args.args[0][0]["content"].split("\nInput:\n")[0]
                if benchmark == "egolife":
                    self.assertIn("DAY/clock", prompt)
                else:
                    self.assertNotIn("EgoLife", prompt)
                    self.assertNotIn("DAY/clock", prompt)
                self.assertNotIn('"context": "question this fact can help answer"', prompt)

    def test_skill_exports_show_one_final_skill_per_task_separate_from_checkpoints(self):
        for record in self.records:
            record["question_types"] = ["entity recognition", "event understanding"]
        reflector = FakeReflector()
        directory = self.root / "reflection"
        runner = ReflectionRunner(reflector, base_config())
        report, _ = runner.run(directory, self.source, self.records, {})
        files = sorted((directory / "skills").rglob("*.json"))
        self.assertEqual([str(path.relative_to(directory / "skills")) for path in files],
                         ["domain.json", "tasks/entity recognition.json", "tasks/event understanding.json"])
        for path in files:
            task = "__domain__" if path.name == "domain.json" else path.stem
            self.assertEqual(json.loads(path.read_text()), report.skills[task])
        checkpoints = list((directory / "checkpoints/skills/tasks/entity recognition").glob("group_*.json"))
        self.assertEqual(len(checkpoints), 2)
        self.assertTrue((directory / "checkpoints/skills/domain/draft.json").is_file())
        calls = len(reflector.calls)
        resumed, _ = runner.run(directory, self.source, self.records, {})
        self.assertEqual(reflector.calls[calls:], [])
        self.assertEqual(resumed.skills, report.skills)

    def test_missing_reference_and_max_samples_never_block_memory(self):
        self.records[0].pop("answer")
        cfg = base_config()
        cfg["evolving"]["reflector"]["sampling"]["max_samples"] = 1
        reflector = FakeReflector()
        report, _ = ReflectionRunner(reflector, cfg).run(self.root / "reflection", self.source, self.records, {})
        self.assertNotIn(("classify", "0"), reflector.calls)
        self.assertEqual(report.stages["classification"]["failed"], 1)
        self.assertEqual(len(report.selection["selected_sample_ids"]), 1)
        self.assertEqual({memory["source"]["sample_id"] for memory in report.memories}, {"0", "1", "2", "3"})

    def test_progress_advances_when_later_futures_finish_and_on_failure_or_reuse(self):
        from tqdm import tqdm as real_tqdm

        first_score_released = Event()
        bars = {}

        def progress_bar(*args, **kwargs):
            bar = real_tqdm(*args, **kwargs, mininterval=0, miniters=1)
            bars[kwargs["desc"]] = bar
            update = bar.update

            def advance(amount=1):
                result = update(amount)
                if bar.desc == "Classification" and bar.n == 3:
                    first_score_released.set()
                return result

            bar.update = advance
            return bar

        reflector = FakeReflector({("memory", "2")})
        classify = reflector.classify

        def delayed_score(item):
            if item["sample_id"] == "0" and not first_score_released.wait(timeout=3):
                raise RuntimeError("Progress waited for the first submitted future")
            return classify(item)

        reflector.classify = delayed_score
        runner = ReflectionRunner(reflector, base_config(), workers=3)
        with patch("videospy.evolving.reflector.runner.tqdm", side_effect=progress_bar):
            for attempt in range(2):
                report, _ = runner.run(self.root / "progress", self.source, self.records, {})
                self.assertEqual(report.stages["classification"]["failed"], 0)
                self.assertEqual(report.stages["memory"]["failed"], 1)
                self.assertEqual(bars["Classification"].n, 4)
                self.assertEqual(bars["Memory extraction"].n, 4)
                self.assertEqual(bars["Task Skill"].n, report.selection["group_count"])
                self.assertEqual(report.stages["domain_skill"]["succeeded"], 1)
                self.assertTrue(all(bar.disable for bar in bars.values()))
                if attempt:
                    self.assertEqual(report.stages["classification"]["reused"], 4)
                    self.assertEqual(report.stages["memory"]["reused"], 3)
        self.assertGreater(reflector.peak, 1)
        self.assertLessEqual(reflector.peak, 3)

    def test_sampling_config_applies_to_training_and_evaluation_after_manifest(self):
        from benchmark.egolife import run as egolife
        from benchmark.lvbench import run as lvbench
        from benchmark.medvidbench import run as medvidbench

        settings = {"lvbench": {"full": 2, "lite": 4, "debug": 7},
                    "egolife": {"full": 3, "lite": 5, "debug": 8},
                    "medvidbench": {"full": 4, "lite": 6, "debug": 9}}
        sampling = self.root / "sampling.yml"
        sampling.write_text(json.dumps(settings))
        indices = list(range(30, -1, -2))
        self.manifest.write_text(json.dumps(indices))
        with patch("benchmark.sampling.SAMPLING_CONFIG", sampling):
            for benchmark, runner in (("lvbench", lvbench), ("egolife", egolife), ("medvidbench", medvidbench)):
                source = self.root / benchmark
                write_source(source, benchmark, count=31)
                source_manifest = source / "run.json"
                recorded = json.loads(source_manifest.read_text())
                recorded["sample_keys"].reverse()
                source_manifest.write_text(json.dumps(recorded))
                for mode, stride in settings[benchmark].items():
                    with self.subTest(benchmark=benchmark, mode=mode):
                        config = base_config()
                        config["evolving"]["mode"] = mode
                        pipeline = EvolutionPipeline(benchmark, self.manifest, self.root / "output", config,
                                                     source, reflector=FakeReflector())
                        self.assertEqual(pipeline._prepare_training_inputs(), indices[::stride])
                        if benchmark == "medvidbench":
                            selected = runner.select_indices(31, mode, self.manifest)
                        else:
                            key = "uid" if benchmark == "lvbench" else "sample_key"
                            samples = [{key: str(i)} for i in range(31)]
                            selected = [int(s[key]) for s in runner.select_samples(samples, mode, self.manifest)]
                        self.assertEqual(selected, indices[::stride])

    def test_sampling_stride_rejects_invalid_values_and_changed_training_resume(self):
        from benchmark.sampling import get_sampling_stride

        sampling = self.root / "sampling.yml"
        with patch("benchmark.sampling.SAMPLING_CONFIG", sampling):
            for value in (0, -1, True, 1.5, "3", None):
                with self.subTest(value=value):
                    sampling.write_text(json.dumps({"lvbench": {"full": value}}))
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        get_sampling_stride("lvbench", "full")
            sampling.write_text("lvbench:\n  full: 1\n")
            reflector = FakeReflector()
            pipeline = self.pipeline(reflector)
            pipeline.run()
            calls = len(reflector.calls)
            sampling.write_text("lvbench:\n  full: 2\n")
            with self.assertRaisesRegex(ValueError, "Resume input changed: base_config_hash"):
                pipeline.run()
            self.assertEqual(reflector.calls[calls:], [])

    def test_legacy_initial_skills_are_rejected_before_writing_artifacts(self):
        initial = {"retrieval": [{"instruction": "OLD SKILL"}]}
        with self.assertRaisesRegex(ValueError, 'Legacy Skill'):
            ReflectionRunner(FakeReflector(), base_config()).run(
                self.root / 'legacy', self.source, self.records, {'skills': initial})
        self.assertFalse((self.root / 'legacy').exists())

    def test_fully_failed_core_stages_block_evaluation_but_export_other_successes(self):
        failures = {"classification": {("classify", str(i)) for i in range(4)},
                    "memory": {("memory", str(i)) for i in range(4)},
                    "task_skill": {("skill", "retrieval"), ("skill", "reasoning")},
                    "embedding": set()}
        for stage, nodes in failures.items():
            with self.subTest(stage=stage):
                pipeline = self.pipeline(FakeReflector(nodes), name=stage)
                if stage == "embedding":
                    pipeline.memory_optimizer.embedding_function = Mock(side_effect=ValueError("embedding unavailable"))
                run = pipeline.run()
                result = pipeline.result
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["evaluation_version"])
                self.assertGreater(result["stages"][stage]["failed"], 0)
                self.assertEqual(result["stages"][stage]["coverage"], 0)
                self.assertIsNone(json.loads((run / "final_version.json").read_text())["version_id"])
                if stage == "classification":
                    self.assertEqual(result["memory_count"], 4)
                elif stage in {"memory", "embedding"}:
                    self.assertGreater(result["skill_count"], 0)
        with patch("videospy.evolving.cli.EvolutionPipeline") as mocked:
            mocked.return_value.run.return_value = self.root
            mocked.return_value.result = {"status": "failed"}
            marker = self.root / "run-path"
            status = main(["run", "--benchmark", "lvbench", "--train-manifest", str(self.manifest),
                           "--inference-run-dir", str(self.source), "--run-dir-file", str(marker)])
            self.assertEqual(status, 1)
            self.assertFalse(marker.exists())

    def test_null_outputs_and_all_others_are_complete(self):
        reflector = FakeReflector()
        reflector.memory = Mock(return_value=None)
        cfg = base_config()
        pipeline = self.pipeline(reflector, cfg)
        pipeline.run()
        self.assertEqual(pipeline.result["status"], "complete")
        self.assertEqual(pipeline.result["memory_count"], 0)
        self.assertEqual(pipeline.result["stages"]["memory"]["empty"], 4)
        self.assertEqual(pipeline.result["stages"]["task_skill"]["failed"], 0)

    def test_pipeline_recovers_failed_nodes_and_versions_changed_artifacts(self):
        reflector = FakeReflector({("classify", "0"), ("memory", "2")})
        pipeline = self.pipeline(reflector)
        run = pipeline.run()
        old = copy.deepcopy(pipeline.result)
        self.assertEqual(old["status"], "partial")
        files = {p: p.read_bytes() for p in (run / "versions").rglob("*.json")}
        store = EvolutionStateStore(run / "evolution.sqlite3", read_only=True)
        old_skills = store.load_skills(old["version_id"])
        calls = len(reflector.calls)
        reflector.failures.clear()
        pipeline.run()
        result = pipeline.result
        self.assertEqual(result["status"], "complete")
        self.assertNotEqual(result["version_id"], old["version_id"])
        self.assertEqual([key for stage, key in reflector.calls[calls:] if stage == "classify"], ["0"])
        self.assertEqual([key for stage, key in reflector.calls[calls:] if stage == "memory"], ["2"])
        self.assertTrue(any(stage == "skill" for stage, _ in reflector.calls[calls:]))
        self.assertEqual(result["memory_count"], 4)
        self.assertEqual(store.load_skills(old["version_id"]), old_skills)
        self.assertEqual(files, {p: p.read_bytes() for p in files})
        calls = len(reflector.calls)
        version = result["version_id"]
        pipeline.run()
        self.assertEqual(reflector.calls[calls:], [])
        self.assertEqual(pipeline.result["version_id"], version)

    def test_repaired_source_reference_reuses_memory_and_other_scores(self):
        self.records[0].pop("answer")
        records_path = self.source / "records.jsonl"
        records_path.write_text("\n".join(map(json.dumps, self.records)))
        reflector = FakeReflector()
        pipeline = self.pipeline(reflector)
        pipeline.run()
        self.assertEqual(pipeline.result["status"], "partial")
        self.records[0]["answer"] = "A"
        records_path.write_text("\n".join(map(json.dumps, self.records)))
        calls = len(reflector.calls)
        pipeline.run()
        self.assertEqual(pipeline.result["status"], "complete")
        self.assertEqual([key for stage, key in reflector.calls[calls:] if stage == "classify"], ["0"])
        self.assertFalse(any(stage == "memory" for stage, _ in reflector.calls[calls:]))

    def test_reclassification_does_not_reuse_skill_reviewed_against_old_inputs(self):
        reflector = FakeReflector({("classify", "0")})
        pipeline = self.pipeline(reflector)
        run = pipeline.run()
        store = EvolutionStateStore(run / "evolution.sqlite3", read_only=True)
        previous = store.load_skills(pipeline.result["version_id"])
        reflector.failures = {("skill", "retrieval")}
        original_classify = reflector.classify
        reflector.classify = lambda item: original_classify(item) | {'evidence_support': 'partial', 'score': 2}
        for checkpoint in (run / 'reflection/classification').glob('*.json'):
            checkpoint.unlink()  # Reassess every source so all old group inputs are invalidated.
        pipeline.run()
        skills = store.load_skills(pipeline.result["version_id"])
        self.assertEqual(pipeline.result["status"], "partial")
        self.assertNotIn('retrieval', skills)
        self.assertIn('reasoning', skills)
        self.assertNotIn('__domain__', skills)
        self.assertGreater(pipeline.result["stages"]["task_skill"]["failed"], 0)
        reflector.failures.clear()
        pipeline.run()
        calls = len(reflector.calls)
        version = pipeline.result["version_id"]
        pipeline.run()
        self.assertEqual(reflector.calls[calls:], [])
        self.assertEqual(pipeline.result["version_id"], version)

    def test_fenced_null_is_a_successful_empty_reflection(self):
        model = Mock()
        model.chat.return_value = {"content": "```json\nnull\n```"}
        reflector = Reflector(base_config(), model)
        for operation in (
            lambda: reflector.update_skill("retrieval", [], None),
            lambda: reflector.domain_skill("lvbench", {}),
            lambda: reflector.memory({"source": {}, "question": "Question"}, {"messages": []}),
        ):
            model.chat.reset_mock()
            self.assertIsNone(operation())
            self.assertEqual(model.chat.call_count, 1)

    def test_resume_drops_stale_skills_when_changed_labels_produce_no_document(self):
        reflector = FakeReflector({("classify", "0")})
        pipeline = self.pipeline(reflector)
        run = pipeline.run()
        final = pipeline.result["version_id"]
        reflector.failures.clear()
        pipeline.memory_optimizer.embedding_function = Mock(side_effect=ValueError("embedding unavailable"))
        pipeline.run()
        self.assertEqual(pipeline.result["status"], "failed")
        store = EvolutionStateStore(run / "evolution.sqlite3", read_only=True)
        previous = store.load_skills(pipeline.result["version_id"])
        self.assertNotEqual(previous["retrieval"], store.load_skills(final)["retrieval"])
        # Changed outcome assessments invalidate old drafts and their reviews.
        for record in self.records:
            record["answer"] = "B"
        (self.source / "records.jsonl").write_text("\n".join(map(json.dumps, self.records)))
        reflector.classify = Mock(return_value={'reason': 'Problematic evidence', 'answer_status': 'incorrect', 'evidence_support': 'insufficient', 'score': 0, 'reasoning_sound': False})
        reflector.update_skill = Mock(return_value=None)
        reflector.domain_skill = Mock(return_value=None)
        pipeline.memory_optimizer.embedding_function = None
        pipeline.run()
        self.assertEqual(pipeline.result["status"], "complete")
        self.assertEqual(store.load_skills(pipeline.result["version_id"]), {})

    def test_medvidbench_dedup_uses_frame_sequence_and_fps(self):
        source = self.root / "medical_source"
        records = write_source(source, "medvidbench", count=4)
        # Real MedVidBench fields: each input frame sequence starts at local t=0.
        samples = []
        for start, fps in ((195, "0.5"), (546, "0.5"), (195, "0.5"), (195, "1.0")):
            samples.append({"data_source": "CoPESD", "video": [
                f"/root/data/CoPESD/high_light_images/014014/{frame:04d}.jpg"
                for frame in range(start, start + 4, 2)],
                "metadata": {"video_id": "014014", "fps": fps,
                             "input_video_start_frame": str(start), "input_video_end_frame": str(start + 2)}})
        metadata_path = source / "metadata.json"
        metadata_path.write_text(json.dumps(samples))
        manifest = json.loads((source / "run.json").read_text())
        manifest["metadata_path"] = str(metadata_path)
        (source / "run.json").write_text(json.dumps(manifest))
        for record in records:
            record["video_id"] = "014014"
        (source / "records.jsonl").write_text("\n".join(map(json.dumps, records)))
        reflector = FakeReflector()
        original = reflector.memory
        reflector.memory = lambda item, evidence: {**original(item, evidence), "content": "Forceps visible"}
        pipeline = EvolutionPipeline("medvidbench", self.manifest, self.root, base_config(), source,
                                     reflector=reflector, run_dir=self.root / "medical_run")
        run = pipeline.run()
        store = EvolutionStateStore(run / "evolution.sqlite3", read_only=True)
        memories = store.load_memories(pipeline.result["version_id"])
        self.assertEqual({row["source_sample_id"] for row in memories}, {"0", "1", "3"})
        for row in memories:
            source = row["metadata"]["source"]
            self.assertEqual(source["video"], "014014")
            self.assertEqual(source["clip"]["frame_count"], 2)
            self.assertIn("frame sequence", source["time_basis"])

        # New benchmark records carry the scope without depending on the dataset file.
        for record, sample in zip(records, samples):
            record["memory_source"] = medvidbench_source_context(sample)
        (pipeline.inference_run_dir / "records.jsonl").write_text("\n".join(map(json.dumps, records)))
        metadata_path.unlink()
        calls = len(reflector.calls)
        pipeline.run()
        self.assertEqual(reflector.calls[calls:], [])
        self.assertEqual(store.load_memories(pipeline.result["version_id"]), memories)

    def test_unknown_medvidbench_clips_do_not_merge_by_video_id(self):
        source = self.root / "unknown_clips"
        records = write_source(source, "medvidbench", count=3)
        for record in records:
            record["video_id"] = "014014"
        records[-1].pop("trajectory")
        with self.assertLogs("videospy.evolving.common.training_data", level="WARNING"):
            report, _ = ReflectionRunner(FakeReflector(), base_config()).run(
                self.root / "unknown_reflection", source, records, {})
        self.assertNotEqual(report.memories[0]["source"]["source_id"], report.memories[1]["source"]["source_id"])
        self.assertIn("unknown", report.memories[0]["source"]["time_basis"])
        self.assertEqual(report.stages["classification"]["failed"], 1)
        self.assertEqual(report.stages["memory"]["failed"], 1)

    def test_embedding_checkpoint_is_reused_and_corrupt_file_is_repaired(self):
        embed = Mock(return_value=[1.] + [0.] * 255)
        cache = CachedEmbedding(embed, base_config()["tools"]["memory_search"], self.root / "embeddings")
        cache("fact")
        cache("fact")
        self.assertEqual(embed.call_count, 1)
        next((self.root / "embeddings").glob("*.json")).write_text("broken JSON")
        cache("fact")
        self.assertEqual(embed.call_count, 2)

    def test_memory_availability_is_bound_to_source_not_model_output(self):
        source = self.root / "ego_source"
        records = write_source(source, "egolife", count=2)
        reflector = FakeReflector()
        original = reflector.memory

        def tampered(item, evidence):
            return {**original(item, evidence), "source": {"available_at": 0, "query_time": "earlier"}}

        reflector.memory = tampered
        report, _ = ReflectionRunner(reflector, base_config()).run(self.root / "ego_reflection", source, records, {})
        for memory, record in zip(report.memories, records):
            self.assertEqual(memory["source"]["query_time"], record["query_time"])
            self.assertEqual(memory["source"]["available_at"], egolife_seconds(record["query_time"]))

    def test_broken_record_is_skipped_with_source_failure_details(self):
        (self.source / "records.jsonl").write_text("broken JSON\n" + "\n".join(map(json.dumps, self.records[1:])))
        pipeline = self.pipeline()
        pipeline.run()
        self.assertEqual(pipeline.result["status"], "partial")
        self.assertEqual(pipeline.result["memory_count"], 3)
        failure = next(f for f in pipeline.result["failures"] if f["stage"] == "source")
        self.assertEqual(failure["node"], "0")
        self.assertIn("not complete", failure["reason"])


def memory_candidate(sid, video="video", start=1., end=3., content="The door opened", event="door opens", query_time=None):
    source = source_context("egolife" if query_time else "lvbench", video, query_time)
    source["sample_id"] = sid
    return {"source": source, "task_types": ["retrieval"], "context": "Which door opened?",
            "content": content, "event": event, "time": {"start_seconds": start, "end_seconds": end, "label": ""},
            "evidence_refs": [{"tool_call_id": "t", "quote": content}]}


class MemoryScopeTest(unittest.TestCase):
    def optimize(self, candidates):
        store = Mock()
        store.load_memories.return_value = []
        store.run_metadata.return_value = {"embedding_dimension": 2}
        store.memory_embedding_text = EvolutionStateStore.memory_embedding_text
        report = ReflectionReport({}, {}, candidates)
        optimizer = MemoryOptimizer(embedding_function=lambda _: [1., 0.])
        entries, summary = optimizer.optimize(store, "initial", report)
        return entries, summary

    def test_source_inspection_frames_survive_optimization_without_becoming_event_time(self):
        candidate = memory_candidate('one', start=None, end=None) | {'evidence_frames': [12.]}
        entries, _ = self.optimize([candidate])
        self.assertEqual(entries[0].metadata['evidence_frames'], [12.])
        self.assertIsNone(entries[0].metadata['time']['start_seconds'])

    def test_same_source_overlap_dedup_ignores_labels_but_preserves_disjoint_ranges(self):
        first = memory_candidate("1")
        same = memory_candidate("2", start=2., end=5., content="  THE door   OPENED ")
        same["time"]["label"] = "a different display label"
        similar = memory_candidate("3", start=3., end=6., content="A door swings open")
        disjoint = memory_candidate("4", start=6., end=7.)
        other_video = memory_candidate("5", video="another")
        entries, summary = self.optimize([first, same, similar, disjoint, other_video])
        self.assertEqual([e.source_sample_id for e in entries], ["1", "4", "5"])
        self.assertEqual(summary["duplicates"], 2)

    def test_unknown_time_only_exact_fact_and_normalized_event_are_deduplicated(self):
        entries, summary = self.optimize([
            memory_candidate("1", start=None, end=None),
            memory_candidate("2", event=" DOOR  Opens "),
            memory_candidate("3", start=None, end=None, content="A door swings open"),
            memory_candidate("4", start=None, end=None, event="another event")])
        self.assertEqual([e.source_sample_id for e in entries], ["1", "3", "4"])
        self.assertEqual(summary["duplicates"], 1)

    def test_egolife_earlier_availability_wins_over_input_order(self):
        early = memory_candidate("early", "/host/A1_JAKE", query_time={"date": "DAY1", "time": "09000000"})
        late = memory_candidate("late", "/another/A1_JAKE", query_time={"date": "DAY1", "time": "10000000"})
        entries, summary = self.optimize([late, early])
        self.assertEqual([entry.source_sample_id for entry in entries], ["early"])
        self.assertEqual(summary["duplicates"], 1)

    def test_unknown_egolife_availability_does_not_disable_exact_dedup(self):
        unknown = memory_candidate("unknown", "/host/A1_JAKE", start=None, end=None,
                                   query_time={"date": "DAY1", "time": "unknown"})
        early = memory_candidate("early", "/host/A1_JAKE", start=None, end=None,
                                 query_time={"date": "DAY1", "time": "09000000"})
        entries, summary = self.optimize([unknown, early])
        self.assertEqual([entry.source_sample_id for entry in entries], ["early"])
        self.assertEqual(summary["duplicates"], 1)

    def test_egolife_filters_future_and_unknown_sources_before_top_k(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_source(source)
            manifest = root / "train.json"
            manifest.write_text("[0,1]")
            pipeline = EvolutionPipeline("lvbench", manifest, root, base_config(), source, reflector=FakeReflector())
            pipeline._prepare_training_inputs()
            store = EvolutionStateStore(root / "test.sqlite3")
            store.create_run(pipeline._run_metadata())
            store.create_initial_version(pipeline.run_id, "initial", {}, {})
            def entry(sid, clock, video="/host/A1_JAKE", vector=(1., 0.)):
                candidate = memory_candidate(sid, video, query_time={"date": "DAY1", "time": clock})
                return MemoryEntry(sid, "retrieval", "question", "fact", sid,
                                   [*vector, *([0.] * 254)], {"source": candidate["source"], "time": candidate["time"]})
            entries = [entry("future", "10000000"), entry("unknown_time", "not a clock"),
                       entry("unknown_source", "08000000", "/unidentified"), entry("past", "09000000", vector=(.8, .6))]
            for entry in entries:
                entry.metadata["source"]["timeline_id"] = "map1"
            store.commit_version(pipeline.run_id, "initial", "final", 1, 1, 1, "h", ReflectionReport({}, {}, []),
                                 EvolutionUpdate({}, entries, {}, {}))
            context = source_context("egolife", "/eval/A1_JAKE", {"date": "DAY1", "time": "09000000"})
            context["timeline_id"] = "map1"
            matches = store.search_by_embedding("final", [1., *([0.] * 255)], None, 1, 0, context)
            self.assertEqual([match["source_sample_id"] for match in matches], ["past"])
            self.assertEqual(store.search_by_embedding("final", [1.] * 256, None, 3, 0), [])
            context["source_id"] = "egolife:another"
            self.assertEqual(store.search_by_embedding("final", [1.] * 256, None, 3, 0, context), [])

    def test_invalid_clock_is_not_comparable(self):
        for time in ("24000000", "09600000", "09006000", "09:00", "9000000"):
            self.assertIsNone(egolife_seconds({"date": "DAY1", "time": time}))
        self.assertEqual(egolife_seconds({"date": "DAY2", "time": "01000050"}), 90000.5)
