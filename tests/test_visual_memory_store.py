"""Visual memory tests: synthetic images and mocked HTTP, never a model server."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np
from PIL import Image

from videospy.visual_memory import (
    EmbeddingClient, EmbeddingError, VisualMemoryStore, build_index, validate_case,
)


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class PixelEmbedder:
    identity = {"model": "synthetic-pixels", "dimension": 2}

    def encode(self, images):
        return np.asarray([[picture.getpixel((0, 0))[0] + 1,
                            picture.getpixel((0, 0))[1] + 1] for picture in images], dtype=np.float32)


class EmbeddingHTTPTest(unittest.TestCase):
    def setUp(self):
        self.images = [Image.new("RGB", (3, 2), (20, 0, 0)), Image.new("RGB", (3, 2), (0, 30, 0))]

    def client(self, **config):
        return EmbeddingClient({"api_base": "http://127.0.0.1:8000/v1", "max_retries": 0, **config})

    def test_default_vllm_chat_encodes_each_image_and_keeps_credentials_out_of_identity(self):
        with patch.dict(os.environ, {"TEST_EMBEDDING_KEY": "private-key", "TEST_BASE": "http://127.0.0.1:8000/v1"}):
            client = EmbeddingClient({"api_base_env": "TEST_BASE", "api_key_env": "TEST_EMBEDDING_KEY"})
            with patch("urllib.request.urlopen", side_effect=[Response({"data": [{"index": 0, "embedding": [1, 2]}]}), Response({"data": [{"index": 0, "embedding": [3, 4]}]})]) as request:
                actual = client.encode(self.images)
        np.testing.assert_array_equal(actual, [[1, 2], [3, 4]])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(client.request_count, 2)
        payload = json.loads(request.call_args_list[0].args[0].data)
        self.assertEqual(payload["encoding_format"], "float")
        self.assertEqual(payload["model"], "WeMM-Embedding-2B")
        url = payload["messages"][0]["content"][0]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as picture:
            self.assertEqual(picture.size, (3, 2))
        self.assertEqual(client.identity["dimension"], 2)
        self.assertNotIn("private-key", json.dumps(client.identity))
        self.assertNotIn("TEST_EMBEDDING_KEY", json.dumps(client.identity))

    def test_openai_batch_sorts_provider_indices(self):
        client = self.client(protocol="openai", model="images")
        with patch("urllib.request.urlopen", return_value=Response({"data": [
            {"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]},
        ]})) as request:
            actual = client.encode(self.images)
        np.testing.assert_array_equal(actual, [[3, 0], [0, 2]])
        self.assertEqual(len(json.loads(request.call_args.args[0].data)["input"]), 2)

    def test_custom_template_path_and_per_image(self):
        client = self.client(protocol="custom", per_image=True,
                             request_template={"name": "{model}", "pictures": "{image_urls}"},
                             response_path="payload.rows", vector_key="values")
        with patch("urllib.request.urlopen", side_effect=[Response({"payload": {"rows": [{"values": [1, 2]}]}}), Response({"payload": {"rows": [{"values": [2, 1]}]}})]) as request:
            self.assertEqual(client.encode(self.images).shape, (2, 2))
        self.assertEqual(len(json.loads(request.call_args.args[0].data)["pictures"]), 1)

    def test_custom_image_item_template(self):
        client = self.client(protocol="custom", per_image=True,
                             request_template={"model": "{model}", "input": "{images}"},
                             image_item_template={"image": "{image_url}"})
        with patch("urllib.request.urlopen", return_value=Response({"data": [[1, 2]]})) as request:
            client.encode(self.images[:1])
        self.assertTrue(json.loads(request.call_args.args[0].data)["input"][0]["image"].startswith("data:image/png"))

    def test_rejects_bad_counts_indices_nonfinite_zero_and_dimension_changes(self):
        payloads = [
            {"data": []}, {"data": [{"index": 1, "embedding": [1, 2]}]},
            {"data": [[0, 0]]}, {"data": [[float("nan"), 1]]},
            {"data": [["one", "two"]]}, {"data": [[True, False]]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), patch("urllib.request.urlopen", return_value=Response(payload)):
                with self.assertRaises(EmbeddingError):
                    self.client().encode(self.images[:1])
        client = self.client()
        with patch("urllib.request.urlopen", side_effect=[Response({"data": [[1, 2]]}), Response({"data": [[1, 2, 3]]})]):
            client.encode(self.images[:1])
            with self.assertRaisesRegex(EmbeddingError, "dimension"):
                client.encode(self.images[:1])

    def test_retry_and_no_hash_or_local_fallback(self):
        error = urllib.error.HTTPError("http://127.0.0.1", 503, "unavailable", {}, None)
        client = self.client(max_retries=1)
        with patch("urllib.request.urlopen", side_effect=[error, Response({"data": [[1, 2]]})]) as request, patch("time.sleep"):
            self.assertEqual(client.encode(self.images[:1]).shape, (1, 2))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(client.request_count, 2)
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(EmbeddingError, "HTTP 503"):
                self.client().encode(self.images[:1])


class VisualMemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="visual-memory-test-", dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        Image.new("RGB", (8, 6), (100, 0, 0)).save(self.root / "red.png")
        Image.new("RGB", (8, 6), (0, 10, 0)).save(self.root / "green.png")
        self.source = self.root / "cases.jsonl"
        self.output = self.root / "index"

    def case(self, case_id="case1", **changes):
        record = {"case_id": case_id, "question": "Which instrument is visible?",
                  "answer_text": "A needle driver is visible.", "source_video_id": "video-" + case_id,
                  "split": "train", "task_type": "instrument", "procedure": "surgery",
                  "evidence_scope": "frame", "timestamps": [1], "image_paths": ["red.png"],
                  "verified": True, "verification_note": "Reviewed the original frame."}
        record.update(changes)
        return record

    def build(self, cases, embedder=None, **config):
        self.source.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
        return build_index(self.source, self.output, config, embedder=embedder or PixelEmbedder())

    def context(self, **changes):
        result = {"source_video_id": "query-video", "task_type": "instrument", "procedure": "surgery"}
        result.update(changes)
        return result

    def test_build_copies_originals_and_mean_is_individually_normalized(self):
        manifest = self.build([self.case(evidence_scope="clip", timestamps=[1, 2], image_paths=["red.png", "green.png"])])
        store = VisualMemoryStore(self.output)
        first, second = np.asarray([101, 1]), np.asarray([1, 11])
        expected = first / np.linalg.norm(first) + second / np.linalg.norm(second)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(store.vectors[0], expected, rtol=1e-5)
        self.assertEqual(manifest["dimension"], 2)
        self.assertFalse(store.vectors.flags.writeable)
        (self.root / "red.png").unlink()
        matches = store.search([1, 1], context=self.context())["matches"]
        self.assertEqual(len(matches), 1)
        self.assertTrue(Path(matches[0]["case"]["image_paths"][0]).is_file())
        self.assertTrue(Path(matches[0]["case"]["image_paths"][0]).is_absolute())

    def test_schema_rejects_unverified_test_split_letter_label_bad_scope_and_time(self):
        changes = [{"verified": False}, {"split": "test"}, {"verification_note": ""},
                   {"answer_text": "D"}, {"source_video_id": ""}, {"task_type": ""},
                   {"timestamps": [-1]}, {"timestamps": [True]},
                   {"timestamps": [1, 1], "image_paths": ["red.png", "green.png"], "evidence_scope": "clip"},
                   {"evidence_scope": "clip"}, {"image_paths": []}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_case(self.case(**change))

    def test_same_source_patient_sample_and_video_are_excluded(self):
        cases = [self.case("video", source_video_id="query-video"),
                 self.case("source", source_id="query-source"),
                 self.case("patient", patient_id="query-patient"),
                 self.case("sample", source_sample_id="query-sample"), self.case("valid")]
        self.build(cases)
        store = VisualMemoryStore(self.output)
        result = store.search([1, 0], context=self.context(source_id="query-source", patient_id="query-patient", source_sample_id="query-sample"))
        self.assertEqual([item["case_id"] for item in result["matches"]], ["valid"])
        self.assertEqual(len(result["exclusions"]), 4)

    def test_task_procedure_threshold_and_stable_one_case_per_video(self):
        self.build([self.case("b", source_video_id="same"), self.case("a", source_video_id="same"),
                    self.case("wrongtask", task_type="phase"), self.case("wrongproc", procedure="other"),
                    self.case("unknownproc", procedure=None)])
        store = VisualMemoryStore(self.output)
        result = store.search([1, 0], context=self.context(), top_k=5)
        self.assertEqual([item["case_id"] for item in result["matches"]], ["a", "unknownproc"])
        exclusions = {item["case_id"]: item["reasons"] for item in result["exclusions"]}
        self.assertEqual(exclusions["b"], ["duplicate_source_video"])
        self.assertIn("task_type_mismatch", exclusions["wrongtask"])
        self.assertIn("procedure_mismatch", exclusions["wrongproc"])
        self.assertEqual(store.search([0, 1], context=self.context(), min_similarity=0.5)["matches"], [])

    def test_context_and_query_dimension_are_mandatory(self):
        self.build([self.case()])
        store = VisualMemoryStore(self.output)
        for context in ({}, {"task_type": "instrument"}, {"source_video_id": "query"}):
            with self.subTest(context=context), self.assertRaises(ValueError):
                store.search([1, 0], context=context)
        for vector in ([1, 2, 3], [0, 0], [float("inf"), 0], [[1, 0], [0, 1]]):
            with self.subTest(vector=vector), self.assertRaises(EmbeddingError):
                store.search(vector, context=self.context())

    def test_refuses_existing_output_even_empty_and_cleans_failed_build(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.build([self.case()])
        self.output.rmdir()
        bad = PixelEmbedder()
        bad.encode = lambda images: np.zeros((len(images), 2))
        with self.assertRaises(EmbeddingError):
            self.build([self.case()], bad)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".index.building-*")), [])

    def test_rejects_duplicate_cases_and_cancelling_mean(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build([self.case(), self.case()])
        bad = PixelEmbedder()
        bad.encode = lambda images: np.asarray([[1, 0], [-1, 0]], dtype=np.float32)
        with self.assertRaisesRegex(EmbeddingError, "cancel"):
            self.build([self.case(evidence_scope="clip", timestamps=[1, 2], image_paths=["red.png", "green.png"])], bad)

    def test_detects_tampered_record_hashes(self):
        self.build([self.case()])
        records = self.output / "records.jsonl"
        records.write_bytes(records.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            VisualMemoryStore(self.output)

    def test_detects_tampered_media_hash(self):
        self.build([self.case()])
        record = json.loads((self.output / "records.jsonl").read_text())
        path = self.output / record["image_paths"][0]
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            VisualMemoryStore(self.output)

    def test_hashes_do_not_allow_path_traversal(self):
        self.build([self.case()])
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"]["../cases.jsonl"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "contained"):
            VisualMemoryStore(self.output)

    def test_search_result_cannot_mutate_stored_timestamps(self):
        self.build([self.case()])
        store = VisualMemoryStore(self.output)
        result = store.search([1, 0], context=self.context())
        result["matches"][0]["case"]["timestamps"][0] = 999
        self.assertEqual(store.search([1, 0], context=self.context())["matches"][0]["case"]["timestamps"], [1.0])

    def test_checks_vector_manifest_dimension_even_with_updated_hash(self):
        self.build([self.case()])
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["dimension"] = 3
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(EmbeddingError, "dimension"):
            VisualMemoryStore(self.output)

    def test_maximum_image_limit_is_explicit_and_configurable(self):
        case = self.case(evidence_scope="clip", timestamps=list(range(7)), image_paths=["red.png"] * 7)
        self.assertEqual(len(validate_case(case)["image_paths"]), 7)
        with self.assertRaisesRegex(ValueError, "reference images"):
            self.build([case], max_images_per_case=6)


if __name__ == "__main__":
    unittest.main()
