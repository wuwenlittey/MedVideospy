"""Fixed model gateway for every RSI call, including retries and failures."""

import base64
import copy
import json
import re
import time
from io import BytesIO

from .io import public_config
from .records import Ledger


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        return value.split(",", 1)[0] + ",<image omitted>"
    return value


def image_metadata(messages):
    result = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if not url.startswith("data:"):
                    raise ValueError("RSI images must come from the legal reader")
                from PIL import Image
                import hashlib
                data = base64.b64decode(url.split(",", 1)[1], validate=True)
                with Image.open(BytesIO(data)) as frame:
                    result.append({"width": frame.width, "height": frame.height,
                                   "sha256": hashlib.sha256(data).hexdigest()})
    return result


class Gateway:
    def __init__(self, config, ledger_path, context):
        self.config = config
        self.ledger = Ledger(ledger_path, config["limits"])
        self.context = dict(context)
        self.calls = []

    def chat(self, role, messages, stage=None, **kwargs):
        stage = stage or self.context.get("stage", role)
        if self.context["scope"] != "final_test":
            self.ledger.check(self.context["scope"], self.context.get("branch") if self.context.get("stage") != "report" else None)
        images = image_metadata(messages)
        if any(i["width"] * i["height"] > self.config["limits"]["max_pixels"] for i in images):
            raise ValueError("Image input exceeds fixed pixel limit")
        recorded = compact(messages)
        if len(json.dumps(recorded, ensure_ascii=False)) > self.config["limits"]["max_input_characters"]:
            raise ValueError("Model input exceeds configured context-character bound")
        started = time.perf_counter()
        if self.config["backend"] == "demo":
            try:
                result = self._demo(role, messages, kwargs)
                # Synthetic costs are intentionally marked as such; no real API claim.
                inputs = 100 + sum((i["width"] * i["height"] + 255) // 256 for i in images)
                self._record(role, stage, recorded, result, images, inputs, 20, started, 1, None)
                return result
            except Exception as error:
                self._record(role, stage, recorded, {}, images, None, None, started, 1, str(error))
                raise
        from videospy.model import ModelClient, collect_model_calls
        model = copy.deepcopy(self.config["models"][role])
        model["seed"] = self.context.get("seed", self.config["seed"])
        calls = []
        try:
            with collect_model_calls() as calls:
                return ModelClient(model).chat(messages, **kwargs)
        finally:
            for call in calls:
                metrics = call["metrics"]
                self._record(role, stage, recorded, call.get("output", {}), images,
                             metrics.get("input_tokens"), metrics.get("output_tokens"), started,
                             metrics.get("api_attempt", 1), metrics.get("error"),
                             elapsed=metrics.get("elapsed_seconds"))
            if not calls:
                self._record(role, stage, recorded, {}, images, None, None, started, 1,
                             "Request ended without usage record")

    def _record(self, role, stage, inputs, output, images, itokens, otokens, started, attempt, error, elapsed=None):
        known = type(itokens) is int and type(otokens) is int and itokens >= 0 and otokens >= 0
        event = {**self.context, "role": role, "stage": stage, "input": inputs,
                 "phase": self.context.get("stage"),
                 "output": output, "images": images, "input_tokens": itokens, "output_tokens": otokens,
                 "total_tokens": itokens + otokens if known else None, "usage_complete": known,
                 "elapsed_seconds": elapsed if elapsed is not None else time.perf_counter() - started,
                 "attempt": attempt, "error": error, "synthetic": self.config["backend"] == "demo",
                 "model": public_config(self.config["models"][role])}
        event = self.ledger.append(event)
        self.calls.append(event)

    def request(self, kind, instruction, payload):
        role = "assessor" if kind == "assessment" else "outer" if kind == "meta_proposal" else "meta"
        response = self.chat(role, [{"role": "system", "content": instruction},
                                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                             stage=kind, return_json=True)
        content = response.get("content") or ""
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        return json.loads(content)

    def _demo(self, role, messages, kwargs):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if role == "task":
            if kwargs.get("tools") and not any(m.get("role") == "tool" for m in messages):
                available = {t["function"]["name"] for t in kwargs["tools"]}
                name = "overview" if "overview" in available else sorted(available)[0]
                parameters = {"query": "Describe the background color."}
                if name == "clip_skim":
                    parameters.update(start_time=0, end_time=10)
                if name == "frame_inspect":
                    parameters["timestamps"] = [0]
                return {"role": "assistant", "content": "Inspect visual evidence.", "tool_calls": [
                    {"id": "demo-tool", "type": "function", "function": {"name": name,
                     "arguments": json.dumps(parameters)}}]}
            return {"role": "assistant", "content": "A" if not kwargs.get("tools") else "The frames show red."}
        if role in {"overview", "clip_skim", "frame_inspect"}:
            result = {"content": "Facts: The sampled frames show a red background.\nInterpretation: red.\nUnknowns: none."}
        else:
            payload = json.loads(messages[-1]["content"])
            if "Assess" in messages[0]["content"]:
                value = {"evidence_support": "sufficient", "score": 4, "reason": "Synthetic visual evidence; redundant overview."}
            elif "Return JSON with hypotheses" in messages[0]["content"]:
                value = {"hypotheses": ["Uniform frames repeat the static background."],
                         "evidence_refs": [s["sample_id"] for s in payload.get("cases", [])],
                         "counterexamples": [], "unknowns": [], "investigations": []}
            elif role == "outer":
                source = payload["meta"]["files"]["task_proposer.py"].replace(
                    "TARGET_NUM_FRAMES=32", "TARGET_NUM_FRAMES=8")
                value = {"candidates": [{"layer": "meta", "parent_id": payload["meta"]["id"],
                         "hypothesis": "Propose smaller uniform scans for verified redundant global evidence.",
                         "files": {"task_proposer.py": source}, "evidence_refs": [], "risks": []}]}
            else:
                target = 8 if "TARGET_NUM_FRAMES=8" in text else 32
                value = {"candidates": [{"layer": "task", "parent_id": payload["task"]["id"],
                         "hypothesis": "Reduce duplicate sampled frames in a static scene.",
                         "files": {}, "parameters": {"tools.overview.num_frames": target},
                         "evidence_refs": [], "risks": ["Brief events may be missed."]}]}
            result = {"content": json.dumps(value)}
        return result
