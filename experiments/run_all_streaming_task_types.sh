#!/usr/bin/env bash
# Two questions from every multiple-choice task type.
# StreamEQA items with fewer than four options, a negative timestamp,
# or no query time are dropped by benchmark/streameqa/run.py before any model call.
set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="${STREAMING_BENCHMARKS_ROOT:-/share/home/tm1005929190170000/a1111547840/xy/streaming-video-benchmarks/streaming_video_benchmark_package}"
export STREAMING_BENCHMARKS_ROOT="$PKG"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-llm_deploy}"
fi

PORT="${LMDEPLOY_PORT:-23333}"
if ! ss -lnt | grep -q ":${PORT} "; then
  echo "lmdeploy is not listening on 127.0.0.1:${PORT}" >&2
  exit 1
fi

python - << 'PY'
from collections import Counter
from pathlib import Path
import os
import sys

from benchmark.streameqa.run import load_dataset as load_eqa
from benchmark.streameqa.run import resolve_video as resolve_eqa
from benchmark.streamingbench.run import load_dataset as load_sb
from benchmark.streamingbench.run import resolve_video as resolve_sb
from benchmark.streaming_runner import select_samples

pkg = Path(os.environ["STREAMING_BENCHMARKS_ROOT"])
ann = pkg / "official_repositories/StreamEQA/data/annotations"
sb_data = pkg / "official_repositories/StreamingBench/src/data"
eqa_video = pkg / "datasets/hd_epic/videos"
sb_video = pkg / "datasets/streamingbench/videos"
expected = {
    "Perception": 13,
    "Interaction": 16,
    "Planning_all": 13,
}
print("=== StreamEQA selection ===")
for split, task_count in expected.items():
    samples = select_samples(load_eqa(ann / f"{split}.json", split), "smoke", None)
    counts = Counter(item["category"] for item in samples)
    print(f"{split} selected={len(samples)} task_types={len(counts)}")
    if len(counts) != task_count or any(count != 2 for count in counts.values()):
        sys.exit(f"{split} did not select 2 questions for every task type: {dict(counts)}")
    resolve_eqa(samples[0], eqa_video)
print("=== StreamingBench selection ===")
for name, task_count in (
    ("questions_real.json", 10),
    ("questions_omni.json", 6),
    ("questions_sqa.json", 1),
):
    samples = select_samples(load_sb(sb_data / name), "smoke", None)
    counts = Counter(item["category"] for item in samples)
    print(f"{name} selected={len(samples)} task_types={len(counts)} {sorted(counts)}")
    if len(counts) != task_count or any(count != 2 for count in counts.values()):
        sys.exit(f"{name} did not select 2 questions for every task type: {dict(counts)}")
    resolve_sb(samples[0], sb_video)
proactive = load_sb(sb_data / "questions_proactive.json")
print(f"questions_proactive.json usable_four_choice={len(proactive)} (expected 0, not a multiple-choice task)")
if proactive:
    sys.exit("Proactive questions were not filtered out")
print("selection_ok")
PY

run_eqa() {
  local split="$1"
  echo "start_streameqa_${split} $(date -Is)"
  python -m benchmark.streameqa.run \
    --agent videospy \
    --mode smoke \
    --split "$split" \
    --qa-path "$PKG/official_repositories/StreamEQA/data/annotations/${split}.json" \
    --video-root "$PKG/datasets/hd_epic/videos" \
    --workers 1
}

run_sb() {
  local name="$1"
  echo "start_streamingbench_${name} $(date -Is)"
  python -m benchmark.streamingbench.run \
    --agent videospy \
    --mode smoke \
    --qa-path "$PKG/official_repositories/StreamingBench/src/data/${name}" \
    --video-root "$PKG/datasets/streamingbench/videos" \
    --workers 1
}

run_eqa Perception
run_eqa Interaction
run_eqa Planning_all
run_sb questions_real.json
run_sb questions_omni.json
run_sb questions_sqa.json
echo "done $(date -Is)"
