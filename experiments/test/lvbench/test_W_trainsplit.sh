#!/usr/bin/env bash
set -euxo pipefail

# Edit these values before running when needed.
mode="${MODE:-full}"
manifest="${MANIFEST:-experiments/data_splits/main/lvbench/train.json}"
output_dir="${OUTPUT_DIR:-output/experiments/test/lvbench/base/train_split}"
run_dir="${RUN_DIR:-}"
WORKERS="${WORKERS:-$((2 * ${NUM_GPUS:-1}))}"
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -w 'H100' >/dev/null; then
  WORKERS=$((WORKERS * 2))
fi
if [[ "$mode" != full && "$mode" != debug && "$mode" != lite ]]; then
  echo "MODE must be full, lite or debug" >&2
  exit 2
fi

if [[ -n "$run_dir" ]]; then
  destination_args=(--run-dir "$run_dir")
else
  destination_args=(--output-dir "$output_dir")
fi

# Let local non-Linux environments use Clash while accessing the campus intranet directly.
if [[ "$(uname -s)" != "Linux" ]]; then
  proxy_bypass_hosts="member.aicloud.szu.edu.cn,127.0.0.1,localhost"
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${proxy_bypass_hosts}"
  export no_proxy="${no_proxy:+${no_proxy},}${proxy_bypass_hosts}"
fi

python -m benchmark.lvbench.run \
  --agent videospy \
  --mode "$mode" \
  --sample-manifest "$manifest" \
  --workers "$WORKERS" \
  "${destination_args[@]}"
