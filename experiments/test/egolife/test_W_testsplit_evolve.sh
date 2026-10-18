#!/usr/bin/env bash
set -euxo pipefail

# Edit these values before running when needed.
mode="${MODE:-full}"
manifest="${MANIFEST:-experiments/data_splits/main/egolife/test.json}"
evolution_run_dir="${EVOLUTION_RUN_DIR:-/path/to/evolution/run}"
evolution_db="${EVOLUTION_DB:-$evolution_run_dir/evolution.sqlite3}"
config="${CONFIG:-experiments/test/test_configs/skill_memory.yml}"
output_dir="${OUTPUT_DIR:-$evolution_run_dir/evaluations}"
run_dir="${RUN_DIR:-$output_dir/$(basename "$config" .yml)/$mode}"
WORKERS="${WORKERS:-${NUM_GPUS:-1}}"
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -w 'H100' >/dev/null; then
  WORKERS=$((WORKERS * 2))
fi
if [[ "$mode" != full && "$mode" != debug && "$mode" != lite ]]; then
  echo "MODE must be full, lite or debug" >&2
  exit 2
fi

if [[ ! -f "$evolution_db" || ! -f "$(dirname "$evolution_db")/final_version.json" ]]; then
  echo "Evolution has not finished: $evolution_db" >&2
  exit 2
fi

# Let local non-Linux environments use Clash while accessing the campus intranet directly.
if [[ "$(uname -s)" != "Linux" ]]; then
  proxy_bypass_hosts="member.aicloud.szu.edu.cn,127.0.0.1,localhost"
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${proxy_bypass_hosts}"
  export no_proxy="${no_proxy:+${no_proxy},}${proxy_bypass_hosts}"
fi

python -m benchmark.egolife.run \
  --agent videospy \
  --mode "$mode" \
  --config "$config" \
  --evolution-db "$evolution_db" \
  --sample-manifest "$manifest" \
  --workers "$WORKERS" \
  --run-dir "$run_dir"
