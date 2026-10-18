#!/usr/bin/env bash
set -euxo pipefail

# Edit these values before running when needed.
mode="${MODE:-full}"
manifest="${MANIFEST:-experiments/data_splits/main/lvbench/train.json}"
inference_run_dir="${INFERENCE_RUN_DIR:-output/experiments/test/lvbench/base/train_split/videospy/lvbench/full_openai_Qwen3.5-4B_20260913_205559_a8eb6c9}"
config="${CONFIG:-experiments/train/train_configs/self_full.yml}"
output_dir="${OUTPUT_DIR:-output/experiments/train/lvbench/${mode}}"
run_dir="${RUN_DIR:-}"
test_manifest="${TEST_MANIFEST:-experiments/data_splits/main/lvbench/test.json}"
WORKERS="${WORKERS:-$((2 * ${NUM_GPUS:-1}))}"
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -w 'H100' >/dev/null; then
  WORKERS=$((WORKERS * 2))
fi
if [[ "$mode" != full && "$mode" != debug && "$mode" != lite ]]; then
  echo "MODE must be full, lite or debug" >&2
  exit 2
fi

if [[ ! -f "$inference_run_dir/run.json" ]]; then
  echo "Offline inference run not found: $inference_run_dir" >&2
  exit 2
fi
if [[ -n "$run_dir" ]]; then
  if [[ ! -f "$run_dir/evolution.sqlite3" ]]; then
    echo "Not an existing Evolution run: $run_dir" >&2
    exit 2
  fi
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

# Receive this run's actual directory without parsing logs or scanning old runs.
run_dir_file="$(mktemp)"
trap 'rm -f "$run_dir_file"' EXIT
python -m videospy.evolving.cli run \
  --benchmark lvbench --mode "$mode" \
  --train-manifest "$manifest" \
  --inference-run-dir "$inference_run_dir" \
  --config "$config" \
  --workers "$WORKERS" \
  --run-dir-file "$run_dir_file" \
  "${destination_args[@]}"
IFS= read -r evolution_run_dir < "$run_dir_file"

# 或者指定特定目录进行评测: Run the evolution with the specified configuration and training manifest
# evolution_run_dir="output/experiments/train/lvbench/lite/external_openai_Qwen3.5-4B_20260922_181600_bcd46de"

evolution_db="$evolution_run_dir/evolution.sqlite3"
if [[ ! -f "$evolution_db" || ! -f "$evolution_run_dir/final_version.json" ]]; then
  echo "Evolution has not finished: $evolution_run_dir" >&2
  exit 2
fi

# Run every evaluation before reporting a nonzero exit status.
evaluation_status=0
python -m benchmark.lvbench.run \
  --agent videospy --mode "$mode" \
  --config "$config" \
  --config experiments/test/test_configs/skill_only.yml \
  --evolution-db "$evolution_db" \
  --sample-manifest "$test_manifest" --workers "$WORKERS" \
  --run-dir "$evolution_run_dir/evaluations/skill_only/$mode" || evaluation_status=$?

# Evaluate the base agent without loading evolution state.
python -m benchmark.lvbench.run \
  --agent videospy --mode "$mode" \
  --config "$config" \
  --sample-manifest "$test_manifest" --workers "$WORKERS" \
  --run-dir "$evolution_run_dir/evaluations/base/$mode" || evaluation_status=$?

python -m benchmark.lvbench.run \
  --agent videospy --mode "$mode" \
  --config "$config" \
  --config experiments/test/test_configs/memory_only.yml \
  --evolution-db "$evolution_db" \
  --sample-manifest "$test_manifest" --workers "$WORKERS" \
  --run-dir "$evolution_run_dir/evaluations/memory_only/$mode" || evaluation_status=$?

python -m benchmark.lvbench.run \
  --agent videospy --mode "$mode" \
  --config "$config" \
  --config experiments/test/test_configs/skill_memory.yml \
  --evolution-db "$evolution_db" \
  --sample-manifest "$test_manifest" --workers "$WORKERS" \
  --run-dir "$evolution_run_dir/evaluations/skill_memory/$mode" || evaluation_status=$?

# show results
find "$evolution_run_dir" -name "report.md" | while read -r f; do echo -e "\n\n==== ${f} ===="; cat "$f"; done
exit "$evaluation_status"
