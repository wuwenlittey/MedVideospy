#!/bin/bash
set -euxo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate llm_deploy

# ============ 任务配置 ============
# 训练示例：experiments/train/lvbench/train_offline.sh
# 测试示例：experiments/test/lvbench/test_W_trainsplit.sh
TASK_SCRIPT="experiments/train/lvbench/train_offline.sh"

# full, lite, debug
MODE=full

MODEL_NAME="Qwen3.5-4B"
MODEL_PATH="/myworkspace/models/Qwen/${MODEL_NAME}"
SERVER_PORT=23333
TP=1
# =================================

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=$((NUM_GPUS))
if (( NUM_GPUS < 1 )); then
    echo "No GPU detected." >&2
    exit 2
fi
if (( TP < 1 || TP > NUM_GPUS || (TP & (TP - 1)) != 0 )); then
    echo "TP must be a power of two between 1 and NUM_GPUS: $TP" >&2
    exit 2
fi
export NUM_GPUS

echo "==== Detected $NUM_GPUS GPU(s), LLM deploy tp=$TP ===="
lmdeploy serve api_server \
    "$MODEL_PATH" \
    --backend pytorch \
    --tp "$TP" \
    --server-name 0.0.0.0 \
    --server-port "$SERVER_PORT" \
    --model-name "$MODEL_NAME" \
    --max-batch-size 8 \
    --cache-max-entry-count 0.8 \
    --trust-remote-code \
    --reasoning-parser default \
    --tool-call-parser qwen3coder \
    >/tmp/lmdeploy.log 2>&1 &

lmdeploy_pid=$!
trap 'echo "==== Stop LLM deploy ===="; kill "$lmdeploy_pid" 2>/dev/null || true' EXIT

echo "==== Wait for API server ===="
curl -fs \
    --retry 60 \
    --retry-delay 5 \
    --retry-connrefused \
    "http://127.0.0.1:${SERVER_PORT}/v1/models" \
    >/dev/null

echo "==== LLM deploy is ready ===="

# Adapt your tasks
echo "==== Run VideoSpy task ===="
conda activate base
cd /myworkspace/code/videospy
git config --global --add safe.directory /myworkspace/code/videospy

MODE="$MODE" NUM_GPUS="$NUM_GPUS" PYTHONUNBUFFERED=1 bash "$TASK_SCRIPT"
echo "==== VideoSpy task completed ===="
