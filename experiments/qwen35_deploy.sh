#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate llm_deploy

# ============ 任务配置 ============
MODEL_NAME=${MODEL_NAME:-"Qwen3.5-4B"}
MODEL_PATH=${MODEL_PATH:-"/share/home/tm1005929190170000/a1005928270/myworkspace/models/Qwen/${MODEL_NAME}"}
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

echo "==== Detected $NUM_GPUS GPU(s), LLM deploy tp=$TP, MODEL_NAME=$MODEL_NAME  ===="
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
    --tool-call-parser qwen3coder
