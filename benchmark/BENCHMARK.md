# Benchmark

## 快速验证

```bash
python -m benchmark.lvbench.run --agent videospy --mode debug

python -m benchmark.egolife.run --agent videospy --mode debug

# 本地评测需要
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m benchmark.medvidbench.run --agent videospy --mode debug --evaluate

python -m benchmark.streameqa.run --agent videospy --mode smoke --split Perception

python -m benchmark.streamingbench.run --agent videospy --mode smoke
```

各数据集的参数与输出格式见对应 README。复现 `main.md` 第 4 节的基础实验、消融、
三领域进化和最终评测时，使用
[experiments/README.md](../experiments/README.md)。
