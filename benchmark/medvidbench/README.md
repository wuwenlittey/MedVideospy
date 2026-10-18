# MedVidBench

MedVidBench 包含两个数据源，由 manifest 选择：

- `test`：`MedVidU_ECCV2026_TrainVal`，包含 Ground Truth，支持 full、lite、debug 和本地评测。
- `leaderboard_test`：`MedVidBench-data`，没有 Ground Truth，只支持 full，结果提交到网页 Leaderboard。

## 运行

test 默认运行 full：

```bash
# 本地评测需要
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python benchmark/medvidbench/run.py --agent videospy --evaluate
python benchmark/medvidbench/run.py --agent videospy --evaluate --mode lite
python benchmark/medvidbench/run.py --agent videospy --evaluate --mode full
```

先按`--sample-manifest`选择样本索引并保留清单顺序，再按 [config/benchmark_sampling.yml](../../config/benchmark_sampling.yml) 的 medvidbench 项取样，训练和评测共用。默认取
`debug: indices[::100]`、`lite: indices[::10]`（约1/10），`full`保留全部。
未传manifest时默认使用`experiments/data_splits/main/medvidbench/test.json`。
公开数据的 `train.json`、`test.json` 保持 JSON 索引列表格式；对象格式可通过
`dataset` 指定数据源，通过 `sample_indices` 指定索引列表或 `"all"`。
隐藏集的 `leaderboard_test.json` 内容如下，不会套用公开测试集的索引或采样步长：

```json
{"dataset": "leaderboard_test", "sample_indices": "all"}
```

不再接受 `--split`、`--dataset` 参数，数据源与样本统一由 manifest 决定。
例如原test manifest的1124题对应debug 12题、lite 113题。切片不保证类别均衡。

```bash
python -m benchmark.medvidbench.run --agent videospy --mode debug \
  --sample-manifest experiments/data_splits/main/medvidbench/test.json --evaluate
```

leaderboard_test 自动使用 full：

```bash
python -m benchmark.medvidbench.run --agent videospy \
  --sample-manifest experiments/data_splits/main/medvidbench/leaderboard_test.json
```

VideoSpy 的固定配置来自 `config/videospy/default.yml`。MedVidBench 不维护专属
VideoSpy 配置，基础运行使用默认抽帧参数和空的人工初始 Skill。

VideoSpy 每次只注入当前问题对应的 Skill，避免不同任务相互干扰。
`dense_captioning_*`、`region_caption_*` 和 `video_summary_*` 会分别使用各自共享的
Skill。固定输出格式仍由 Benchmark 的问题构造代码提供，不随进化改变。

可以通过 `--evolution-db` 使用训练得到的版本化 SQLite：

```bash
python benchmark/medvidbench/run.py \
  --agent videospy \
  --evolution-db output/evolving/medvidbench/<run_id>/evolution.sqlite3 \
  --mode full
```

SQLite 只覆盖 Skill、Memory 和白名单执行参数。`--evolution-version` 可以指定
明确版本或 `latest`。Agent 配置与下文用于 LLM Judge 的 Evaluation 配置
相互独立。

目录规则与 LVBench 一致：

```text
output/<agent>/medvidbench/<mode>_<model>_<timestamp>/
```

## Run 目录

核心 JSON 产物只保留四类：

- `run.json`：运行配置和样本列表，用于安全续跑。
- `records.jsonl`：逐题检查点、Prediction、轨迹路径和运行消耗。test 还包含 question、Ground Truth 和 `struc_info`，可直接逐题对比。
- `submission.json`：网页提交格式，只包含 `id`、`qa_type`、`prediction`。
- `evaluation.json`：本地评测结果、完整性状态和 LLM 配置，仅在本地评测后生成。

此外还有便于阅读的 `report.md`、日志文件和 `trajectories/`。leaderboard_test 的
`records.jsonl` 不写入 Ground Truth。

## 本地评测

仅 test 支持本地评测：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python benchmark/medvidbench/evaluate.py \
  --run-dir output/<agent>/medvidbench/<run_dir_name> \
  --skip-llm-judge
```

`--skip-llm-judge` 跳过 DVC、Video Summary、Region Caption 的 LLM 指标，
但仍要求其余 7 个确定性指标完整。缺依赖、任务异常或指标缺失都会使评测返回失败，
具体原因写入 `evaluation.json` 和 `evaluation.log`。

评测只读取仓库中的 `benchmark/medvidbench/config/evaluation.yml`：

```yaml
evaluation:
  llm:
    model: gpt-4.1
    api_base: https://example.com/v1
    api_key: <api-key>
```

本地 OpenAI 兼容服务也使用 `model` 字段，值必须与服务 `/v1/models` 返回的模型 ID 一致。
这里直接使用 OpenAI SDK，不会解析 Agent 的 LiteLLM `openai/` 路由前缀；
例如服务模型 ID 为 `Qwen3.5-4B` 时，填写 `model: Qwen3.5-4B`。

配置完成后直接运行：

```bash
python benchmark/medvidbench/evaluate.py \
  --run-dir output/<agent>/medvidbench/<run_dir_name>
```

评测模型配置与 Agent 配置相互独立，Key 不会写入 run 目录。首次评测前安装官方依赖：

```bash
pip install -r data/MedVidBench/MedVidBench-Leaderboard/requirements.txt
```

`report.md` 按网页顺序展示 10 个官方指标：`CVS_acc`、`NAP_acc`、`SA_acc`、
`STG_mIoU`、`TAG_mIoU@0.3`、`TAG_mIoU@0.5`、`DVC_F1`、`DVC_llm`、
`VS_llm`、`RC_llm`。Run Statistics 中的 `Generated` 仅表示样本生成成功，不表示
答案正确。

test 的本地评测还会计算诊断指标 `NAP_exact_acc`。它对 Next Action 的完整
标签做大小写不敏感和空白规范化后的精确匹配，不使用语义映射，也不替代官方
`NAP_acc`。如果两者不同，`evaluation.json` 和 `report.md` 会保留两个结果并给出
警告。这可以识别 小规模子集中只有一个类别时，官方语义映射将任意预测计为正确的
情况。

## Prediction 如何解析

官方评测器主要使用正则表达式从模型原始文本提取结构化结果，不使用额外 LLM：

- TAL：正则提取 `start-end`，找不到时再尝试 `start to end` 和时分秒区间，然后转换为浮点时间段。
- STG：正则提取 `timestamp seconds: [x1, y1, x2, y2]`，坐标按逗号切分并转换为浮点数。
- DVC：正则逐行提取 `start-end seconds: caption`，得到开始时间、结束时间和描述文本。
- Skill Assessment：按逗号切分，再用正则提取 `维度名: N/5`。
- CVS Assessment：用三个固定名称的正则提取整数分数。
- Next Action：完整文本先做类别精确匹配，失败后用语义相似度映射到动作类别。
- Region Caption 和 Video Summary：完整文本直接交给官方 LLM 评分。

本地脚本不重复解析 Prediction，也不修改官方计分公式和
`data/MedVidBench/MedVidBench-Leaderboard` 中的官方代码。

## 续跑与提交

使用原目录续跑，已成功完成的题目会跳过：

```bash
python benchmark/medvidbench/run.py \
  --agent videospy \
  --run-dir output/videospy/medvidbench/<run_dir_name>
```

续跑参数必须与 `run.json` 一致。leaderboard_test 续跑还需传入同一个
`--sample-manifest experiments/data_splits/main/medvidbench/leaderboard_test.json`。全量完成后，
将 `submission.json` 上传到
[MedVidBench Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)。
