# EgoLifeQA

该 adapter 只支持公开的 A1_JAKE 500 道多项选择题，并只服务 VideoSpy。它把
DAY1–7 的分段 MP4 映射为惰性的连续录像时间线，跳过夜间和录像间隙，不会拼接
或改写原始视频。每道题只能访问其 `query_time` 之前的视频，避免未来信息泄漏。

```bash
python -m benchmark.egolife.run \
  --agent videospy \
  --mode debug
```

默认数据路径如下：

未指定`--sample-manifest`时，默认使用`experiments/data_splits/main/egolife/test.json`。
先按该清单选择问题，再按 [config/benchmark_sampling.yml](../../config/benchmark_sampling.yml) 的 egolife 项取样，训练和评测共用。
默认取`debug: samples[::100]`、`lite: samples[::3]`，full保留全部。
可显式传入其他manifest覆盖默认值；自定义QA数据也应提供对应清单。

- QA：`data/EgoLife/lmms-lab_EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json`
- 视频：`data/EgoLife/lmms-lab_EgoLife/A1_JAKE`
- Transcript：`data/EgoLife/lmms-lab_EgoLife/EgoLifeCap/Transcript/A1_JAKE`

可以用 `--qa-path`、`--video-root` 和 `--transcript-root` 覆盖这些路径。官方
Transcript 默认开启，但整周内容不会进入初始问题，只在 Agent 检查对应时间区间
时提供给视频工具。评测不使用 DenseCaption，也不向 Agent 提供 `target_time`、
`reason`、`trigger` 或 `keywords`。

加载进化版本：

```bash
python -m benchmark.egolife.run \
  --agent videospy \
  --mode full \
  --sample-manifest output/experiments/manifests/egolife/val.json \
  --evolution-db output/evolving/egolife/<run-id>/evolution.sqlite3
```

每次新运行会写入 `output/videospy/egolife/`。输出格式与 LVBench 保持一致：

- `predictions.json`：官方问题 ID 到选项字母的映射
- `metrics.json`：总体 Accuracy、五类 Accuracy、Macro Average 和运行统计
- `report.md`：按任务类别汇总模型、完成数、准确率、轮数、token 和耗时
- `records.jsonl`：逐题回答、状态、轮数、token 消耗和耗时
- `run.log`：运行配置、逐题状态和异常堆栈
- `trajectories/`：逐题 Agent 轨迹

终端通过 `tqdm` 显示进度。使用原来的 `--run-dir` 可以断点续跑，已成功完成且
包含统计信息的问题会被跳过，`report.md`、`predictions.json` 和 `metrics.json`
会根据完整记录重新生成。标注中的 target time 或 evidence time 不会提供给 Agent。
