# LVBench

从仓库根目录运行：

```bash
python benchmark/lvbench/run.py --mode debug
python benchmark/lvbench/run.py --mode lite
python benchmark/lvbench/run.py --mode full
```

默认运行原始 VideoSeek。使用 `--agent` 切换到优化版 VideoSpy：

```bash
python benchmark/lvbench/run.py --agent videospy --mode debug
```

先按`--sample-manifest`选出样本并保留清单顺序，再按模式取子集。
步长在 [config/benchmark_sampling.yml](../../config/benchmark_sampling.yml) 的 lvbench 项配置，训练和评测共用，默认：

- `debug`：`samples[::100]`。
- `lite`：`samples[::3]`，约1/3。
- `full`：全部选定样本。

未传manifest时，默认使用`experiments/data_splits/main/lvbench/test.json`。
原test315题对应debug 4题、lite 105题，三组消融的样本一致。
切片不保证任务类别均衡。实际UID写入`run.json`，续跑必须保留原模式和manifest。

```bash
python -m benchmark.lvbench.run --agent videospy --mode debug \
  --sample-manifest experiments/data_splits/main/lvbench/test.json
```

默认数据位置：

- 元数据：`data/LVBench/zai-org-LVBench/video_info.meta.jsonl`
- 视频：`data/LVBench/AIWinter-LVBench/all_videos`

可以使用 `--metadata-path` 和 `--video-dir` 覆盖。VideoSeek 使用
`config/videoseek/`，VideoSpy 使用 `config/videospy/`。模型参数也可以通过
`--model-name`、`--api-base`、`--api-key`、`--max-steps` 等参数覆盖。对于
VideoSpy，这些参数只覆盖 Agent 决策模型，三个视觉工具的模型仍由配置文件控制。

VideoSpy 的固定 Agent、工具和 System Prompt 配置来自
`config/videospy/default.yml`，默认不包含人工初始 Skill。`--config` 用于加载相对
默认配置的差异，可以重复使用，后面的 YAML 覆盖前面的 YAML。

进化训练输出一个版本化 SQLite。推理时使用 `--evolution-db`，默认加载 final
版本，也可以用 `--evolution-version` 指定明确版本或 `latest`：

```bash
python benchmark/lvbench/run.py \
  --agent videospy \
  --evolution-db output/evolving/lvbench/<run_id>/evolution.sqlite3 \
  --mode full
```

加载顺序为默认 YAML、显式 YAML、SQLite、显式 CLI。SQLite 只覆盖
`TASK_SKILLS`、Memory 和白名单执行参数，CLI 优先级最高。VideoSeek 的配置行为
保持不变。

每次新运行会写入 `output/<agent>/lvbench/`（例如 `output/videoseek/lvbench/`）。运行目录命名为 `{mode}_{model_name}_{time}`，如 `debug_openai_qwen3.6-flash_20260816_220736`。主要结果包括：

- `predictions.json`：官方 UID 到选项字母的映射
- `metrics.json`：总体准确率和分类准确率
- `report.md`：按任务类别汇总 Agent、模型、准确率、轮数、token 和耗时
- `records.jsonl`：逐题回答、状态、Agent 轮数、token 消耗和耗时
- `run.log`：运行配置、逐题状态、Agent 详细输出和异常堆栈
- `trajectories/`：逐题 Agent 轨迹

终端会通过 `tqdm` 持续显示题目级进度。断点续跑时 `run.log` 会追加记录，不会
覆盖上一次运行日志。
如需同时查看并保存 Agent 的逐步日志，请显式传入 `--verbose`。
`metrics.json` 的 `run_statistics` 会汇总总轮数、token 和耗时，并给出每题均值。
`report.md` 会在每次运行或续跑结束后重新生成，便于直接查看和比较实验结果。

## 断点续跑

每道题完成后都会立即写入 `records.jsonl`。断网或手动停止进程后，使用原来的
运行目录继续执行：

```bash
python benchmark/lvbench/run.py \
  --mode full \
  --run-dir output/videoseek/lvbench/<run_dir_name>
```

VideoSeek 的 `<run_dir_name>` 形如 `full_<model>_<timestamp>`。VideoSpy 会在末尾追加运行时 Git Commit ID 的前 7 位，形如 `full_<model>_<timestamp>_<commit7>`。目录名可从上次运行的终端输出获取。

恢复时会跳过已成功完成且包含完整统计信息的题目，重新执行失败题目以及中断时
尚未写入结果的题目。`run.log` 会继续追加，最终的 `predictions.json` 和
`metrics.json` 会根据完整记录重新生成。

续跑必须保持原来的 Agent、运行模式、样本列表、数据路径和模型配置。若这些配置
与目录中的 `run.json` 不一致，程序会拒绝续跑，避免把 VideoSeek 和 VideoSpy 或
其他不同实验的结果混在一起。旧的 `run.json` 没有 Agent 字段时按 VideoSeek
兼容处理。

LVBench 数据目录中没有字幕文件，评测默认按无字幕模式执行。

LVBench 的 `--workers` 默认 2，只接受 1 或 2，该限制仅作用于单次评测。
多个评测进程可共享工作区并行运行，各自使用独立的运行目录。Memory 开启时，`memory_retrieval.jsonl`
记录问题哈希、命中 ID、相似度、空命中和案例正文长度，不记录 API Key。
离线训练脚本末尾串行运行三组消融，保存到本次Evolution目录的`evaluations/`。
独立评测入口为`experiments/test/lvbench/test_W_testsplit_evolve.sh`，
默认启用Skill+Memory，可用CONFIG指定Skill Only或Memory Only配置。
