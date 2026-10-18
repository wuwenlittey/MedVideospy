# StreamingBench

目前只接 Real-Time Visual Understanding 的四选一题，并且只服务 VideoSpy。每道题只能看到 `time_stamp` 及之前的画面。官方仓库和视频都不进本仓库。

```bash
python -m benchmark.streamingbench.run --agent videospy --mode smoke
```

`STREAMING_BENCHMARKS_ROOT` 指向数据包根目录。默认路径：

- QA：`official_repositories/StreamingBench/src/data/questions_real.json`
- 视频：`datasets/streamingbench/videos/<文件名>.mp4`

`questions_real.json` 不引用 `sample_316_real.mp4`，这是官方题表和视频包本身不齐，视频保留，runner 不会去加载它。同一加载器也可以跑 `questions_omni.json` 和 `questions_sqa.json`；SQA 的分组列表会先展平。`questions_proactive.json` 不是四选一，加载时跳过。一次覆盖这些任务类型用 `experiments/run_all_streaming_task_types.sh`。

`smoke` 每个 task type 2 题，`lite` 每类 10 题，`full` 为该 JSON 里的全部题目。
