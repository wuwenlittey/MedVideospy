# StreamEQA

VideoSpy 只读取提问时刻及之前的视频。官方标注和 HD-EPIC 视频都不进本仓库。

`STREAMING_BENCHMARKS_ROOT` 指向数据包根目录，也就是含有 `official_repositories/` 和 `datasets/` 的那一层。默认是共享盘上的 `streaming_video_benchmark_package`。

```bash
python -m benchmark.streameqa.run --agent videospy --mode smoke --split Perception
```

- QA 默认：`official_repositories/StreamEQA/data/annotations/Perception.json`
- 视频默认：`datasets/hd_epic/videos/`
- `Interaction` 有 12 条官方缺陷题：9 条只有 3 个选项，3 条 `realtime` 为负。不要改官方 JSON。`benchmark/streameqa/run.py` 的 `load_dataset` 会在任何模式里跳过它们，不依赖单独的 smoke 脚本。
- `Planning_all` 里 `realtime` 为空的题会被跳过，这是上游就没有查询时刻，不是下载损坏。

`smoke` 每个 task 2 题，`lite` 每类 10 题，`full` 为全部可加载题目。
