# LVBench

Run commands from the package root after completing the main README setup.
Obtain the LVBench videos and `video_info.meta.jsonl` metadata separately.
Create a JSON manifest containing the question UIDs to evaluate, such as
`["question_uid_1", "question_uid_2"]`, using actual UIDs from the metadata.

```bash
python -m benchmark.lvbench.run \
  --agent videospy \
  --mode full \
  --metadata-path /path/to/video_info.meta.jsonl \
  --video-dir /path/to/all_videos \
  --sample-manifest /path/to/lvbench_ids.json \
  --output-dir ./output
```

Use `--mode debug` or `--mode lite` for the subsets defined in
`config/benchmark_sampling.yml`. Pass the manifest explicitly because the
repository's default split files are not included. Evaluation uses no subtitles.
`--workers` accepts 1 or 2 for this adapter.

## Evolved inference and configuration

Add `--evolution-db /path/to/evolution.sqlite3` to load evolved experience.
Use `--evolution-version` to select a specific version. YAML overrides may be
supplied with repeated `--config` arguments. Configuration precedence is defaults,
YAML overrides, evolution state, then explicit CLI arguments.

The model environment variables in the main README configure both the agent and
visual tools. CLI options such as `--model-name` override only the agent model.

## Outputs and resuming

Each run writes a directory under `output/videospy/lvbench/` containing:

- `predictions.json`: question UID to answer-letter mapping.
- `metrics.json`: overall and per-type accuracy and execution statistics.
- `report.md`: performance and inference-cost summary.
- `records.jsonl`: per-question checkpoints and statistics.
- `run.json`, `run.log`, and `trajectories/`: configuration, logs, and execution records.

To resume, add `--run-dir /path/to/existing_run` to the same command. Keep the
original mode, manifest, dataset paths, and model settings. Completed samples
with valid statistics are skipped; failed or incomplete samples are retried.
Use `--verbose` to include detailed agent output in the logs.
