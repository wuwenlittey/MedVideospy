# EgoLifeQA

This adapter supports the public A1_JAKE subset of 500 multiple-choice questions.
It maps the DAY1-7 video segments to a continuous timeline without rewriting the
videos. Each question can access only footage available before its `query_time`.
A1_JAKE is the public dataset identifier required by the adapter.

Run commands from the package root after completing the main README setup.
Obtain the question annotations, video segments, and official transcripts
separately. Supply a JSON list of question IDs with `--sample-manifest`:

```bash
python -m benchmark.egolife.run \
  --agent videospy \
  --mode full \
  --qa-path /path/to/EgoLifeQA_A1_JAKE.json \
  --video-root /path/to/A1_JAKE \
  --transcript-root /path/to/Transcript/A1_JAKE \
  --sample-manifest /path/to/egolife_ids.json \
  --output-dir ./output
```

Pass the manifest explicitly because the default split files are not included.
`debug` and `lite` sampling strides are defined in `config/benchmark_sampling.yml`.

Official transcripts are provided to visual tools only for the inspected time
intervals. The full week's transcript is not placed in the initial prompt.
Dense captions and annotation fields such as `target_time`, `reason`, `trigger`,
and `keywords` are not provided to the agent.

## Evolved inference and outputs

Add `--evolution-db /path/to/evolution.sqlite3` to use stored Skill and Memory.
Each run writes to `output/videospy/egolife/`, including `predictions.json`,
`metrics.json`, `report.md`, `records.jsonl`, `run.json`, `run.log`, and
`trajectories/`. Metrics include overall accuracy, per-type accuracy, macro
average, and execution costs.

Resume with `--run-dir /path/to/existing_run` while keeping all original data,
manifest, mode, and model settings. Successfully completed questions are skipped
and aggregate outputs are regenerated from the collected records.
