# MedVidBench

The adapter uses the benchmark's pre-extracted frame sequences. A sample manifest
selects one of two dataset variants:

- `test`: public data with outcome references, supporting local evaluation.
- `leaderboard_test`: hidden-reference data, evaluated through the official leaderboard.

Obtain the corresponding metadata and frames separately. Run all commands from
the package root after completing the main README setup.

## Local inference

Create a manifest with zero-based indices into the public metadata JSON:

```json
{"dataset": "test", "sample_indices": [0, 1, 2]}
```

A plain JSON list of indices is also accepted. The indices above are an example;
use the sample list for your intended evaluation split.

```bash
python -m benchmark.medvidbench.run \
  --agent videospy \
  --mode full \
  --metadata-path /path/to/public_metadata.json \
  --frame-root /path/to/valdata \
  --sample-manifest /path/to/medvidbench_ids.json \
  --output-dir ./output
```

Pass the manifest explicitly because the default split files are not included.
Use `debug` or `lite` for the subsets defined in
`config/benchmark_sampling.yml`. Add `--evolution-db /path/to/evolution.sqlite3`
to load learned Skill and Memory. Task-specific Skill selection and output
formats are handled by the adapter.

## Local evaluation

Obtain the official MedVidBench leaderboard evaluator and install its own
requirements separately. Evaluate a completed public-data run with:

```bash
python -m benchmark.medvidbench.evaluate \
  --run-dir /path/to/completed_run \
  --ground-truth-path /path/to/public_metadata.json \
  --leaderboard-dir /path/to/MedVidBench-Leaderboard \
  --skip-llm-judge
```

`--skip-llm-judge` computes the seven deterministic metrics and skips the three
LLM-based captioning and summarization metrics. To include those metrics, set a
judge endpoint and omit `--skip-llm-judge`:

```bash
export MEDVIDBENCH_JUDGE_API_BASE="http://localhost:8000/v1"
export MEDVIDBENCH_JUDGE_API_KEY="your-api-key"
export MEDVIDBENCH_JUDGE_MODEL="Qwen3.5-4B"
```

Judge settings are independent of the agent settings. The judge uses the OpenAI
SDK directly, so its model name must match the endpoint's model identifier
without LiteLLM's `openai/` prefix. Other parameters are in
`benchmark/medvidbench/config/evaluation.yml`. Credentials are read from the
environment and are not written to the evaluation report.

The report includes CVS, next-action prediction, skill assessment,
spatiotemporal grounding, temporal grounding, dense captioning, video
summarization, and region captioning metrics. The diagnostic `NAP_exact_acc`
uses exact label matching and is reported separately from official `NAP_acc`.
The adapter preserves the official evaluator's scoring rules.

## Leaderboard inference

Use this manifest to select the full hidden-reference dataset:

```json
{"dataset": "leaderboard_test", "sample_indices": "all"}
```

```bash
python -m benchmark.medvidbench.run \
  --agent videospy \
  --mode full \
  --metadata-path /path/to/leaderboard_metadata.json \
  --frame-root /path/to/testdata \
  --sample-manifest /path/to/leaderboard_manifest.json \
  --output-dir ./output
```

This variant supports only `full` mode and cannot be scored locally. Submit the
resulting `submission.json` to the official leaderboard.

## Outputs and resuming

Runs are stored under `output/videospy/medvidbench/`:

- `run.json`: configuration and selected samples.
- `records.jsonl`: per-sample predictions, checkpoints, and costs.
- `submission.json`: benchmark IDs, task types, and predictions.
- `evaluation.json`: local metrics, when evaluation is performed.
- `report.md`, log files, and `trajectories/`: summaries and execution records.

To resume, add `--run-dir /path/to/existing_run` to the original inference
command, retaining the same manifest, dataset, mode, and model settings.
