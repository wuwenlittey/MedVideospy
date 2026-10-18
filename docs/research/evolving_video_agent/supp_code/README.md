# VideoSpy: Supplementary Code

VideoSpy answers video questions through iterative visual inspection with three
tools: `overview`, `clip_skim`, and `frame_inspect`. This package includes the
inference agent, task-grouped reflection, evidence-based review, Skill and Memory
storage, and evaluation adapters for LVBench, EgoLifeQA, and MedVidBench.

## Setup

Use Python 3.10 or newer. Linux x86_64 is recommended for the `decord` video
decoder; other platforms require a compatible installation. Run all commands
from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Provide an OpenAI-compatible endpoint with image-input and function-calling
support. The agent and visual tools use the same model by default:

```bash
export VIDEOSPY_API_BASE="http://localhost:8000/v1"
export VIDEOSPY_API_KEY="your-api-key"
export VIDEOSPY_MODEL="openai/Qwen3.5-4B"
```

Replace these example values with your deployment. The `openai/` prefix selects
LiteLLM's OpenAI-compatible provider; the remaining name must match the model
served by the endpoint. The package does not download or serve model weights.
Keep credentials in environment variables rather than in the source files.

`config/videospy/default.yml` contains the prompts, tool settings, and model
parameters. Its connection fields resolve the environment variables above.
Use `--config /path/to/local.yml` for additional YAML overrides.

## Run inference

```bash
python -m videospy.cli \
  --video_path /path/to/video.mp4 \
  --user_query "What happens in the video?" \
  --output_dir ./output \
  --verbose
```

Subtitles are optional: add `--subtitle_path /path/to/video.srt`. VideoSpy gathers
evidence for up to 16 rounds, followed by a separate final-answer call. Results
are written under `output/videospy/demo/`:

- `prediction.json`: the final answer.
- `trajectory.json`: messages, tool calls, observations, and model-call records.

This foundational-agent example does not require an evolution database.

## Evolved experience

Add `--evolution-db /path/to/evolution.sqlite3` to load a compatible stored state.
The final version is selected by default; `--evolution-version` selects a specific
version. For Skill-only inference, also pass
`--evolved-skill --no-evolved-memory --no-evolved-hyperparameter`.

Memory retrieval requires `VIDEOSPY_EMBEDDING_API_BASE`,
`VIDEOSPY_EMBEDDING_API_KEY`, and `VIDEOSPY_EMBEDDING_MODEL`. The embedding model
and dimensions must match those used to create the database. Benchmark adapters
supply task types automatically. For the standalone CLI, set `_runtime.task_type`
in an override YAML to select task-specific Skill; its default is `general`.

Offline evolution consumes previously collected benchmark execution records:

```bash
python -m videospy.evolving.cli run \
  --benchmark lvbench \
  --train-manifest /path/to/train_ids.json \
  --inference-run-dir /path/to/completed_training_run \
  --output-dir ./output/evolution
```

The input run must contain `run.json`, `records.jsonl`, and the referenced
trajectories with outcome references. Collect training experience separately
from evaluation data. Datasets, split manifests, collected records, and trained
evolution databases are not included in this source package.

## Code layout

- `videospy/agent.py`, `model.py`, and `tools/`: inference and visual inspection.
- `videospy/evolving/reflector/`: reflection and experience extraction.
- `videospy/evolving/optimizer/`: Skill review and Memory preparation.
- `videospy/evolving/storage/` and `evolved_config.py`: storage and context loading.
- `config/videospy/`: agent configuration.
- `benchmark/`: dataset adapters and evaluation utilities.

See the [benchmark guide](benchmark/BENCHMARK.md) for dataset-specific commands.
