# VideoSpy Code

VideoSpy answers questions about videos through iterative visual inspection.
It uses three tools: `overview` for global context, `clip_skim` for a selected
interval, and `frame_inspect` for details at specific timestamps. The package
also includes the Reflector, Agent Optimizer, and Skill/Memory storage code.

## Installation

Use Python 3.10 or newer. Linux x86_64 is recommended for the `decord` video
decoder. Other platforms require a compatible `decord` installation.
Run these commands from the extracted `videospy-code/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Configure the model

Provide an OpenAI-compatible endpoint serving a vision-language model that
supports image inputs and function calling. The agent and visual tools use
the same model by default. Set these variables to your own deployment:

```bash
export VIDEOSPY_API_BASE="http://localhost:8000/v1"
export VIDEOSPY_API_KEY="your-api-key"
export VIDEOSPY_MODEL="openai/Qwen3.5-4B"
```

The `openai/` prefix selects LiteLLM's OpenAI-compatible provider. Replace the
model name after that prefix with the identifier exposed by your endpoint.
The package does not download or serve model weights.

`config/videospy/default.yml` contains the prompts, tool settings, and model
parameters. Connection fields in the exported configuration refer to the
environment variables above. An optional `--config local.yml` can override
these settings. Environment variables should be set before launching Python.

## Run inference

Use a local video file readable by `decord`:

```bash
python -m videospy.cli \
  --video_path /path/to/video.mp4 \
  --user_query "What happens in the video?" \
  --output_dir ./output \
  --verbose
```

Subtitles are optional: add `--subtitle_path /path/to/video.srt`.
The agent gathers evidence for up to 16 rounds and then makes a separate
model call to produce the final answer. Each run creates a directory under
`output/videospy/demo/` containing:

- `prediction.json`: the final answer.
- `trajectory.json`: messages, tool calls, observations, and model-call records.

No evolution database is required for this foundational-agent example.

## Optional clip follow-up experiments

High-resolution re-inspection and temporal gap sampling are independent,
default-off options. Add `--config config/videospy/refinement_detail.yml`,
`--config config/videospy/refinement_temporal.yml`, or
`--config config/videospy/refinement_both.yml` to the existing command.
Use separate output directories. To disable both, apply
`--config config/videospy/refinement_off.yml` last.
See [experiment instructions](../docs/devs/clip_refinement_ablation.md) for frame
selection, cost accounting, audit records, controls, and rollback.

## Core code and evolved experience

- `videospy/agent.py`, `model.py`, and `tools/`: the inference loop and visual tools.
- `videospy/evolving/reflector/`: task-grouped reflection and experience extraction.
- `videospy/evolving/optimizer/`: Skill review and Memory preparation.
- `videospy/evolving/storage/` and `evolved_config.py`: versioned storage and loading.
- `config/videospy/`: configuration loading and exported defaults.
- `benchmark/sampling.py` and `config/benchmark_sampling.yml`: the shared sampling
  helper required by the evolution entry point.

To load an existing compatible evolution database, add
`--evolution-db /path/to/evolution.sqlite3` to the inference command.
Memory retrieval also requires `VIDEOSPY_EMBEDDING_API_BASE`,
`VIDEOSPY_EMBEDDING_API_KEY`, and `VIDEOSPY_EMBEDDING_MODEL`; the embedding model
and dimensions must match those used to create the database. Skill-only
inference can use `--evolved-skill --no-evolved-memory --no-evolved-hyperparameter`.
Task-specific Skill loading additionally requires a matching task type in
`_runtime.task_type` in an override YAML; the standalone CLI defaults to `general`.

## Medical visual case memory

Optional, independently switchable visual-case retrieval is described in
[the setup and ablation guide](../docs/devs/visual_case_memory.md). It connects to
an existing WeMM embedding service and supports text or image/text references
inside the existing local observation call. Trajectory exports remain pending
review until their image/label correspondence has been verified.
