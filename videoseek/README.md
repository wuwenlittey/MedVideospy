# VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2603.20185)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

<div>
  <a href="https://jylin.me/">Jingyang Lin</a><sup>1,2</sup>,
  <a href="https://jialianwu.com/">Jialian Wu</a><sup>1</sup>,
  <a href="https://joellliu.github.io/">Jiang Liu</a><sup>1</sup>,
  <a href="https://cs-people.bu.edu/sunxm/">Ximeng Sun</a><sup>1</sup>,
  <a href="https://zewang95.github.io/">Ze Wang</a><sup>1</sup>,
  <a href="https://www.xiaodongyu.me/">Xiaodong Yu</a><sup>1</sup>,
  <a href="https://www.cs.rochester.edu/u/jluo/">Jiebo Luo</a><sup>2</sup>,
  <a href="https://zicliu.wixsite.com/mysite">Zicheng Liu</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=bX1YILcAAAAJ">Emad Barsoum</a><sup>1</sup>
  <br>
  <sup>1</sup>AMD&emsp;
  <sup>2</sup>University of Rochester&emsp;
</div>

This repository contains the original [VideoSeek](https://arxiv.org/abs/2603.20185)
agent and VideoSpy, an optimized implementation with a native function-calling
loop and message-native trajectories.

## Introduction

VideoSeek is a long-horizon video agent that leverages video logic flow to actively seek answer-critical evidence instead of exhaustively parsing the full video.
![alt text](../docs/assets/videoseek.png)

VideoSeek (w/ subtitles) achieves the best performance while processing only about 1/300 as many frames as the second-best video agent.

Toolkit of the VideoSeek agent, including `<overview>`, `<skim>`, and `<focus>` tools:
![alt text](../docs/assets/toolkit.png)

- `<overview>`: rapidly scans the entire video to build a coarse storyline and highlight promising intervals.
- `<skim>`: takes a quick glance at these candidate intervals at low cost to check whether query-relevant evidence is nearby.
- `<focus>`: zooms in on a fine-grained clip with dense inspection to obtain answer-critical observations.

## Installation

1. Create a conda virtual environment and activate it:

```bash
conda create -n videoseek python==3.13
conda activate videoseek
```

2. Clone the repository:

```bash
git clone https://github.com/jylins/videoseek
cd videoseek
```

3. Install the package:

```bash
pip install -e .
```

Notes:

- **`ffmpeg` is required**.

## Usage

### Agent configuration

VideoSeek and VideoSpy use independent configurations:

```bash
cp config/videoseek/general.example.yml config/videoseek/general.yml
```

VideoSeek keeps the original flat model configuration. VideoSpy separately
configures the Agent model and the models used by `overview`, `clip_skim`, and
`frame_inspect`. `clip_skim` browses a selected interval using uniformly sampled
frames, while `frame_inspect` checks one or more selected full-resolution frames.
Model-related CLI options override only the VideoSpy Agent model;
its tool models remain controlled by the selected YAML config.

VideoSpy keeps its fixed scaffold prompt and all defaults in `default.yml`.
It accepts repeated YAML overrides and an optional versioned evolution database:

```bash
videospy-cli \
  --video_path video.mp4 \
  --user_query "What happens?" \
  --config custom-general.yml \
  --evolution-db output/evolving/lvbench/<run_id>/evolution.sqlite3 \
  --evolution-version latest
```

YAML files define the base Agent. The evolution database stores versioned Skill,
Memory, and whitelisted execution parameters. SQLite values override YAML values,
while explicit options such as `--model-name` or `--max-steps` have the highest
priority. Fixed prompts, model endpoints, data paths, and evaluators cannot be
overridden by the database. VideoSeek configuration behavior is unchanged.

### CLI

Please execute `videoseek-cli -h` for help:
```
usage: videoseek [-h] --video_path VIDEO_PATH --user_query USER_QUERY [--subtitle_path SUBTITLE_PATH] [--output_dir OUTPUT_DIR] [--verbose] [--model_name MODEL_NAME] [--api_base API_BASE] [--api_key API_KEY] [--api_version API_VERSION]
                 [--reasoning_effort REASONING_EFFORT] [--seed SEED] [--temperature TEMPERATURE] [--max_tokens MAX_TOKENS] [--max_steps MAX_STEPS]

options:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        YouTube URL or local path to video.
  --user_query USER_QUERY
                        Question/query towards this video.
  --subtitle_path SUBTITLE_PATH
                        Local path to subtitle file.
  --output_dir OUTPUT_DIR
                        Directory to write outputs (default: ./output/).
  --verbose             Print agent step logs.
  --model_name MODEL_NAME
                        Model name.
  --api_base API_BASE   API base.
  --api_key API_KEY     API key.
  --api_version API_VERSION
                        API version.
  --reasoning_effort REASONING_EFFORT
                        Reasoning effort of the LLM.
  --seed SEED           Seed.
  --temperature TEMPERATURE
                        Temperature.
  --max_tokens MAX_TOKENS
                        Max output tokens of the LLM.
  --max_steps MAX_STEPS
                        Max steps of the VideoSeek agent.
```

Run with a local video:

```bash
# Example from LVBench (qid: 3094)
videoseek-cli \
    --video_path ./wgBlACG927Y.mp4 \
    --subtitle_path ./wgBlACG927Y.srt \
    --user_query "What animal statue is under the Dong Men Ding Food Street sign?\n(A) Hawk\n(B) Panda\n(C) Tiger\n(D) Lion\nPlease directly answer with the best option's letter from the given choices directly (A, B, C, or D)." \
    --verbose
```

Run the optimized VideoSpy Agent with the same input:

```bash
videospy-cli \
    --video_path ./wgBlACG927Y.mp4 \
    --subtitle_path ./wgBlACG927Y.srt \
    --user_query "What animal statue is under the sign?" \
    --verbose
```

VideoSpy performs reasoning and tool selection in one native function-calling
request per round. Its `trajectory.json` stores the complete chat message
sequence, including assistant tool calls and matching tool observations. When a
tool calls a model, its `role: tool` message also contains a `model_call` object:

```json
{
  "model_call": {
    "input": {"model": "...", "messages": []},
    "output": {"role": "assistant", "reasoning_content": "...", "content": "..."}
  }
}
```

Multimodal inputs are preserved as sent, including base64 image data, so these
trajectory files can be substantially larger than VideoSeek trajectories.

Outputs are written under `output/videoseek/demo/<VIDEO_ID>_<timestamp>/`:

- `prediction.json`
- `trajectory.json`

### Benchmarks

LVBench and MedVidBench runners can evaluate either Agent through
`--agent videoseek` or `--agent videospy`. VideoSeek remains the default. See
[`benchmark/lvbench/README.md`](../benchmark/lvbench/README.md) and
[`benchmark/medvidbench/README.md`](../benchmark/medvidbench/README.md) for dataset
paths, full/lite/debug modes, outputs, and resume instructions. MedVidBench uses
the local `test` split by default and `--split leaderboard_test` for full
Leaderboard submissions.

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{lin2026videoseek,
  title={VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking},
  author={Lin, Jingyang and Wu, Jialian and Liu, Jiang and Sun, Ximeng and Wang, Ze and Yu, Xiaodong and Luo, Jiebo and Liu, Zicheng and Barsoum, Emad},
  journal={arXiv preprint arXiv:2603.20185},
  year={2026}
}
```
