# Benchmark Evaluation

First install the dependencies and configure the model endpoint as described in
the [main README](../README.md). Run commands from the package root.

Each adapter requires separately obtained dataset files and an explicit sample
manifest. Dataset-specific formats, commands, and outputs are documented here:

- [LVBench](lvbench/README.md)
- [EgoLifeQA](egolife/README.md)
- [MedVidBench](medvidbench/README.md)

Always select `--agent videospy`; other agents are not included in this package.
Sampling strides are defined in
[`config/benchmark_sampling.yml`](../config/benchmark_sampling.yml). The adapter
first selects the manifest entries in their listed order, then applies the
`full`, `lite`, or `debug` stride. This subsampling does not guarantee balanced
task categories. MedVidBench leaderboard evaluation uses the full selected set.
