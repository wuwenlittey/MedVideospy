# VideoSpy 进化实验

本目录保存固定数据划分、训练配置和按数据集拆分的运行脚本。所有脚本都应从
仓库根目录执行，在脚本顶部配置路径。三个数据集的离线脚本均在训练成功后串行评测 Skill Only、Memory Only、Skill+Memory，
也可单独运行评测入口。不再使用自动候选调度。

## 参数配置

所有Bash脚本的顶部配置均采用 `变量="${环境变量:-默认值}"`。
环境变量未设置或为空时使用原默认值，无需编辑脚本即可覆盖参数：

```bash
MODE=lite WORKERS=2 OUTPUT_DIR=output/manual/lvbench \
  bash experiments/test/lvbench/test_W_testsplit.sh

INFERENCE_RUN_DIR=/path/to/completed/benchmark/run \
CONFIG=experiments/train/train_configs/self_full.yml \
MODE=lite OUTPUT_DIR=output/manual/evolution \
  bash experiments/train/lvbench/train_offline.sh
```

- `MANIFEST`：当前训练或独立推理入口的样本清单。
- `TEST_MANIFEST`：各数据集离线训练结束后的三组评测清单，默认原test split。
  离线脚本的`MANIFEST`只影响训练，覆盖评测清单请使用`TEST_MANIFEST`。
- `CONFIG`、`INFERENCE_RUN_DIR`、`OUTPUT_DIR`、`RUN_DIR`：覆盖脚本中对应配置和路径。
- `EVOLUTION_RUN_DIR`、`EVOLUTION_DB`：独立Evolve推理入口的进化目录和数据库。
  数据库默认取进化目录内的`evolution.sqlite3`。
- `MODE`：训练与测试采样模式，默认full。MedVIDBench隐藏leaderboard仍只允许full。
- `WORKERS`：显式设置优先。未设置时LVBench为`2 × NUM_GPUS`（默认2），
  EgoLife为`NUM_GPUS`（默认1），MedVIDBench为`4 × NUM_GPUS`（默认4）。

部署脚本的`MODEL_NAME`、`MODEL_PATH`、`SERVER_PORT`、`TP`同样支持覆盖，
`aicloud_submit.sh`还支持`TASK_SCRIPT`。GPU检测、进程ID和临时文件等运行状态不作为配置参数。
各脚本只使用自己顶部声明的参数；三个数据集的三组评测始终保存到本次训练的实际Evolve目录。

## 目录结构

```text
experiments/
├── data_splits/
│   └── main/                # 当前训练和测试清单，按模式动态采样
├── prepare_medvidbench_split.py
├── test/
│   ├── test_configs/
│   ├── lvbench/
│   ├── egolife/
│   └── medvidbench/
└── train/
    ├── train_configs/
    ├── lvbench/
    ├── egolife/
    └── medvidbench/
```

Test 数据集目录的基础入口如下。LVBench只保留前三个入口：

```text
test_W_testsplit.sh           基础 VideoSpy 运行 test split
test_W_trainsplit.sh          基础 VideoSpy 运行 train split
test_W_testsplit_evolve.sh    加载进化数据库运行 test split
test_VideoSeek_testsplit.sh   原始 VideoSeek 运行 test split
```

MedVidBench 还包含两个隐藏测试集提交脚本：

```text
test_W_leaderboard_testsplit.sh           基础 VideoSpy 运行 leaderboard_test
test_W_leaderboard_testsplit_evolve.sh    加载进化数据库运行 leaderboard_test
```

三个 Train 数据集目录均包含：

```text
train_offline.sh              从历史推理轨迹独立蒸馏 Skill 与 Memory
```

## 流式评测

`run_all_streaming_task_types.sh` 不参与进化训练。它从仓库根目录启动，对 StreamEQA 三个 split，以及 StreamingBench 的 real、Omni、SQA，每个任务类型取 2 题。坏题由 `benchmark/streameqa/run.py` 和 `benchmark/streamingbench/run.py` 在加载时跳过。

```bash
STREAMING_BENCHMARKS_ROOT=/path/to/streaming_video_benchmark_package \
  bash experiments/run_all_streaming_task_types.sh
```

未设置 `STREAMING_BENCHMARKS_ROOT` 时，使用学校共享盘上的数据包路径。学校环境会激活 `llm_deploy`，并要求本机 `23333` 端口已有模型服务。

## 基础推理

直接运行对应数据集脚本：

```bash
bash experiments/test/lvbench/test_W_testsplit.sh
bash experiments/test/egolife/test_W_testsplit.sh
bash experiments/test/medvidbench/test_W_testsplit.sh
```

EgoLife和MedVIDBench仍保留原始 VideoSeek 评测入口：

```bash
bash experiments/test/egolife/test_VideoSeek_testsplit.sh
bash experiments/test/medvidbench/test_VideoSeek_testsplit.sh
```

生成离线进化所需的 train split 轨迹：

```bash
bash experiments/test/lvbench/test_W_trainsplit.sh
```

各数据集的 `test_W_*split.sh` 使用默认 VideoSpy 配置，不加载进化数据库。
`train_offline.sh` 中 Base 还会加载训练时指定的 `CONFIG`。`test_VideoSeek_testsplit.sh` 保留原始 VideoSeek 对照。
基础入口的 `OUTPUT_DIR` 是新运行的父目录；`RUN_DIR` 可指定新目录或恢复已有目录。
恢复训练后的 Base 评测时，将 `RUN_DIR` 设为 `<evolution_run_dir>/evaluations/base/<mode>/`。

命令结束时会打印实际的 `Run directory`。该目录是对应离线训练脚本的
`inference_run_dir`。

## MedVidBench Leaderboard

基础 VideoSpy 直接运行隐藏的完整测试集：

```bash
bash experiments/test/medvidbench/test_W_leaderboard_testsplit.sh
```

使用进化结果时，先修改 `test_W_leaderboard_testsplit_evolve.sh` 顶部的
`evolution_run_dir`，再运行：

```bash
bash experiments/test/medvidbench/test_W_leaderboard_testsplit_evolve.sh
```

`leaderboard_test` 没有本地 Ground Truth，脚本不会执行本地评测。运行完整成功后，
将 Run 目录中的 `submission.json` 上传到 MedVidBench Leaderboard。

MedVidBench 统一由 `--sample-manifest` 选择数据，不再使用 `--split` 或 `--dataset`。
`train.json`、`test.json` 是公开数据中的样本索引；`leaderboard_test.json`
在文件内声明隐藏数据源及全量样本。两个 leaderboard 脚本也支持 `MANIFEST` 环境变量。

## 手动离线训练

编辑 `train/lvbench/train_offline.sh` 顶部的 `manifest`、`inference_run_dir`、
`config`、`output_dir` 和 `run_dir`，然后从仓库根目录执行：

```bash
bash experiments/train/lvbench/train_offline.sh
```

三个训练脚本默认写入 `output/experiments/train/<benchmark>/<mode>/<run_id>/`。
`OUTPUT_DIR` 指定运行的直接父目录，Python 只追加运行名称，不再重复追加 benchmark。
`RUN_DIR` 仍指定完整运行目录，已有目录可按原路径续跑，评测放在该目录的 `evaluations/` 下。

Offline 必须指定已有 Base 轨迹的 `inference_run_dir`，不重新
运行训练集推理。训练成功后，脚本末尾依次运行Skill Only、
Memory Only、Skill+Memory和Base。底层Evolving CLI只负责训练，已移除自动测试逻辑及
`--test-after-training`、`--test-manifest` 参数。通过 `--run-dir-file` 将本次成功运行的
实际目录交给Shell，不扫描历史目录或解析日志。默认使用原 `main/lvbench/train.json`，
不再使用历史 Reflection/Dev 的额外划分作为默认配置。

V19 默认 `self_full.yml` 使用 `max_samples=0`，不额外限制 Skill 入选数量。该参数不限制 Classification 和 Memory。
每条轨迹使用完整可读对话，综合答案、证据、逻辑与效率评分，再按任务类别归一化。
`evolving.classification_threshold=0.2` 控制 Good/Bad 阈值，Others 不参与 Skill 分组，但仍提取 Memory。
标准差为零时该类别全部为 Others。默认 `group_size=4`、`bad_ratio=0.5`、seed=42，
同任务内分组，某侧不足用另一侧剩余样本补齐，不重复采样。

每类任务只有一个 Task Skill，依次结合各组轨迹更新，再归纳一个 Domain Skill。
Memory 独立覆盖模式采样后的全部有效轨迹，包括 Others 和分类失败项，每条最多一条事实，附来源视频、事件时间、可用时间、不确定性和原文引用。
入库前与已有及本批记忆去重，默认语义阈值0.95，只在同来源、数值时间重叠时比较语义。时间不可比较时只按相同事件和事实文本精确去重。
最终 `skills.json`、`memories.json` 直接展示已入库内容，SQLite用于推理加载。
Stop Optimizer 默认关闭，同时跳过 Policy Replay 和新停止规则的注入，Skill 与 Memory 的其他流程保持运行。
需要学习提前停止规则时，在训练 CONFIG 中显式开启：

```yaml
evolving:
  stop_optimizer:
    enabled: true
```

开启后，Policy Replay 使用 Agent 模型验证停止策略。关闭时 `selection.json` 中的 `stop_policy.reason` 为 `disabled`。
该开关只控制训练，不屏蔽旧数据库中已有停止规则的推理执行。要获得不含此类规则的成果，使用默认关闭配置启动新的训练运行。

所有模型输出使用 JSON，格式错误纠正一次，仍失败警告并跳过。没有额外 findings 或模型审核阶段。

`WORKERS` 控制训练中的 Futures 并发，也控制后续四轮评测各自的并发。
同类别 Skill 更新保持串行，其他任务和单轨迹 Memory 可以并行。
Agent开启Skill时加载Task和Domain两层，Memory以当前Question检索Top K。EgoLife 先排除来源 query_time 晚于当前问题或来源不可比较的记忆，再取 Top K。
完整数据格式、边界和验证方案见 [V19 规划](../docs/devs/trainoff_v19_plan.md)。

## 评测与三组消融

训练固定执行 Skill 和 Memory 两条通路，CONFIG 用于设置推理模型、反思模型、分组等参数。三个数据集的离线脚本均运行三组消融和 Base 四轮评测，先加载训练 CONFIG，再由三组消融各自的配置覆盖组件开关。Base 也加载训练 CONFIG，但不加载进化数据库。MedVidBench 使用本地 test split 并执行 `--evaluate`。已有进化结果可通过统一入口单独评测，默认Skill+Memory：

```bash
export EVOLUTION_RUN_DIR=/path/to/completed/evolution/run
bash experiments/test/lvbench/test_W_testsplit_evolve.sh
# 单独运行或恢复Skill Only
CONFIG=experiments/test/test_configs/skill_only.yml bash experiments/test/lvbench/test_W_testsplit_evolve.sh
# 单独运行或恢复Memory Only
CONFIG=experiments/test/test_configs/memory_only.yml bash experiments/test/lvbench/test_W_testsplit_evolve.sh
```

按顺序执行，不要加 `&` 并行。分别加载 `test/test_configs/` 下的
`skill_only.yml`、`memory_only.yml`、`skill_memory.yml`，三者都关闭超参数加载。
脚本要求Evolution目录存在
`evolution.sqlite3` 和 `final_version.json`，默认加载完成的final版本。

默认评测原 `main/lvbench/test.json` 的315题。训练脚本沿用数据库默认 final，结果分别写入原来的固定目录：

```text
<evolution_run_dir>/evaluations/<skill_only|memory_only|skill_memory|base>/<full|lite|debug>/
```

三个数据集的独立 Evolve 评测脚本均默认使用 `skill_memory.yml`，同时加载 Skill 和
Memory，关闭超参数加载，组件开关与训练后的 Skill+Memory 评测一致，但不会自动继承训练 CONFIG 中的模型等参数。默认使用
`evaluations/<config-name>/<mode>/`，重复执行可恢复同一评测。
隐藏 leaderboard 的 Evolve 结果单独写入 `evaluations/leaderboard_test/<config-name>/<mode>/`。
Evolve 入口的 `OUTPUT_DIR` 可覆盖上述 `evaluations` 父目录，`RUN_DIR` 可直接指定
新建或恢复的完整评测目录，优先于 `OUTPUT_DIR`。数据库所在目录必须有 `final_version.json`。
训练成果改变后，需要使用新的评测目录。训练脚本不自动按版本建立目录，重新进行整套训练与评测时使用新的训练运行目录。

每组自动生成 `report.md`、`metrics.json`、逐题 `records.jsonl` 和轨迹。
报告含总体及类别准确率、平均轮数、token和时间。不再调用历史固定门槛比较器。
Memory检索保存在 `memory_retrieval.jsonl`，分数变高不代替命中与稳定性检查。

### Mode：从同一 manifest 按步长抽取

```bash
# 同时控制离线训练和训练后的评测采样
MODE=debug bash experiments/train/lvbench/train_offline.sh
# 或单独评测已有进化结果
MODE=lite bash experiments/test/lvbench/test_W_testsplit_evolve.sh
```

各 benchmark 先选择 manifest 内的样本，再按模式切片。默认模式为 full。
训练和评测共用 [config/benchmark_sampling.yml](../config/benchmark_sampling.yml)，无需修改脚本。
每个 benchmark 的 full、lite、debug 数值为取样步长 N，表示 `[::N]`，必须是正整数。
例如将 `lvbench.debug` 改为 20，就从对应训练或测试清单中约取 1/20。
M 条样本实际取 `ceil(M / N)` 条。以下为默认配置：

| 模式 | LVBench / EgoLife | MedVidBench |
|---|---|---|
| full | 全部 | 全部 |
| lite | `[::3]`，约 1/3 | `[::10]`，约 1/10 |
| debug | `[::100]`，约 1/100 | `[::100]`，约 1/100 |

各 benchmark CLI 未指定 `--sample-manifest` 时，默认加载
`experiments/data_splits/main/<dataset>/test.json`。
MedVidBench 的隐藏 leaderboard_test 由 `main/medvidbench/leaderboard_test.json`
选择自己的完整集合，不套用公开测试集的索引或采样步长。
LVBench 默认 315 题对应 debug 4 题、lite 105 题；
MedVidBench 默认 1124 题对应 debug 12 题、lite 113 题。
切片保留清单顺序，不保证类别均衡，不修改原数据或清单。

独立评测入口可用 `MANIFEST` 指定评测清单；离线训练脚本使用 `MANIFEST`
指定训练清单、`TEST_MANIFEST` 指定评测清单。模式分别作用于对应清单。
训练数据内部的开发清单不能作为未见测试集，评测轨迹不应进入 Reflector。

Shell 脚本使用 `MODE` 环境变量，Python CLI 使用 `--mode`。
直接验证 Base 的同一子集可执行：

```bash
python -m benchmark.lvbench.run --agent videospy --mode debug --sample-manifest experiments/data_splits/main/lvbench/test.json --workers 2 --output-dir output/experiments/test/lvbench/base/test_debug
```

## 续跑与资源

沿用原来的 `bash experiments/train/<dataset>/train_offline.sh` 即可串联训练和四轮评测。
`RUN_DIR` 留空时自动创建目录，指向已有 V19 运行时续跑。
输入未变的成功模型输出从检查点复用，
失败项重新尝试，分类补齐后重算受影响的分组及 Skill。允许修复原来源目录中的失败轨迹或缺失答案，Memory 不随分组重复提取。
训练内部保存成果快照，Shell 直接使用数据库默认 final，不再添加版本参数或按版本建立目录。成果变化时，固定评测目录会触发已有的配置一致性检查，应使用新的训练运行目录重新执行。训练配置或清单变化也需使用新目录。底层 CLI 的 `--reuse-reflection-dir` 可在新运行中
复用模型、输入和实现版本匹配的阶段结果。V18反思和旧数据库不迁移。
单项模型或embedding失败记录warning并跳过，结果摘要分别统计各阶段成功、失败与覆盖率。任一实际执行阶段全部失败时标记 failed，部分失败标记 partial，合法 null 或全 Others 属于正常无产出。

训练成功后脚本依次执行四轮评测。某轮评测失败仍运行剩余评测，最后返回非零状态。
训练 failed 时不会开始评测，partial 允许评测并保留失败明细。已完成训练可通过独立Evolve评测入口继续评测。
三个数据集的训练脚本在非Linux环境保留已有代理配置并为校园端点追加NO_PROXY。
各benchmark使用独立运行目录。

## 其他数据集与历史内容

EgoLife、MedVIDBench的脚本和数据划分工具保留，共用进化算法主体。
LVBench和MedVIDBench统一使用按模式切片，EgoLife及各数据集的评分规则没有改动。
External-Reflector仍可使用 `train/train_configs/external.yml` 及已有的
`EXTERNAL_REFLECTOR_MODEL`、`EXTERNAL_REFLECTOR_API_BASE`、`EXTERNAL_REFLECTOR_API_KEY`。

当前保留 `data_splits/main/` 中三个数据集的 train/test 清单，以及 MedVidBench
的 leaderboard 清单。旧的静态 debug
清单、Reflection/Dev 划分和不再读取的聚合 baseline 已删除。debug/lite 均从当前
主清单按 `config/benchmark_sampling.yml` 动态采样。MedVIDBench的划分工具仍可手动运行：

```bash
python experiments/prepare_medvidbench_split.py
```

实验脚本与划分工具只读取 `data/`，不会修改其中的第三方内容。
