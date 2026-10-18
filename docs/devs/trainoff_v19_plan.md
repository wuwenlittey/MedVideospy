# V19 离线进化方案

本正文统一已确认的规则。训练从已有 Base VideoSpy 轨迹中生成 Skill 和 Memory，不重新运行训练集推理，不更新模型权重或推理超参数。

实现遵循最小改动，优先保持代码干净，不为旧版协议增加兼容或迁移层。继续使用原来的三个入口，准备好脚本顶部配置、离线源目录和模型服务后即可开始训练及评测：

```bash
bash experiments/train/lvbench/train_offline.sh
bash experiments/train/egolife/train_offline.sh
bash experiments/train/medvidbench/train_offline.sh
```

保留原来的 MODE、CONFIG、WORKERS、INFERENCE_RUN_DIR 等变量及脚本结构。脚本自动接续 Skill Only、Memory Only、Skill+Memory、Base 四轮评测，使用原来的固定评测目录，不读取成果版本、不传入版本参数、不按版本建立目录。RUN_DIR 留空时自动创建训练目录，指定已有运行时续跑。CONFIG 只传给训练，用于反思模型、分组等训练参数。评测使用默认推理配置和各自的消融配置。

训练脚本的默认目录简化为 `output/experiments/train/<benchmark>/<mode>/<run_id>/`，移除 offline、self_<mode> 和重复 benchmark 层。OUTPUT_DIR 表示运行的直接父目录，Python 只追加 run_id；self/external 信息保留在运行名称中。已有输出不搬迁，RUN_DIR 继续接受原完整路径。

## 1. 核心流程

先按 full/lite/debug 确定共同训练来源，再进入两条独立通路。训练始终生成 Skill 和 Memory，不受 evolving.inference 组件开关影响。这些开关只控制推理时使用哪些成果：

训练和评测的取样步长统一读取 `config/benchmark_sampling.yml`，按 benchmark 和模式配置正整数 N，在各自的 manifest 上执行 `[::N]`。默认 full=1，LVBench/EgoLife lite=3，MedVidBench lite=10，debug=100。实际训练步长写入 run_config.yml 并参与配置一致性检查，改变步长后使用新运行目录。三个 train_offline.sh 无需新增参数。

```text
轨迹 → 综合评分 → 按任务归一化 → Good/Bad 分组 → 迭代 Task Skill → Domain Skill
轨迹 → 独立提取事实 Memory → 去重（含 Embedding）→ SQLite 入库
```

| 名称 | 产生阶段 | 含义与使用位置 |
|---|---|---|
| 可读轨迹 | 读取来源 | 保留全部可读对话，用于 Classification 和分组 Skill 反思 |
| evidence | 读取来源 | 原始工具观察及调用标识的内存结构，用于 Memory 提取和引用校验，不单独调用模型 |
| Classification | 每条轨迹评分、按任务统计 | 原始 score、reason，以及各任务的均值、总体标准差、z-score 和标签 |
| selection | 选择与分组 | 入选 Skill 的样本及各任务分组，不限制 Memory 来源 |
| Task Skill | 每任务逐组更新 | 可跨样本复用的任务决策策略，每任务最多一个 |
| Domain Skill | Task Skill 完成后 | 适用于当前 benchmark 各类任务的领域策略，保存于 __domain__ |
| Memory | 每条轨迹独立提取 | 有原始工具观察支持的事实，包含来源、事件、时间、不确定性与引用，每轨迹最多一条 |

不生成 findings，不增加候选审核、分块 findings 综合、反复总结或独立模型语义审核。

### Classification

- 每条轨迹只评分一次，包括多标签轨迹。输入包含初始字幕、全部工具调用与观察、Agent 输出及实际记录的推理文本，保持原始顺序。
- 移除图片编码、重复的模型请求记录和非内容型服务元信息，不再以约 10,000 字符摘要替代完整轨迹。
- 综合答案正确性、证据支持、逻辑连贯性和执行效率，返回 `score ∈ [0,1]` 与 `reason`。步骤少只有在证据充分时才是优点。
- 参考答案单独提供，仅用于评分。缺少参考答案时分类失败，Memory 继续。评分不重新观看视频，也不假定能审查未记录的推理。
- 每任务分别计算 `z=(score-mean)/std`，使用总体标准差。多标签轨迹在各任务内分别归一化。
- `z < -threshold` 为 Bad，`z > threshold` 为 Good，其余为 Others，含边界相等。默认 `evolving.classification_threshold=0.2`。
- 失败分数不参与统计，记录为 failed。零方差归为 Others。Good/Bad 表示相对质量，不保证答案绝对正确。

### Skill

- 仅使用 Good/Bad，按同任务、`group_size` 和 `bad_ratio` 分组。同任务内不重复抽样，一侧不足时用另一侧补齐。
- 默认 `group_size=4`、`bad_ratio=0.5`、seed=42。每组目标 Bad 数按 `round(group_size * bad_ratio)` 计算，尾组允许不足。
- `max_samples` 仅限制 Skill 入选的唯一轨迹数量，默认 0 表示不限，不限制 Classification 和 Memory。
- 每组输入轨迹、原始评分、z-score、理由和当前 Skill，输出更新后的完整 Task Skill，允许返回 null。
- 同任务各组顺序更新，不同任务并行。没有入选分组、无有效更新或某组失败时，保留此前有效 Skill。
- 最后根据各 Task Skill 归纳一个 Domain Skill，失败或无更新时保留已有 Domain Skill。
- 不宣称逐组重写必然提高质量。程序不增加独立的候选审核层。

### Memory

- 面向共同训练来源内全部有效轨迹，包括 Others 和分类失败轨迹，与 Skill 分组无关。
- 每轨迹最多一条有用事实，允许 null。读取完整的成功工具观察，失败工具输出、Agent 猜测和参考答案不作为事实依据。
- 保存 `context`、`content`、`event`、`uncertainty`、`time`、`evidence_refs`。time 包含 start_seconds、end_seconds、label，未知时间使用 null/空文本。
- 数值时间使用来源视频或拼接时间线的秒数，不能与日历时间混用。引用为一至三段原文，每段最多 800 字符，校验引用标识和原文存在性。
- 时间提示按来源 benchmark 选择，只有 EgoLife 使用 DAY/clock 说明。label 只填写观察中实际出现的时间或片段，不复制提示词及字段说明，未知时留空。
- 程序检查必需字段、有限非负时间、区间顺序和引用。引用存在不等于整条事实已核实，不增加模型语义核验。
- 来源 benchmark、video、source_id、sample_id、trajectory、query_time、available_at 由程序绑定。模型不能修改来源或可用时间。
- MedVidBench 的秒数相对于当前输入帧序列起点。来源身份绑定数据集、原视频、完整帧序列和 FPS，另存首尾帧、帧数和时间基准供查看。相同原视频的不同帧序列或 FPS 不共用时间坐标，不猜测原视频偏移。优先读取记录中的 memory_source，否则从 run.json 指定的 metadata_path 恢复。无法恢复时警告并限定为当前轨迹来源，避免跨未知片段误删。

## 2. Memory 去重与推理使用

### 去重

- 同一来源且数值时间区间重叠时，先比较规范化事实文本，再按向量相似度去重，默认 `evolving.memory_dedup_threshold=0.95`。规范化使用大小写折叠与空白合并。
- 区间端点相同视为重叠。时间展示 label 不影响数值比较。不同来源或不相交区间不做语义去重。
- 时间不可比较时，仅删除同来源、同规范化事件且事实文本完全相同的重复项。
- EgoLife 按来源轨迹 query_time 从早到晚处理，优先保留更早可用的记忆，后来的来源不能抑制更早记忆。
- 与当前状态已有记忆及本批已接受记忆去重。本轮不新增跨训练数据库接续能力。
- 接受向量相似度可能误删不同或矛盾事实的限制，不增加额外模型判断。

### 推理

- 开启 Skill 时，每题开始加载对应 Task Skill 和一个 Domain Skill。
- 开启 Memory 时，直接使用当前 Question 作为 Query，先按来源与可用时间过滤，再按相似度取 Top K，默认 K=3。
- EgoLife 使用来源轨迹 query_time 作为保守的可用时间，仅允许不晚于当前问题的来源。事件时间与可用时间分别保存和展示，不从 Question 文本猜测截止时间。
- 当前 EgoLife 适配器只支持 A1_JAKE，通过来源路径中的明确身份规范化。来源身份或时间无法可靠比较时排除，不回退到无约束检索。
- 其他 benchmark 允许跨视频参考，但展示来源并提示模型区分历史事实与当前场景。相似问题不意味着相同事实。
- 删除手动 `memory_search` 工具的实现与注册，仅保留同名自动检索配置和每题自动检索路径。检索失败警告后继续当前推理。

## 3. 并发、失败与续跑

统一使用 Python Futures，复用 `WORKERS`。Classification 和 Memory 按轨迹并行，Task Skill 按任务并行，同任务内部顺序更新，Domain Skill 等 Task Skill 完成后执行。去重入库顺序确定，避免并发接纳重复项。

控制台用 tqdm 分别显示 Classification、Memory extraction、Task Skill（按组计数）、Domain Skill 和 Memory deduplication 进度。反思节点完成、失败或复用缓存时立即更新，不等待排在前面的慢任务。进度表示已处理数量，另列 failed 和 reused。日志通过 tqdm 输出以避免覆盖进度条，evolution.log 仍保存普通文本日志。

单项失败记录 warning、阶段、节点标识和原因，继续处理其他节点并保留成功成果。模型格式错误允许纠正一次。上下文超限明确失败，不静默截断，也不重复请求相同超长输入。

分阶段统计 Classification、Task Skill、Domain Skill、Memory 提取和 Embedding。统计包括 succeeded、failed、empty、skipped、reused 和 coverage。合法 null（包括 JSON 代码块中的 null）计为成功且 empty，coverage 为 succeeded / (succeeded + failed)，没有执行项时为 null。

| 完成状态 | 判断与后续行为 |
|---|---|
| complete | 没有失败。合法 null、全 Others、去重后无新增均可正常完成 |
| partial | 有失败，但没有任何已执行阶段全部失败。允许评测，展示失败明细与覆盖率 |
| failed | 任一已执行阶段有失败且无成功项。保留其他成果，停止自动评测 |

配置错误、没有有效来源或数据库不可写等无法继续的整体错误直接报错。无效来源项单独记录 source 失败，不冒充已执行的模型请求。不增加额外失败比例阈值。

续跑规则：

1. `RUN_DIR` 指向现有运行，成功且输入未变的检查点复用，失败或损坏项重新执行，不能因完成文件存在而直接跳过。
2. 分类补齐后重算相关统计和分组，Skill 检查点包含分组输入与此前 Skill，受影响的后续更新自动重算。Memory 不随分组重复提取。
3. 从初始状态与有效检查点重建成果，不能以上次 final 为起点再学一遍相同分组。另取最近保存快照中的 Skill 作为兜底，包括整体失败运行中的成功成果。任务没有分组、全部无更新或失败时保留该 Skill，Domain Skill 同理。兜底内容不作为分组迭代起点，不改变检查点输入，也不掩盖失败统计。
4. 训练内部继续保存成果快照，成果不变时复用。脚本直接读取数据库默认 final，不再处理版本。训练失败时 CLI 返回非零，停止后续评测。评测恢复固定目录后，成果改变会触发 benchmark 原有的配置一致性检查，应使用新的训练运行目录重新训练和评测，不自动另建版本目录或覆盖旧评测。
5. 同一运行允许修复原来源目录中的失败轨迹或缺失答案。benchmark、训练清单或训练配置变化时使用新目录，可通过 `--reuse-reflection-dir` 复用输入、模型和实现一致的成功检查点。

## 4. 产物与接口

保留 JSON 和 SQLite。SQLite schema 为 5，本轮反思协议为 19.1，历史反思结果不自动迁移。

```text
<run>/
  evolution.sqlite3
  evolution.log
  inference_source.json
  offline_result.json
  final_version.json
  skills.json
  memories.json
  reflection/
    classification.json
    classification/<sample-hash>.json
    selection.json
    skills/tasks/<task-name>.json        # 每个任务的最终 Skill
    skills/domain.json                  # 最终 Domain Skill
    checkpoints/skills/tasks/<task-name>/group_<index>.json
    checkpoints/skills/domain.json
    memories/<sample-hash>.json
    embeddings/<input-hash>.json
    report.json
  versions/<snapshot>/skills.json
  versions/<snapshot>/memories.json
  evaluations/<skill_only|memory_only|skill_memory|base>/<mode>/
```

根目录 JSON 表示最近一次构建的成果，状态以 offline_result.json 为准。版本目录保存对应不可变快照。memories.json 导出不含向量，向量保存在 SQLite。offline_result.json 包含状态、阶段统计、失败明细、覆盖率、成果数量和明确的评测版本。

reflection/skills 按可读任务名导出最终成果，每个文件为一个 Skill。逐组中间结果只放在 reflection/checkpoints/skills，用于续跑，不作为多个 Skill 注入推理。任务名中的路径分隔符等特殊字符使用百分号编码。没有 Good/Bad 分组且无已有 Skill 的任务不生成占位 Skill。

三个训练脚本的四轮评测均使用 config/videospy/default.yml 作为基础推理配置，不继承训练 CONFIG。Skill Only、Memory Only、Skill+Memory 分别加载对应消融配置控制组件开关，Base 不加载进化数据库或消融配置。进化组通过 `--evolution-db` 沿用默认 final 加载，不增加 Shell 版本读取或目录管理。某轮评测失败仍执行其余评测，最后返回非零状态。

默认 full 使用全部清单，lite 的 LVBench/EgoLife 使用 [::3]，MedVidBench 使用 [::10]，debug 使用 [::100]，可在 config/benchmark_sampling.yml 修改。模式同时影响训练和脚本后续评测，但分别作用于各自清单。

清理范围包括未使用函数、旧 uncertain 分支、旧 Skill 排序特例、废弃摘要裁剪、未使用的引用解析结果、重复 JSON 写入、手动 Memory Search，以及训练不再使用的 MedVidBench evaluation.json 和旧指标前置条件。删除 config/videospy/__init__.py 与 EvolutionPipeline 中针对旧 evolving.optimizer 配置的检测及报错，不增加兼容转换或替代检查。保留推理所需超参数快照。不修改第三方数据和历史输出，不扩展按时间重新训练 Skill。

## 5. 代码组织

目录按数据准备、反思、Skill 迭代与 Memory 入库前处理、存储划分。Reflector 负责模型反思，SkillOptimizer 和 MemoryOptimizer 分别负责两类成果的更新与整理。

```text
videospy/evolving/
  cli.py                      # 命令行入口
  pipeline.py                 # 串联各阶段，组装成果并保存
  common/
    training_data.py           # 清单、来源校验、记录和轨迹读取
    memory_scope.py            # 来源身份、片段时间坐标与记忆可用时间
    task_types.py              # 各数据集任务类别归一化
    schemas.py                 # 各阶段共享的数据结构和状态
  reflector/
    runner.py                 # 评分归一化、分组、并发反思与检查点
    model.py                  # Classification / Skill / Memory 提示词及模型请求
    evidence.py               # 完整可读轨迹、工具观察和引用校验
  optimizer/
    skill_optimizer.py        # Task Skill 逐组迭代、Domain Skill 归纳与已有内容保留
    memory_optimizer.py       # Memory 去重、Embedding，产出待入库条目
  storage/
    memory_store.py            # SQLite 存储、Embedding 缓存和检索
    artifacts.py               # JSON/YAML、内容哈希、日志及脱敏
```

- 原 offline_source.py 合并到 common/training_data.py，属于 Reflector 之前的数据准备，不是 Optimizer。
- 原 offline_evidence.py 移至 reflector/evidence.py，并接收完整可读轨迹的预处理逻辑。
- 原 offline_experience.py 移至 reflector/runner.py，类名为 ReflectionRunner，明确负责反思调度。
- 原 reflector/base.py 改名为 reflector/model.py，它包含具体模型请求逻辑，并非抽象基类。
- 原 batch_reflection.py 的轨迹读取、分组和哈希分别归入 common/training_data.py、reflector/runner.py 和 storage/artifacts.py，删除混合职责文件及重复实现。

SkillOptimizer 负责同任务各组顺序更新、维护完整 Task Skill、汇总 Task Skills 生成 Domain Skill，以及无更新或失败时保留已有内容。具体提示词、模型请求与输出校验仍在 reflector/model.py。ReflectionRunner 负责分类与分组、不同任务的并发调度，以及两条通路共用的检查点和阶段统计。SkillOptimizer 复用这些检查点，不增加反思阶段。Memory 提取结果的去重和向量化由 MemoryOptimizer 执行，SQLite 的实际写入由 pipeline 调用 memory_store 完成。

删除只有结果转交和组装逻辑的 AgentOptimizer，pipeline 直接组装 EvolutionUpdate。旧模块路径不保留兼容文件。本次模块调整会改变反思实现指纹及统一后的检查点哈希，旧检查点不直接复用，不改写历史输出。

## 6. 实现与验收记录

首轮实现已完成基础评分、任务归一化、双路反思和 JSON/SQLite 存储。当时模拟测试为 151 项测试、96 个子测试通过，另有一个已存在的部署脚本默认值断言失败。该历史结果不用于证明本轮修订。

本轮已落实上述统一流程、完整输入、Memory 时间约束、失败统计、检查点补齐、不可变成果版本及评测组件控制，并增加边界回归测试。验收覆盖：

- 完整评分输入、边界和零方差、多标签、分组比例与 max_samples 作用范围。
- 每任务一个 Skill，无分组、null、失败时保留已有内容。
- Others 和分类失败轨迹独立提取 Memory，引用与字段校验。
- 来源、重叠区间、未知时间、较早可用记忆的去重，以及 EgoLife 过滤后取 Top K。
- 五个阶段全部失败、正常无产出、部分完成、补跑后产生新成果和旧版本不变。
- 三个数据集真实字段结构的测试夹具，以及真实 Shell → 训练 CLI → JSON/SQLite → 四轮评测参数解析和配置加载。
- 模拟模型和视频评测用于验证程序路径，不等于真实模型训练已经跑通。MedVidBench 仍需提供有效的离线源目录。

前一轮全量验证记录：`python -m pytest -q` 为 168 项测试、104 个子测试通过，1 项原有部署脚本默认值断言失败（`test_deployment_header_defaults_and_overrides_without_launching_services`）。未修改该无关部署脚本。三个 train_offline.sh 均通过 `bash -n`，`git diff --check` 无错误。没有调用真实模型服务，也没有修改第三方数据或历史输出。MedVidBench 脚本默认离线源目录当前不存在，真实训练前需设置有效的 `INFERENCE_RUN_DIR`。

前次入口调整：恢复原脚本结构与固定评测目录，移除新增的 Shell 版本处理逻辑和两处旧 evolving.optimizer 检查。当时相关 35 项测试、14 个子测试通过，覆盖原入口、默认 final 加载、失败停止评测和旧评测配置一致性检查。此后取消评测继承训练 CONFIG，当前要求以正文为准。三个脚本语法检查与差异检查通过。LVBench/EgoLife 默认源目录存在，MedVidBench 仍需配置有效来源。测试使用模拟模型和视频评测。

目录整理验证：修改前相关 35 项测试、14 个子测试通过。整理后全量测试为 168 项测试、104 个子测试通过，仍仅有上述原有部署脚本断言失败。三个 train_offline.sh 在本次目录整理中未修改，模拟端到端测试覆盖其训练与评测入口。没有调用真实模型服务。

SkillOptimizer 拆分验证：Task Skill 逐组更新和 Domain Skill 归纳已移至 optimizer/skill_optimizer.py，训练流程与脚本入口保持不变。相关 35 项测试、14 个子测试通过，覆盖更新失败时保留已有 Skill、检查点续跑和三个数据集的模拟训练评测入口。本次未运行无关的部署脚本默认值测试，也未调用真实模型服务。

训练与评测配置分离验证：三个脚本已删除评测阶段新增的训练 CONFIG 传递。相关 35 项测试、14 个子测试通过，确认即使训练配置中的推理组件开关为 false，仍执行 Skill 和 Memory 两条通路，四轮评测使用默认推理参数并按各自消融配置加载成果。三个脚本语法检查与差异检查通过，测试未调用真实模型服务。

审查修复与目录收敛：修复 MedVidBench 不同帧序列误共用时间坐标、续跑无有效更新时丢失此前 Skill、JSON 代码块中的 null 被误判为失败三项问题。根目录仅保留 cli.py、pipeline.py 和 __init__.py，其余归入 common、reflector、optimizer、storage，不保留旧路径兼容文件。全量验证为 173 项测试、104 个子测试通过，仍有上述原有部署脚本断言失败。最终相关 44 项测试、14 个子测试通过，覆盖片段及 FPS 去重、来源缺失时隔离、整体失败快照中的 Skill 保留和重复续跑复用。三个 train_offline.sh 未修改，语法检查通过；未调用真实模型服务，未修改第三方数据或历史输出。
