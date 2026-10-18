# VideoSpy

VideoSpy 是基于原生函数调用实现的长视频理解 Agent。每轮模型负责判断是否继续收集视觉证据，工具结果以标准消息形式保存在完整轨迹中，最终答案由独立的无工具模型调用统一生成。

## 工具

- `overview`：对完整视频均匀抽帧，建立全局认识。
- `clip_skim(query, start_time, end_time)`：粗粒度浏览指定视频片段并定位候选时刻。
- `frame_inspect(query, timestamps)`：检查一个或多个指定时间点附近的原分辨率视频帧。

没有字幕和视觉证据时，Agent 不允许直接回答。

加载进化 Memory 后，Agent 会在任务开始时以当前 Question 自动检索最相关的 Top K 条历史记忆，默认 3 条，并注入初始 Prompt。Memory 保存有工具观察依据的身份、关系、事件、时间和原文引用，不作为 Agent 工具或占用工具调用步骤。EgoLife 先按来源身份和来源 query_time 排除未来或无法比较的记忆，再取 Top K。其他 benchmark 允许跨视频参考，Agent 需区分历史事实与当前场景。Skill 与之独立，提供对应任务策略和一个 Domain Skill。

## 配置

配置文件位于 `config/videospy/`：

- `default.yml`：纳入版本管理的完整默认配置，包括固定 Prompt、可进化项默认值和进化开关。
- 本地模型或密钥配置不会自动加载，需要通过 `--config` 显式传入。

`tools.enabled` 是本次运行允许调用的视觉工具清单。例如仅启用全局浏览：

```yaml
tools:
  enabled: [overview]
```

VideoSpy CLI 和三个 benchmark 入口支持 `--enabled-tools`，优先于配置文件：

```bash
python -m benchmark.lvbench.run --agent videospy --mode debug --enabled-tools overview frame_inspect
```

不传该参数时沿用配置，`--enabled-tools overview` 只启用全局浏览，单独传 `--enabled-tools` 则禁用全部视觉工具。工具名只能是 `overview`、`clip_skim`、`frame_inspect`。

该清单同时控制发给模型的工具定义和实际执行权限。模型即使返回了禁用工具的调用，Agent 也会拒绝执行，并告知当前可用工具。多个 YAML 中的 `enabled` 列表按配置顺序替换，不做合并。`enabled: []` 禁用全部视觉工具，没有可用字幕时会因缺少证据而报错。

工具能力和使用限制统一写在 `videospy/tools/` 中的工具 `description` 与参数 schema，抽帧数量和时间戳数量上限按当前 Agent 配置填入。System Prompt 只保留通用的证据判断、调用约束和回答规则。历史 Skill 或 Memory 中出现的工具名不会改变执行权限，自动记忆检索仍由 `tools.memory_search.enabled` 独立控制。

运行时可以依次加载多个 YAML，并叠加训练生成的版本化 SQLite：

```bash
videospy-cli \
  --video_path video.mp4 \
  --user_query "What happens in the video?" \
  --config local.yml \
  --evolution-db output/evolving/lvbench/<run_id>/evolution.sqlite3 \
  --verbose
```

配置优先级为默认 YAML、显式 YAML、SQLite、显式 CLI。SQLite 默认加载 final
版本，也可以通过 `--evolution-version` 指定明确版本或 `latest`。任务 Skill、Memory 和
白名单执行超参数保存在 SQLite，不会修改固定 System Prompt。

Offline 从已有的完整 Benchmark 运行中蒸馏经验：

```bash
python -m videospy.evolving.cli run \
  --benchmark medvidbench \
  --train-manifest med_train.json \
  --inference-run-dir output/benchmark/medvidbench/<run_id> \
  --output-dir output/evolving/medvidbench
```

V20 离线训练先对每条轨迹的答案、观察依据、推理过程与效率综合评分（0–1），
再按任务类别计算 z-score。默认低于 -0.2 为 Bad，高于 0.2 为 Good，其余为 Others。
Others 不参与 Skill 分组，但仍提取 Memory。同类任务全部同分时归为 Others，分类失败单独记录。
配置 `evolving.classification_threshold` 可以调整阈值。

同任务内按 `reflector.sampling.group_size` 和 `bad_ratio` 配组，默认4条、Bad占比0.5。
`max_samples` 只限制 Skill 入选数量，默认 0 表示不限。每类任务维持一个 Task Skill，
按组顺序积累条件规则的有效支持与反例，最终筛选最多 5 条实践，再归纳最多 3 条跨任务 Domain 规则。
成功支持要求答案正确且证据充分；工作状态最多 10 条，依据不足或存在反例的规则不发布。
推理按任务加载，Task 与 Domain 总注入不超过 3,000 字符，统计和原始依据留在产物中。
Memory 独立覆盖模式采样后的全部有效轨迹，包括 Others 和分类失败轨迹，每条最多一条事实，保存来源视频、
事件时间、来源可用时间、不确定性和逐字引用。同一来源且数值时间重叠的重复事实在入库前去除，阈值为
`evolving.memory_dedup_threshold`，默认0.95。不同视频的相似事件分别保留。
引用由模型选择原文片段 ID 后程序回填。推理先过滤来源与可用时间，再按问题相似度取 Top K，
Memory 注入不超过 1,600 字符。LVBench 只复用同视频，MedVidBench 只复用同帧序列，
EgoLife 还要求同主体、来源不晚于当前问题且事件坐标可比较。无适用来源时不调用 Query Embedding。

`--workers` 控制分类、不同任务的 Skill 更新和 Memory 提取并发。
控制台通过 tqdm 显示分类、Memory 提取、Task Skill 分组更新、Domain Skill 和 Memory 去重进度。
反思进度按实际完成更新，包括失败和缓存复用，并显示 failed、reused 数量。
三个 `train_offline.sh` 把 `WORKERS` 同时传给训练和后续评测。单项模型失败警告并跳过，
Skill 更新失败保留上一版。分阶段记录成功、失败和覆盖率，部分完成允许评测，任一实际执行阶段全部失败时停止自动评测并保留其他成果。
配置错误、无有效输入或数据库不可写等整体错误会报错。反思输入不静默裁剪，上下文超限记为失败。
不再使用 findings、独立模型审核或 Memory 上下文重写阶段。

根目录的 `skills.json` 和 `memories.json` 与实际入库结果一致。
`reflection/classification.json` 保留原始评分、理由、任务均值和标准差、z-score及分组标签，
`reflection/selection.json` 保留分组，`reflection/skills/tasks/<task-name>.json` 保存每个任务的最终 Skill，
`reflection/skills/domain.json` 保存最终 Domain Skill。Skill 逐组检查点放在 `reflection/checkpoints/skills/`，
Memory 检查点仍放在 `reflection/memories/`。
`--run-dir` 可以续跑 V19 运行，复用输入未变的成功检查点并补齐失败项。分类补齐后重算受影响的分组与 Skill，Memory 不随分组重复提取。
成果变化时新增 SQLite 版本和 `versions/<snapshot>/` JSON，保留旧版本。`offline_result.json` 记录本次状态及明确的评测版本。
`--reuse-reflection-dir` 可在新运行中复用身份匹配的检查点。
旧版反思协议和数据库不迁移，请使用新运行目录。

推理开启 Skill 时加载当前 Task Skill 和一次 Domain Skill。每题开始时以 Question
作为 Query 自动检索 Memory，保留相似度达到阈值的 Top K（默认3条），并展示来源视频、
事件时间、来源可用时间和事件。EgoLife 的可用性由程序先行过滤，事实适用性仍需根据当前视频判断。
检索失败警告后继续当前视频推理。训练不更新模型权重或推理超参数。

脚本训练完成后依次运行 Skill Only、Memory Only、Skill+Memory、Base 四轮评测。
训练固定生成 Skill 和 Memory，CONFIG 只传给训练，用于反思模型、分组等参数。四轮评测使用默认推理配置，再由各消融配置控制组件开关，Base 不加载进化数据库。脚本沿用数据库默认 final，输出位于原来的 `evaluations/<component>/<mode>/`，不处理版本参数或版本目录。成果改变后，已有评测目录的配置一致性检查会拒绝混用，应使用新的训练运行目录重新执行。
训练 CLI 只负责训练，用 `--run-dir-file` 将允许评测的结果目录交给 Shell。
`MODE=full|lite|debug` 默认 full，取样步长在 [config/benchmark_sampling.yml](../config/benchmark_sampling.yml) 配置，训练和评测共用。
默认 lite 在 LVBench、EgoLife 使用 `[::3]`，MedVidBench 使用 `[::10]`，debug 使用 `[::100]`。
详细方案及首轮验证记录见 [V20 规划](../docs/devs/trainoff_V20_plan.md)。
新版 SQLite schema 为 6，旧训练产物不做转换，请使用新的训练运行目录。

运行根目录下的 `evolution.log` 记录阶段进度、单项耗时和失败警告。

进化代码按职责组织：`evolving/common/` 负责训练数据读取、来源规则和共享数据结构，
`evolving/reflector/` 负责模型反思、引用校验、调度与检查点，
`evolving/optimizer/skill_optimizer.py` 负责 Task Skill 逐组迭代、Domain Skill 归纳及无更新或失败时保留已有内容，
`evolving/optimizer/memory_optimizer.py` 负责事实去重和向量化，`pipeline.py` 串联各阶段，`evolving/storage/` 负责 JSON、SQLite 存储和检索。根目录只保留入口、流程编排和包初始化文件。
完整目录说明见 [V19 代码组织](../docs/devs/trainoff_v19_plan.md#5-代码组织)。

自动运行目录为
`<output-dir>/<reflector>_<model>_<timestamp>_<commit7>`，不再额外追加 benchmark。
三个 train_offline.sh 默认写入 `output/experiments/train/<benchmark>/<mode>/<run_id>/`。
单独调用 CLI 且未指定 `--output-dir` 时，父目录默认为 `output/evolving/<benchmark>`。

字幕是可选参数：

```bash
--subtitle_path video.srt
```

## 输出

结果写入 `output/videospy/demo/`：

- `prediction.json`：最终答案。
- `trajectory.json`：完整消息、工具调用、工具观察及工具内部模型调用。

运行 benchmark 请参阅 [LVBench](../benchmark/lvbench/README.md)、
[EgoLifeQA](../benchmark/egolife/README.md) 和
[MedVidBench](../benchmark/medvidbench/README.md)。完整实验见
[experiments](../experiments/README.md)。
