# 已有正确标签驱动的图文经验库

普通作答轨迹没有现成图文对应标注时，使用新增的[文本大模型轨迹整理流程](visual_fact_trajectory_alignment.md)：在 inventory 与 assemble 之间运行 `align-trajectories`，让文本模型判断可用性并提出带引用的候选配对。该来源标记为 `trajectory_aligned`，需要独立的建库/在线准入开关；不能冒充本文的外部标注 `source_grounded`。下面保留无模型、已有明确标注的路径用于对照。

本版把离线工作改成“整理历史轨迹中已有正确描述和对应画面”。`inventory-trajectories` 直接读取轨迹保存的标准答案，不需要原数据集；`assemble` 不调用 9B 或其他 VLM，不重新看图生成标签，不把最终答对当作图文标注。在线使用 `visual_facts_supervised.yml`：WeMM 检索参考后，由原局部观察模型完成一次观察。旧在线 support/conflict review 代码、调用入口及两个配置已删除。

## 1. 来源和使用范围

支持两种输入：

- 原始轨迹及原视频/帧序列：重建所有成功 `clip_skim`、`frame_inspect` 实际观察的帧，取其与外部证据标注的对应部分。原题答对或答错都可使用。
- 已有正确图文对：直接导入图片路径、实际时间戳和正确描述及对应关系，无须先伪造轨迹。

正确文字允许来自 `trajectory_gold`（轨迹保存的评测标准答案）、`dataset_gold`（已有数据集正确选项）、`dataset_description`（已有数据集描述）或 `verified_description`（已有人工确认描述）。后者必须记录确认人。模型的 `final_answer`、解释、review note 都不会成为标签。文本原样保留，不扩写视觉细节。

每个标签必须提供明确的图文对应关系，其来源为数据集证据标注或人工确认。**仅有 QA 正确答案、一次覆盖较大的 clip 调用、或者“模型最终答对了”，均不足以自动确认所有画面的标签。** 缺少对应关系时导出待定项，可批量补充标注后重建；不调用模型猜标签。

数据集没有时间依据时，不能凭这个工具自动制造时间真值。当前没有指定服务器专有字段，因此采用下述明确格式；已有事件边界/关键帧标注可以批量转成该格式，无须逐条看图重标。时间统一为原视频的数字秒数，`01:30` 应转为 `90`，不能写成 `1.30`。来源必须按原视频/患者划分，测试视频不得入库。

## 2. 直接从整批轨迹读取标准答案（当前推荐入口）

只需要轨迹目录、明确的视频划分和媒体路径，不提供 `--dataset`：

```bash
python -m videospy.visual_memory.cli inventory-trajectories \
  --trajectories /runs/train_dev_trajectories \
  --splits /runs/video_splits.jsonl \
  --schema config/videospy/visual_fact_trajectory_schema.yml \
  --output output/visual_memory/inventory_v3
```

划分文件示例，所有值均需换成真实身份：

```json
{"source_video_id":"original-video-087","split":"train","procedure":"actual_procedure","video_path":"/datasets/videos/procedure/087.mp4","trajectory_video_aliases":["procedure/087.mp4"]}
```

默认识别轨迹中的 `correct_answer`、`gold_option`、`ground_truth`、`gt_answer`、`reference_answer` 等标准答案字段，以及 `options` / `choices`。嵌套字段用点分路径，可在 schema YAML 中映射：

```yaml
trajectory_inventory:
  gold_fields: [evaluation.reference_option]
  options_fields: [metadata.options]
  video_fields: [metadata.video_id]
  task_fields: [metadata.task_name]
```

字段名只声明来源，不证明该字段真是标准答案；请先核对一条服务器轨迹。默认不读取含义不明的 `answer`，也禁止把 `final_answer`、`prediction`、`correct` 等模型预测或对错标志映射为标准答案。多个标准答案字段互相冲突会跳过。只保存了标准答案全文也可使用；只保存一个字母却没有选项文字会跳过。

若 options 字段缺失，可从题干中连续、单行的 `A. ...` / `B. ...` 选项块提取选项文字。多行选项或复杂包装应映射显式 options 字段，避免截断。整数答案必须设置 `gold_format: index_0` 或 `index_1`，不会猜测编号从 0 还是 1 开始。

轨迹没有视频身份时，在对应视频划分行加 `sample_ids: ["283", "284"]`，明确 QA 属于哪个原视频；不凭文件名相似或答题对错推断。轨迹没有 QA ID 时使用文件名 stem，例如 `283.json` → `283`。同视频、患者或同一媒体不能跨集合；即使测试轨迹已经作答、保存了标准答案，也不能建入本轮测试的库。

`inventory.json` / `skipped.jsonl` 会列出缺标准答案、缺选项、视频无法对应等原因。`dataset_reads` 和 `vlm_calls` 均为 0。新建 sources 保留标准答案字段路径、轨迹哈希、划分哈希；组装时再对照原轨迹验证标准答案未被替换。

轨迹中若已有按下述规范保存的 `visual_fact_annotations` 或 `supervision` 数组，会原样复制。也可用 schema 映射已有标注字段，或提供单独 annotations 文件。工具调用的观察窗口和最终答对本身都不会自动成为正确时间标注。

如果需要兼容原先独立数据集标注的工作流，旧 `inventory --dataset ...` 命令仍可使用，其标签类型为 `dataset_gold`；当前推荐的轨迹入口不调用它。

## 3. 提供已有标签的证据对应关系

推荐单独准备 `annotations.jsonl`，每行对应一个已有标注，以 **source_video_id + source_sample_id** 精确连接 inventory 中的来源。同一 QA 可有多条不同 annotation_id；未知来源会列为待定，不按 QA ID 单独串接。

### 轨迹保存的正确选项＋已标注事件区间

```json
{"source_video_id":"original-video-087","source_sample_id":"283","annotation_id":"event-283","fact_type":"phase","label_source":{"kind":"trajectory_gold"},"alignment":{"kind":"interval","start":120.0,"end":135.0,"scope":"phase_interval","source":"dataset_annotation","reference":"saved-event-annotations/283"}}
```

`trajectory_gold` 的标签从已盘点轨迹的标准选项和文字解析，或使用轨迹保存的标准答案全文。自动记录轨迹哈希、QA ID 和标准答案字段路径；若另填 text，必须与原标准答案一致。不会使用轨迹的 `final_answer`。`alignment.source: dataset_annotation` 表示该时间依据是已有数据标注；它可以已经存进轨迹或单独标注文件，本命令不会因此打开原数据集。人工确认的时间依据使用 `human_verified` 并记录 verified_by。

`alignment` 声明的是**这个标签对应的真实局部事件/阶段区间**，不能填整题时间范围、搜索窗口或模型自行判断的时间。只保存这个区间内原工具实际采到的帧，不重新采样。区间两端包含；记录的是一组帧对应的片段标签，不是“每帧都完整展示动作”。动作/片段/阶段证据至少需两张观察帧。

### 已有人工确认描述＋指定关键帧

```json
{"source_video_id":"original-video-087","source_sample_id":"283","annotation_id":"verified-instrument-283","fact_type":"instrument_identity","label_source":{"kind":"verified_description","text":"A needle holder is visible.","reference":"curated-pairs-v1/record-24","verified_by":"annotator-id"},"alignment":{"kind":"timestamps","timestamps":[126.5],"scope":"frame_set","source":"human_verified","reference":"curated-pairs-v1/record-24/evidence","verified_by":"annotator-id"}}
```

示例英文标签仅演示格式，不是本项目真实医学标注。`dataset_description` 同样要求 text 和 reference，来源应是已经存在的数据集描述。填写“已确认”字段不会自动证明描述正确；代码验证来源格式和对应关系，不冒充人工医学核验。

指定时间点必须能对应轨迹实际观察的帧，允许不超过半个原视频帧周期的取整误差；必须找到全部指定帧，不能找到一部分便带着完整标签入库。超过每条 16 帧的上限会待定，不任意删帧。可以使用已有关键帧标注收窄证据集合。

支持 8 类：`instrument_identity`、`anatomy`、`action`、`visible_state`、`spatial_relation`、`phase`、`lesion_attribute`、`visible_text`。类型由已有标注明确指定。`action recognition` 任务可能问阶段，因此不按任务名自动把答案写成 action。`phase_interval` 只生成 phase；不能从阶段名称推导器械、解剖或动作事实。

作用域：`frame_set` 是已标注的局部帧集合；`local_clip` 是动作等局部片段；`phase_interval` 是阶段区间。全片统计、首次出现、组织学和未标定物理尺寸等答案不能整条塞入本地视觉事实。若其中已有单独标注的部位、器械等可见事实，可用其独立描述和对应证据。

## 4. 直接导入现成图文对

无需轨迹时，sources.jsonl 可以直接包含：

```json
{"source_video_id":"original-video-012","source_sample_id":"curated-pair-12","task_type":"instrument identity","procedure":"actual_procedure","split":"train","question":"Which instrument is visible?","image_paths":["images/pair-12.png"],"timestamps":[42.5],"supervision":[{"annotation_id":"pair-12","fact_type":"instrument_identity","label_source":{"kind":"verified_description","text":"A needle holder is visible.","reference":"curated-pairs-v1/12","verified_by":"annotator-id"},"alignment":{"kind":"timestamps","timestamps":[42.5],"scope":"frame_set","source":"human_verified","reference":"curated-pairs-v1/12/evidence","verified_by":"annotator-id"}}]}
```

图片相对 sources.jsonl 定位，复制到新输出目录，保留内容哈希。不提供 trajectory_path/video_path/frame_paths。这里的 image_paths 是已配对关键帧；旧 frame_paths 是带 FPS 的整段视频帧序列，两者用途不同。

## 5. 离线组装与索引

```bash
python -m videospy.visual_memory.cli assemble \
  --sources output/visual_memory/inventory_v3/sources.jsonl \
  --annotations /runs/annotations.jsonl \
  --config config/videospy/visual_fact_assembly.yml \
  --output output/visual_memory/supervised_facts_v3
```

sources 已内嵌 supervision 时省略 `--annotations`。输出：

- `cases.jsonl`：原标签＋对应图片＋来源/标注/工具采样记录。
- `pending.jsonl`：缺标签、缺对应关系、未观察到指定帧、超帧预算及无匹配来源等原因。缺标注时保留正确选项候选和已观察时间，便于批量整理。
- `report.json`：任务/事实类型/视频分布、缺失类型、来源标签类别、原轨迹对错分布、去重及配额排除记录，VLM 调用数固定为 0。
- `images/`、`checkpoints/`、`run.json`：图片、逐事实记录及恢复指纹。

支持同一条原命令加 `--resume`。恢复会检查来源、标注、配置、媒体状态及已导出图片；变化时拒绝混合，应换输出目录。组装本身不含付费模型调用，恢复时会重新核对采样与图片。异常输入不会偷偷补充为事实。

按类型和原视频轮转选取；默认每视频/类型最多 20 条、总量 1000。只对相同标签、类型、术式、作用域和图片哈希去重，不声称语义去重。稀缺类型不由模型补造。

```bash
python -m videospy.visual_memory.cli validate \
  --cases output/visual_memory/supervised_facts_v3/cases.jsonl \
  --allow-source-grounded

export WEMM_EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
python -m videospy.visual_memory.cli build \
  --cases output/visual_memory/supervised_facts_v3/cases.jsonl \
  --output output/visual_memory/supervised_index_v3 \
  --embedding-config config/videospy/visual_memory_wemm.yml \
  --allow-source-grounded
```

新记录是 `verification.status: source_grounded`，`verified: false`，记录的是外部标签与对齐监督来源，不声称导出结果又经过一次人工检查。建库和在线均需显式允许这种来源。旧 model_reviewed 记录仍使用独立权限；不能直接把旧库改状态绕过本次组装。空库 build 会拒绝，先查看 pending。

## 6. 在线使用及回退

在服务器原评测命令中叠加 `visual_memory_wemm.yml` 和 **最后叠加** `visual_facts_supervised.yml`，修改 database_path 为新索引位置。关闭组最后叠加 `visual_facts_supervised_off.yml`。服务器专有 runner 不在本仓库，其他评测参数保持原样。

新配置明确设定 `usage_mode: inject`、`allow_source_grounded: true`、`allow_model_reviewed: false`。请从服务器命令中移除旧 `visual_facts_review.yml` / `visual_facts_current_only.yml` 及自定义的 `review:` 配置，它们已删除；旧模式会给出迁移错误，不会悄悄启用另一种行为。每次 `clip_skim` / `frame_inspect` 只有原来一次局部视觉模型调用，另外有 WeMM 请求。每题最多检索 4 次，每次最多 2 个参考，支持跨任务事实映射和同源排除。关闭时不读索引、不调用嵌入服务。

当前图片与历史图片明确区分。输入模型的参考只有标签、图像和必要来源/作用域；历史题干、错误选项、原预测、原解释不会作为监督示例传入。完整标注和审计记录保留在 diagnostics。阶段/病灶属性仍要求相同且非空术式。

`min_similarity: 0.0` 是未校准的候选阈值；检索相似不证明适用。帧嵌入平均也不编码时序。正确配对的事实能否帮助 9B，需要独立评测，不能由单元测试证明。

验证时固定原模型、帧采样、预算和评测划分，比较 off 与新配置，并统计实际命中、原错改对/原对改错。不要混用旧自动生成库，以免无法判断标签来源的影响。

运行时回退：最后叠加 `visual_facts_supervised_off.yml`（旧 `visual_memory_off.yml` 也可），或设 `tools.visual_memory.enabled: false`。原案例库与离线 VLM `mine` 入口保留，后者需显式 `--allow-generated-labels`；它不属于当前实验。在线 review 已删除，可通过 Git 历史恢复旧版本。新建库请用新目录，不覆盖旧库。本地回滚包可恢复本次修改前代码，并拒绝覆盖后续修改。

增量包解压到仓库根目录后，代码回滚先检查再执行：

```bash
python .rollback/restore_supervised_facts.py --root /path/to/MedVideospy
# 需要恢复本次修改前的代码时执行：
python .rollback/restore_supervised_facts.py --root /path/to/MedVideospy --apply
```

脚本只恢复清单内的原文件、删除本次新增文件，不删除事实库、视频或实验结果。没有 `--apply` 时只读检查；检测到后来修改过的代码时拒绝覆盖。
