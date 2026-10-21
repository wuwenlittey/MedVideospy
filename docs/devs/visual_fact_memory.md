# 医学视觉事实：批量提取、跨任务复用与观察后核对

这一版在原视觉案例记忆上增加两个可以独立验证的模块：离线大模型事实整理，以及在线观察后的图文核对。不做 RSI，不训练模型参数，不改变原视频采样策略。

原有 `visual_memory_off.yml`、`visual_memory_text.yml`、`visual_memory_image_text.yml` 仍可用。新功能均需显式选择配置。WeMM-Embedding-2B 仍只负责向量，不负责文字生成或事实审核。

## 1. 工作流程和真实性边界

离线：

```text
整批轨迹 + 原始图像/视频 + 明确的视频划分
→ 遍历全部成功 clip_skim / frame_inspect 调用
→ VLM 直接看图提出原子事实（不提供旧模型解释或整题 GT）
→ 每条候选单独审核，只看它引用的图片
→ 检查证据 ID、动作时序、作用域
→ 图文重复项去重，按事实类型及原视频轮转挑选
→ cases.jsonl + pending.jsonl + report.json + checkpoints/
```

不按整题答对与否筛选。中间观察可从最终答错、输出格式错误、统计失败等轨迹中提取。失败、重复跳过、只有 Overview 的调用不直接提供已核实局部事实；报告会显示被跳过的原因。不会为了补齐类型而生成不存在的事实。

事实类型为：`instrument_identity`、`anatomy`、`action`、`visible_state`、`spatial_relation`、`phase`、`lesion_attribute`、`visible_text`。动作要求至少两张有序图片和明确可见变化。阶段需要实际可见的区分依据。病灶属性仅描述外观，组织学、全片计数/排名、首次出现和未经标定的物理尺寸不会自动成为本地视觉事实。文字转录保留为文字，不自动转换成医学结论。

自动导出记录是 `verification.status: model_reviewed`、`verified: false`，并记录提取/审核模型是否相同。独立调用不等于独立真值，同一个模型可能重复同一错误。结构检查不能证明视觉判断正确。建议抽查小批样本，分别检查器械、动作、部位等类型再全量运行。

## 2. 准备原视频划分与轨迹清单

原始数据只读；输出放在新的 `output/visual_memory/...` 目录。按原视频/患者划分，不按 QA 奇偶号划分。视频 ID 必须与运行时 `memory_context` 中的身份一致，不能使用单个 QA ID 代替。跨数据集复用的小整数患者 ID 应加数据集前缀。

若使用 MedHorizon 的视频记录加 `qa` 数组 JSON，创建 `video_splits.jsonl`。`dataset_video_key` 精确对应数据集的 `key`、`video_path_original` 或 `video_path` 中任意一个，**每个视频仅选择一种键**：

```json
{"dataset_video_key":"tmvp/087.mp4","source_video_id":"087","split":"train","procedure":"mitral_valve_repair","video_path":"/datasets/videos/tmvp/087.mp4"}
{"dataset_video_key":"tmvp/088.mp4","source_video_id":"088","split":"test","procedure":"mitral_valve_repair","video_path":"/datasets/videos/tmvp/088.mp4"}
```

这里只是格式示例，术式与视频 ID 应填写实际值。可填 `patient_id`、`source_id`。训练侧有帧目录时，可以用 `frame_paths`、`frame_root`、`fps` 替代 `video_path`，与旧版 curate 的格式一致。所有源路径相对清单文件解析。

```bash
python -m videospy.visual_memory.cli inventory \
  --trajectories /runs/trajectories \
  --dataset /datasets/MedHorizon.json \
  --splits /runs/video_splits.jsonl \
  --output output/visual_memory/inventory_v2
```

清单没有指定媒体路径时，可加 `--video-root /datasets/videos`，使用数据集的原视频相对路径。清单内未出现的视频、测试视频、题干不对应或重复的轨迹会记录在 `skipped.jsonl`。轨迹 ID 优先用 `uid/sample_id/id`，否则使用 JSON 文件名（如 `283.json`）。题干支持原始题目、`Question:` 前缀，以及另起行附加选项；其他自定义包装需先规范化。

若轨迹记录了视频身份，会与清单核对；服务器路径改变时可在划分行中显式填写 `trajectory_video_aliases`。没有视频身份的旧轨迹只能按 QA ID 和题干对应，这一限制会写入 `inventory_provenance.video_identity_check`，需确认原视频路径。工具会拒绝同一个 canonical 视频、同一患者或同一 resolved 媒体路径跨集合，但无法识别两个独立复制文件是否来自同一次手术，应在划分清单中统一其原视频身份。

不使用该数据格式时，可直接提供 `sources.jsonl`，每条轨迹一行，无须指定某一个工具调用：

```json
{"trajectory_path":"/runs/train/283.json","source_video_id":"train_video_283","source_sample_id":"283","split":"train","task_type":"instrument_onset","procedure":"actual_procedure","video_path":"/datasets/videos/train_video_283.mp4"}
```

`split` 只能为 train/dev；最终测试集不能进入建库流程。GT 可作为来源信息保留，但这条自动管线不会把 GT 广播成帧标签，也不会将模型最终答案当真值。旧的已人工核验 QA 案例仍可使用旧流程。

## 3. 配置看图大模型，先提取一小批

使用项目现有的 `ModelClient` / OpenAI 兼容聊天接口。模型必须支持多图输入及 JSON 输出，`model_name` 遵循现有 LiteLLM 命名约定。

```bash
export FACT_MINER_MODEL=openai/your-served-vision-model
export FACT_MINER_API_BASE=http://127.0.0.1:8000/v1
export FACT_MINER_API_KEY=EMPTY
```

这里的 `EMPTY` 用于无鉴权的本地服务；有鉴权时通过环境变量设置真实密钥。不要把密钥写入 YAML 或轨迹。PowerShell 使用 `$env:FACT_MINER_MODEL='openai/your-served-vision-model'` 等形式。

配置位于 `config/videospy/visual_fact_mining.yml`。默认提取和审核使用相同模型、不同请求；`reviewer_vlm` 可单独指定另一个模型及端点。默认一次观察最多提出 6 条候选，每条候选另用一次请求审核：每观察最多 7 次 VLM 调用，另计失败重试。默认首次处理 5 个未完成观察，便于先检查接入和质量。

```bash
python -m videospy.visual_memory.cli mine \
  --sources output/visual_memory/inventory_v2/sources.jsonl \
  --config config/videospy/visual_fact_mining.yml \
  --output output/visual_memory/facts_v2
```

检查 `report.json` 的类型、来源任务、视频分布、缺失类型、待定/驳回数量、模型调用数和失败项，以及 `cases.jsonl` 对应的真实图片。`pending.jsonl` 保留待定/拒绝候选，不会送入索引。整组原观察帧超过 `max_images_per_observation` 时显式跳过，不默默删帧后保留标签。

确认接入后，将 `max_observations` 改为 `null`，继续同一运行：

```bash
python -m videospy.visual_memory.cli mine \
  --sources output/visual_memory/inventory_v2/sources.jsonl \
  --config config/videospy/visual_fact_mining.yml \
  --output output/visual_memory/facts_v2 --resume
```

已完成的抽取/审核不会再次请求；失败的候选可重试。预算可以改变，其余来源、图片、模型、提示或选择配置改变时拒绝混合恢复，需新建目录。恢复会重新验证源图片并重新生成汇总清单，不改变已建成的独立索引。

类型平衡只从审核通过的已有事实中挑选；默认每个视频/事实类型最多 20 条，总量 1000，先按类型轮转，再按原视频轮转。去重基于规范化事实文字、事实类型和支持图像哈希，不宣称完成语义等价去重。纯粹增加同一种器械的近似描述仍需通过报告和抽查发现。

## 4. 使用已有 WeMM 建立新索引

继续使用 `visual_memory_wemm.yml` 与服务器现有 WeMM 服务。自动审核事实需要显式允许进入实验库：

```bash
python -m videospy.visual_memory.cli validate \
  --cases output/visual_memory/facts_v2/cases.jsonl --allow-model-reviewed

export WEMM_EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
python -m videospy.visual_memory.cli build \
  --cases output/visual_memory/facts_v2/cases.jsonl \
  --output output/visual_memory/facts_index_v2 \
  --embedding-config config/videospy/visual_memory_wemm.yml \
  --allow-model-reviewed
```

不加 `--allow-model-reviewed` 时自动拒绝机器审核记录；运行时也需要 `allow_model_reviewed: true`。两个显式开关用于区分机器事实实验与原人工核验案例，不能视作质量认证。库内仍保留 `verified: false`。空库不允许 build，应先查看缺失类型与失败原因。

向量仍为帧嵌入归一化后平均，不编码动作先后顺序；有序支持帧会完整提供给 VLM 做比较。图像相似度是候选排序信号，不是医学正确性分数。

## 5. 在线复用与独立消融

在服务器原评测命令中叠加下面配置；实际 runner 的其他参数保持原样。`database_path` 根据实际索引位置修改。含检索的组还需叠加 `visual_memory_wemm.yml`。

| 实验 | 配置文件 | 局部 VLM 调用 |
|---|---|---|
| 关闭全部视觉记忆 | `visual_memory_off.yml` | 原始一次 |
| 旧 QA 案例直接注入 | 原 `visual_memory_image_text.yml` | 一次 |
| 新事实库直接注入 | `visual_facts_inject.yml` | 一次 |
| 原观察后只复看当前画面 | `visual_facts_current_only.yml` | 最多两次，无嵌入/索引 |
| 原观察后与事实对照 | `visual_facts_review.yml` | 有合格参考时最多两次 |

在线 review 的第一次 VLM 请求保持原观察内容，完全不含历史参考。第二次请求使用相同的当前采样图片、第一次观察及少量参考，逐条输出支持/冲突/未知、当前帧 ID、参考 ID、可见区别和局部修订建议。主 Agent 收到原观察与核对建议，原观察不会被覆盖。引用非法、动作证据不足、无参考、调用失败或只有未知判断时，保留原观察。

两个新实验开关互相独立：使用新库而不核对可选 inject；旧 132 个案例也能使用 review，只需 `match_mode: task`、旧索引路径，便于固定检索和案例库比较机制。`review.model: {}` 默认沿用局部工具模型，也支持 `model_name_env/api_base_env/api_key_env` 指定另一个模型。

`match_mode: fact` 只会跨任务使用新 `visual_fact`，不会把旧 QA 答案跨任务传播。器械等事实作为相关子问题的参考，不能直接替代阶段起点、全片统计等任务的答案。阶段和病灶属性参考要求双方 `procedure` 非空且相同；任务和术式名称应规范化。

新路由内置常用别名；未知任务会显式跳过。可使用任务到事实类型的配置补充服务器特有命名：

```yaml
tools:
  visual_memory:
    match_mode: fact
    task_fact_types:
      your_instrument_task: [instrument_identity, visible_state, spatial_relation]
      your_local_action_task: [action, instrument_identity, anatomy]
```

显式 `fact_types`（所有任务共用）优先于 `task_fact_types`，后者优先于内置映射。建议多任务实验优先使用细分映射。当前问题的任务和原视频身份依旧要由 runner 每题传入，`procedure` 有条件时一起提供。

## 6. 检查收益与回滚

轨迹 `diagnostics.visual_memory` 保存原观察、参考事实及分数、完整排除原因计数、原帧/参考图像来源、核对结果、调用次数和耗时。新增 VLM 调用通过现有 ModelClient 计入工具调用与 token 记录。机器核对结果不会自动写回事实库。

分别汇报全部题目和有实际参考题目的准确率、原错改对/原对改错、任务分布，以及调用/token/时延。任务类型匹配不等于检索适用，更不等于核对后修正成功。无参考时 review 少一次调用，比较同预算控制时应在有参考子集单独比较；整体成本也需要保留。

固定原案例库验证核对机制，再固定机制验证扩库。首次实验不要同时改帧数、嵌入模型和匹配规则。运行时关闭 `tools.visual_memory.enabled` 即回到原流程；旧索引不覆盖，新索引采用新目录。修改前源文件与逐文件校验回滚脚本随本次本地交付包提供，发现后续修改时回滚脚本会拒绝覆盖。

将增量包解压到仓库根目录后，可先只读检查回滚范围：

```bash
python .rollback/restore_visual_facts.py --root /path/to/MedVideospy
# 确认需要恢复这一版代码时，显式执行：
python .rollback/restore_visual_facts.py --root /path/to/MedVideospy --apply
```

脚本只恢复清单中的 6 个原文件并删除本次新增的 14 个文件，不删除数据、运行记录或已建索引。默认没有 `--apply` 时不写文件。

本地验证使用合成图像、假 VLM/嵌入响应和原有回归测试；真实事实质量、病例覆盖和准确率仍需在服务器模型及隔离评测集上测量。
