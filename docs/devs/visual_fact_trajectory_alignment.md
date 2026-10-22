# 文本大模型参与的离线轨迹整理

新流程是 `inventory-trajectories → align-trajectories → assemble → build`。文本模型负责判断轨迹可用性、理解题干与观察调用的关系并提出候选配对；脚本负责复制标准标签、核验引用、重建实际观察帧及保存来源；最后由服务器上的 WeMM-Embedding-2B 建立索引。在线仍是一次原有局部观察，不恢复旧 support/conflict review。

## 1. “明确配对”与“模型推断配对”

“已有明确配对”指已有标注明确说某条描述对应指定帧或事件区间，例如“120–135 秒属于关闭肠系膜缺损阶段”，并记录标注来源/确认人。`correct=true`、最终答对或某次 clip 覆盖该区间，都不是这种标注。脚本只核对来源、时间和一致性，不会重新证明标注的医学真实性。

普通轨迹可能只有题干、标准答案、调用参数和旧观察结果。现在由文本模型阅读这些内容，判断是否能把题目所指对象/时间与一次或多次实际观察联系起来。成功、失败轨迹一视同仁，不以最终对错筛选。

结果分为：

- **usable**：存在可解释的题干到观察调用的联系，提出候选配对；通过程序检查才导出。
- **uncertain**：指代/时间/画面覆盖不清楚，或只有旧模型的视觉猜测作为依据，留待确认。
- **unusable**：答案无法成为当前证据支持的可复用局部视觉标签，例如全片总数、全局评分、首次出现时间、没有视觉依据的组织学诊断等。

模型收到的题干会去除常见 A./B. 选项段，标准答案单独提供。不会传入最终预测、答对标记或 Agent 的推理过程。成功工具的 observation 文本会保留并显式标为可能出错；它只能作为候选关联上下文，不能成为新事实标签，也不能单独证明定位。模型接收的是文本与帧编号/时间清单，**不传入图片**。

## 2. 约束与边界

- `label_text` 必须逐字等于轨迹保存的完整标准答案，不能改写、扩写或把旧模型描述替换进去。复合答案无法共同落到局部证据时保留不确定；本版不自动拆解答案生成新标签。
- 每个候选必须给出题干原文引用、被选观察的原文引用、工具消息编号、实际观察帧编号、类型、作用域和关联理由。
- 脚本逐字检查引用，检查工具调用成功且帧确实属于被引用的观察；拒绝编造标签、帧、工具编号或引用。不让模型输出任意新时间或补采帧。
- 动作需要至少两帧有序证据；不能把阶段标签直接改成器械或解剖事实。每个候选最多 16 帧，超过上限留待确认，不静默裁掉证据。
- 全部候选的来源为 `trajectory_aligned`、`verified=false`，与外部标注 `source_grounded`、旧视觉模型生成的 `model_reviewed` 分开。`assemble` 会再次读取原轨迹、核验保存的标准答案并重建观察记录，防止候选被移接到别的视频或轨迹修改后继续使用。
- 只能使用 train/dev 原视频，按原视频/患者划分；来源路径、身份和跨划分冲突检查继续生效。测试轨迹即使保存了标准答案也不得加入记忆库。

**机械检查不能证明语义关联正确。** 原文引用真实不代表推断成立；文本模型仍可能选错观察。因此这里只能保证标签不被生成、证据引用可追溯，不能宣称“绝不产生错误图文对”。先抽检候选与原图，分别统计可用率、配对正确率及在线净纠错效果。语义不确定应由模型明确留待确认，而不是靠自报置信度强行入库。

## 3. 服务器命令

先沿用[轨迹字段映射和视频划分配置](visual_fact_supervised.md)，无需外部数据集文件：

```bash
python -m videospy.visual_memory.cli inventory-trajectories \
  --trajectories /runs/train_dev_trajectories \
  --splits /runs/video_splits.jsonl \
  --schema config/videospy/visual_fact_trajectory_schema.yml \
  --output output/visual_memory/inventory_v4

# 文本整理模型，独立于在线 9B 及 WeMM；模型名称遵循当前 ModelClient 配置。
export FACT_ALIGNER_MODEL=openai/YOUR_TEXT_MODEL
export FACT_ALIGNER_API_BASE=http://127.0.0.1:8002/v1
export FACT_ALIGNER_API_KEY=YOUR_KEY

python -m videospy.visual_memory.cli align-trajectories \
  --sources output/visual_memory/inventory_v4/sources.jsonl \
  --config config/videospy/visual_fact_alignment.yml \
  --output output/visual_memory/alignment_v4
```

配置默认先处理 5 条轨迹，便于检查输出。原命令加 `--resume` 继续下一批；将 `max_sources: null` 改为处理全部。已完成的可用/不确定/不可用决定不会重复请求模型，接口失败可在 resume 时重试。变更模型、输入、媒体或提示词后恢复会被拒绝，须换输出目录。输入超过 `max_packet_chars` 时留待确认，不静默截断；上下文长度按字符控制，需按服务器模型容量调整。

输出 `annotations.jsonl` 是通过引用/编号检查的候选，不是人工金标。`pending.jsonl` 保留不确定、不可用、非法输出及请求错误；`checkpoints/` 保存提交的文本包、响应及调用统计，`decisions.jsonl` 和 `report.json` 汇总可用性、类型、模型调用数和待处理数量。API 密钥不写入配置身份或报告。

```bash
python -m videospy.visual_memory.cli assemble \
  --sources output/visual_memory/inventory_v4/sources.jsonl \
  --annotations output/visual_memory/alignment_v4/annotations.jsonl \
  --config config/videospy/visual_fact_assembly.yml \
  --output output/visual_memory/trajectory_facts_v4

python -m videospy.visual_memory.cli validate \
  --cases output/visual_memory/trajectory_facts_v4/cases.jsonl \
  --allow-trajectory-aligned

export WEMM_EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
python -m videospy.visual_memory.cli build \
  --cases output/visual_memory/trajectory_facts_v4/cases.jsonl \
  --output output/visual_memory/trajectory_index_v4 \
  --embedding-config config/videospy/visual_memory_wemm.yml \
  --allow-trajectory-aligned
```

如果 `sources.jsonl` 为空，先看 inventory 报告中的标准答案缺失/字段映射错误；如果没有候选或组装后无案例，不要直接建库，应检查 pending。若来源还带有旧外部 supervision，输出可能包含两类来源；仅在确实要混合实验时另加 `--allow-source-grounded`，不要把两种状态互相改名。

## 4. 在线开关与对照

在原服务器命令中先加载 WeMM 配置，最后加载 `config/videospy/visual_facts_trajectory.yml`，修改 `database_path` 指向新索引。它只允许 `trajectory_aligned` 候选，默认排除旧 model_reviewed 和外部 source_grounded 案例，便于独立评估。现有 `visual_facts_supervised.yml` 默认不允许新来源。

关闭时最后加载 `visual_facts_supervised_off.yml` 或设 `tools.visual_memory.enabled: false`，不会读库或调用嵌入服务。新文本模型仅在离线 `align-trajectories` 使用，在线不会调用它，也没有额外的 review。历史题干、引用、推断理由仅保留审计，不作为参考推理文本注入当前观察；模型输入保留参考图片、原样标签、作用域和明确的 inferred 来源状态。

离线大模型整理是独立阶段：已有明确标注可跳过它，继续原 `inventory → assemble → build`。新旧索引使用不同目录，不覆盖之前的库与实验结果。
