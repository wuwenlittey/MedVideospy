# Evolving Video Agent：从通用 VideoSpy 到领域专业智能体
<!-- Evolving Foundational Video Agent: 从这个角度去讲故事。 -->

## 1. Introduction

视频理解要求模型在长时序、多模态信息中定位证据、组织推理，并根据任务需要调用不同的视频工具。现有视频智能体通常依赖人工设计的提示词、工具和执行流程。系统部署后，其行为方式基本固定。即使智能体持续处理新的带标签样本，这些样本也主要用于评测，难以自动转化为智能体自身的能力提升。

本研究受到深度学习训练范式的启发。典型流程先固定网络架构，再按 Epoch 和 Batch 遍历训练数据。模型在每个 Batch 上计算预测误差，并通过反向传播和优化器更新权重，测试集只用于评价最终模型的泛化性能。在这一过程中，网络架构通常保持稳定，模型权重则通过训练持续更新。

本文将这一流程推广到视频智能体。我们首先构建基础视频智能体 VideoSpy，并固定 Agent 框架代码、Agent Loop、基础 System Prompt 模板、Prompt 组装机制、工具接口和轨迹格式。在训练集上，VideoSpy 产生答案和完整执行轨迹，Reflector 根据 Ground Truth 提炼成功经验与失败原因，Agent Optimizer 再更新任务 Skill、长期 Memory 和执行超参数。在线协议让更新后的 Agent 直接处理下一个 Batch；离线协议则复用一次基础推理的固定轨迹进行单 Epoch 蒸馏。测试集只用于评价最终 Agent。整个过程中保持不变的是 Agent 骨架、Prompt 骨架及其组装方式，随训练变化的是 Skill、Memory 和执行超参数。

![智能体进化框架与神经网络学习框架的类比](figures/deep-network-vs-agent-evolution-v2.png)

经过不同领域数据的持续训练，同一个初始 VideoSpy 可以逐步形成面向不同任务的专业智能体。本文重点研究三个问题：

1. 如何从答案与执行轨迹中稳定地提炼有用经验？
2. 不同经验提炼方式和不同 Agent 更新路径分别有何作用？
3. 进化后的 Agent 能否在通用、第一视角生活和医疗手术视频中形成专业能力，并达到有竞争力的性能？

本文的主要贡献暂定如下：

1. 提出一个面向视频理解的智能体进化框架，将带标签样本从评测资源转化为 Agent 的持续优化信号。
2. 设计基于结果评估和轨迹分析的经验提炼机制，并研究 Agent 自我反思与外部 Agent 分析两种实现方式。
3. 研究任务 Skill、Memory 和执行超参数三条更新路径，分析它们对领域专业化的作用。
4. 在通用长视频、第一视角生活视频和医疗手术视频上评估 Agent 的进化过程、领域泛化和模型泛化。

## 2. Related Work

### Video Understaning Task

任务输入输出

各种领域、各种场景，由此出现各种Benchmark。如长视频理解、医疗视频理解、第一人称流式视频理解。

带来的挑战，每种场景和任务可能都需要对应的 agent 设计。

### Agentic Video Understanding 长视频理解与视频智能体

[VideoAgent](https://arxiv.org/abs/2403.10517) 使用大语言模型规划证据搜索，并调用视觉模型迭代定位回答问题所需的少量关键帧。另一项[记忆增强的 VideoAgent](https://arxiv.org/abs/2403.11481) 构建视频事件和目标状态的结构化 Memory，并通过视频定位和目标查询工具处理长时序关系。[Adaptive Video Understanding Agent](https://arxiv.org/abs/2410.20252) 进一步结合查询自适应抽帧和语言自反思，在单次视频理解过程中改善证据选择效率。

现有视频智能体主要关注单次任务中的分段、检索、记忆和工具调用，其 Agent 配置在跨样本执行过程中通常保持不变。本文研究另一层问题：如何利用带标签任务的结果和执行轨迹，持续更新视频 Agent 的任务技能、经验 Memory 和工具使用策略，使同一个基础 Agent 逐步形成领域专业能力。 

### 智能体反思与自我改进

语言反馈为不更新模型权重的 Agent 改进提供了基础。[Reflexion](https://arxiv.org/abs/2303.11366) 将任务反馈转化为自然语言反思，并写入情景 Memory 以指导后续尝试，重点改善同一任务中的多轮决策。[ExpeL](https://arxiv.org/abs/2308.10144) 进一步从多个训练任务的成功与失败轨迹中提炼自然语言经验，在新任务中检索相关经验，实现跨样本的经验学习。[Agent-Pro](https://aclanthology.org/2024.acl-long.292/) 则从完整交互轨迹和动态信念中进行策略级反思，持续修订 Agent 的行为策略。

这些方法说明执行反馈可以转化为非参数化的学习信号，但主要面向文本推理、决策或交互环境。本文关注视频智能体中的轨迹分析，研究如何从答案、视频证据获取过程和工具调用记录中提炼经验，并比较 Self-Reflector 与 External-Reflector 两种实现。

### 2.3 无梯度的 Agent 优化

[AgentOptimizer](https://arxiv.org/abs/2402.11359) 将 Agent 的函数视为可学习参数，根据训练任务的执行历史和 Ground Truth 迭代更新函数集合，并通过回滚和早停控制优化过程。[GEPA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) 从推理、工具调用和工具输出等执行轨迹中生成语言反思，提出并验证 Prompt 候选，通过 Pareto 选择保留互补的优化经验。[TextGrad](https://www.nature.com/articles/s41586-025-08661-4) 则将语言反馈视为文本梯度，用于迭代优化复合系统中的文本组件。与 Prompt 或函数的单一路径优化不同，本文将 Skill、Memory 和工具与执行超参数共同视为可进化状态，并通过统一的经验反馈按批次持续更新。

另一类方法将成功轨迹、错误反馈或偏好信号转化为监督微调、偏好优化和强化学习数据，直接更新模型权重。本文固定底座模型和工具模型的权重，重点研究 Agent 配置层面的进化，模型参数学习留作未来工作。


## 3. 视频智能体进化框架

本节介绍 VideoSpy 的整体进化方法。如图所示，基础 VideoSpy 在带有 Ground Truth 的训练任务上执行视频理解，生成最终答案和完整轨迹。Reflector 对比预测结果与 Ground Truth，通过 Agent 自我反思或外部 Agent 分析，从轨迹中提炼正确经验和错误经验。Agent Optimizer 随后将这些信号转化为可复用的任务 Skill，或用于更新长期 Memory 和工具与执行超参数。在线协议让更新后的 Agent 继续生成后续 Batch 轨迹；离线协议让多个 Reflector 复用同一批基础轨迹。下文依次介绍问题定义、基础 VideoSpy、Reflector、Agent Optimizer，以及两种训练协议与领域专业化机制。

![VideoSpy 智能体进化框架](figures/videospy-self-evolution-framework-v2.png)

### 3.1 问题定义

给定某一领域的带标签视频理解训练集

$$
\mathcal{D}_{\mathrm{train}}=\{(x_i,y_i^*)\}_{i=1}^{N},
$$

其中，$x_i=(v_i,s_i,q_i)$ 表示任务输入，分别包含视频 $v_i$、可选字幕 $s_i$ 和问题 $q_i$，$y_i^*$ 表示 Ground Truth。第 $t$ 个优化步骤的智能体表示为

$$
A_t=(F,\Theta_t),
$$

其中，$F$ 表示固定的 Agent 运行骨架，包括框架代码、Agent Loop、基础 System Prompt 模板、Prompt 组装与上下文注入机制、工具接口和轨迹结构，$\Theta_t$ 表示可进化部分：

$$
\Theta_t=(S_t,M_t,H_t).
$$

这里，$S_t$ 是从执行轨迹中提炼的任务 Skill，$M_t$ 是经验 Memory，$H_t$ 是工具和执行超参数。本文固定底座模型及工具内部模型的权重，不将其作为进化变量。设 $P_{\mathrm{core}}$ 为固定的基础 System Prompt，$\Gamma$ 为固定的 Prompt 组装函数，则第 $t$ 轮 Agent 在第 $j$ 个决策步骤实际使用的 Prompt 为

$$
P_{t,j}(x)=\Gamma\left(
P_{\mathrm{core}},
\operatorname{MatchSkill}(S_t,x),
\operatorname{RetrieveMemory}(M_t,x,\tau_{<j})
\right).
$$

其中，$\tau_{<j}$ 表示当前步骤之前已经产生的部分轨迹。当 Agent 尚未调用 `Memory Search` 时，Memory 检索结果为空。问题、选项和输出要求属于输入 $x$，由 Benchmark runner 构造，不是另一类可学习或固定的 Task Schema。因此，固定的是 Prompt 骨架和组装机制，最终实例化的 $P_{t,j}(x)$ 可以随 Skill、Memory、任务输入和当前轨迹发生变化。智能体处理任务后得到答案与轨迹：

$$
(\hat y_i,\tau_i)=A_t(x_i).
$$

进化过程利用一个 Batch 中的 $(\hat y_i,\tau_i,y_i^*)$ 生成可追溯的 Batch 反思报告 $G_t$，并通过确定性 Agent Optimizer 得到下一版本配置：

$$
G_t=R(\{(\hat y_i,\tau_i,y_i^*)\}_{i\in B_t}),\qquad
\Theta_{t+1}=U(\Theta_t,G_t),
$$

其中，$R$ 表示 Batch 反思过程，$U$ 只负责校验、合并和保存候选更新，不再发起第二轮模型概括。每处理一个 Batch 生成一个新版本，并直接将其用于后续 Batch。

### 3.2 基础视频智能体 VideoSpy

基础 VideoSpy 是进化开始前的通用视频智能体，对应初始版本 $A_0=(F,\Theta_0)$。它以视频、可选字幕和问题 $x_i=(v_i,s_i,q_i)$ 为输入，通过多轮工具调用收集视觉证据，最终输出预测答案 $\hat y_i$ 和完整执行轨迹 $\tau_i$。基础版本尚未包含从训练轨迹中学习得到的 Skill 或经验 Memory。若采用从零进化设置，则

$$
\Theta_0=(S_0,M_0,H_0),\qquad S_0=\varnothing,\quad M_0=\varnothing,
$$

其中，$H_0$ 是人工设定的初始超参数。若实验保留人工编写的初始任务策略，则应显式记为 $S_0=S_{\mathrm{human}}$，并与 $S_0=\varnothing$ 的无 Skill 基线区分。任务要求、输出字段和评测格式属于输入 $x$，不计入 Skill。证据搜索、工具调用、验证、推理和停止策略属于可进化的 $S_t$。

VideoSpy 采用常见的 ReAct Agent Loop，通过原生函数调用形成“思考、行动、观察”的闭环。模型根据问题、字幕和已有视觉证据进行思考，选择并调用视频工具执行行动，再将工具返回的观察加入上下文，进入下一轮决策。当证据收集结束或达到最大步数时，系统通过一次独立的无工具模型调用生成最终答案。全部消息、工具调用、工具观察和内部模型调用均被保存到轨迹 $\tau_i$，供后续 Reflector 分析。VideoSpy 使用三个粒度互补的视觉工具：`overview` 对完整视频均匀抽帧以建立全局认识，`clip_skim` 浏览指定时间区间以定位相关片段，`frame_inspect` 检查一个或多个指定时间点的原分辨率帧，以验证细粒度信息或比较候选时刻。

固定部分 $F$ 包括 Agent 框架代码、Agent Loop、基础 System Prompt 模板、Prompt 组装与上下文注入机制、三个视觉工具、`Memory Search` 接口，以及统一的轨迹结构。这些部分定义所有进化版本共同遵循的执行边界，在进化过程中保持不变。这里固定的是 Prompt 骨架和组装机制，而不是每次执行时最终实例化的完整 Prompt。

可进化部分为 $\Theta_t=(S_t,M_t,H_t)$。$S_t$ 是按任务类型和适用条件组织的程序性策略，实现时以 Prompt 片段填入基础 System Prompt 预留的 Skill 插槽。$M_t$ 是可检索的情景性经验，通过固定的 `Memory Search` 接口按当前问题和轨迹状态检索。$H_t$ 只包含白名单内的 Agent 步数、视频抽帧数量、片段时长和字幕上下文窗口。本文在所有进化步骤中使用相同的模型权重，只更新 $\Theta_t$，不改变固定部分 $F$。

### 3.3 Reflector：如何从结果与轨迹中提炼经验

Reflector 对应神经网络训练中的 Loss 计算与反向传播，输入预测答案、Ground Truth 和执行轨迹。系统先用逐样本指标或轻量 Result Judge 将结果分为 good、bad 和 uncertain，再按 outcome 与标准化任务类型确定性分层抽样。`max_samples` 表示一次反思的总样本上限，`group_size` 表示单次 Reflector 调用包含的轨迹数。LVBench 小规模验证和正式实验分别固定选择 16 条和 64 条，目标 bad 比例为 80%，每组 4 条。

为控制上下文，Reflector 接收确定性的 Reflection Digest，而不是完整对话副本。Digest 删除 System Prompt、完整字幕消息、模型内部字段和重复观察，保留任务结果、执行成本、工具序列、参数、观察摘要和错误信号。每条 Digest 不超过 10,000 字符，工具事件最多保留前 4 次和后 8 次。

每组 Reflector 为每条样本生成一个结构化 finding。随后按任务类型分别进行 Skill synthesis 和 Memory synthesis，形成唯一的 $G_t$。报告保存 `sample_findings`、`patterns`、`skill_candidates` 和 `memory_candidates`。Skill 只来源于至少两个独立样本支持的重复成功或失败模式；Memory 允许单样本情境经验，并且每个候选只绑定一个任务类型。同一 finding 可以同时支持二者，不使用互斥的目标选择。

**Self-Reflector** 复用任务 Agent 的最终生效模型配置。**External-Reflector** 在该配置上深合并一个显式指定的独立强模型。两者运行完全相同的数据分类、抽样、分组分析、综合提示和 JSON 结构，只改变模型来源。运行元数据记录实际模型、去除密钥后的配置哈希，以及 `agent` 或 `custom` 来源。

### 3.4 Agent Optimizer：经验如何驱动 Agent 进化

Agent Optimizer 输入当前配置 $\Theta_t=(S_t,M_t,H_t)$ 和 Batch 反思报告 $G_t$，输出下一版本 $\Theta_{t+1}$。三个 Optimizer 不再调用模型，只执行确定性的证据校验、增量合并、向量化和搜索空间约束，并在同一个 SQLite 事务中保存报告与完整版本。训练过程始终执行三个 Optimizer，不使用组件开关改变训练路径。

**Skill 更新**

Skill 表示跨样本抽象得到的程序性知识，回答“遇到某类任务时应该怎么做”。每项 Skill 必须引用至少两个独立样本支持的 pattern，并且候选任务类型必须与 pattern 一致。Skill Optimizer 只执行增量 upsert 或 remove，未出现在候选中的旧 Skill 保持不变。多标签任务会同时注入各任务类型的 Skill。

**Memory 更新与检索**

Memory 表示具体历史任务的情景性经验，明确分开适用情境、抽象观察模式、可应用教训、应避免陷阱、结果和失败类型。Memory synthesis 以逐样本 finding 为证据，但独立重写全部可注入文本，避免把 finding 中的具体答案或场景事实带入状态；候选只绑定一个规范化任务类型并生成语义向量。正式实验通过 OpenAI 兼容 API 生成默认 1024 维向量，并固定记录模型、配置哈希和实际向量维度；固定哈希向量只用于 Debug。推理时 `Memory Search` 可同时检索多标签任务对应的 Memory，并将其作为需要由当前视频证据验证的类比提示。

**工具与执行超参数更新**

对于能够归因到执行预算或采样策略的问题，Hyperparameter Optimizer 只在现有白名单和预设搜索空间中前进一步。一次更新至少需要两个独立 bad 样本支持，失败类型必须是当前 Batch 唯一最多的可行动问题，每个 Batch 最多更新一个参数，同一参数每个 Epoch 最多更新一次。证据不足、类型并列、缺少候选空间或达到上界时均记录 no-op 原因。

LVBench Offline 验证阶段冻结 $H_0$。Reflector 不生成超参数候选，Pipeline 在提交前断言 final 与 initial 的超参数快照完全一致。因此该阶段只比较 Skill、Memory 及二者组合，不把执行预算变化混入效果归因。

三条路径在训练中始终同时启用。Skill 与 Memory 的区别不取决于其是否以文本保存，而取决于知识粒度和使用方式：Memory 是供相似案例检索的实例级记录，Skill 是直接指导一类任务的程序性策略。每次更新及其来源经验和修改内容都记录在版本化 SQLite 中。训练完成后，推理阶段可以分别选择是否加载进化后的 Skill、Memory 和超参数，用于分析各组件对最终行为的贡献。

### 3.5 Batch 训练与测试

本文将每个领域的数据划分为训练集 $\mathcal{D}_{\mathrm{train}}$ 和测试集 $\mathcal{D}_{\mathrm{test}}$。两者在视频、日期或来源视频层面保持互斥，分别用于 Agent 自进化和最终评价。LVBench 进一步将 83 个训练视频固定划分为 66 个 Reflection 视频和 17 个 Dev 视频。Dev 中固定 5 个视频作为 68 题固定子集，用于低成本候选筛选。

训练集按时间顺序或数据源顺序组织，并划分为大小为 $B$ 的 Batch。在线模式下，当前 Agent $A_t$ 在一个 Batch 上执行任务，Reflector 生成 $G_t$，完整新版本随后用于下一个 Batch，形成闭环。离线模式下，基础 Agent 先对完整训练集执行一次推理，不同 Reflector 复用同一份只读轨迹并执行单 Epoch 的 Batch 经验蒸馏。离线模式提高实验复用效率，但不会让更新后的 Agent 重新生成训练轨迹。最后一个 Batch 的补齐样本在反思前去重。

分类、抽样、每组 finding、Skill synthesis、Memory synthesis 和最终报告分别保存检查点。Offline checkpoint 身份包含 schema、提示词、模型参数和输入摘要哈希。严格 JSON 模式失败时自动切换为普通文本 JSON，并支持代码块或首个完整 JSON 对象提取。在线和离线训练都可通过 `--run-dir` 续跑，身份匹配的模型调用不会重复执行。旧 Offline checkpoint 不混合恢复，已有完成态 SQLite 仍可用于推理。

训练结束后，系统冻结最后一个 Batch 产生的 Agent $A^*$，并在 $\mathcal{D}_{\mathrm{test}}$ 上进行一次最终评价。测试结果不反馈给 Reflector 或 Agent Optimizer。推理时默认加载 SQLite 中标记为 final 的版本，也可以显式指定历史版本用于分析。

在不同领域的训练集上独立运行这一流程，即可实现领域专业化。各领域 Agent 从同一个基础版本 $A_0$ 出发，共享固定框架 $F$ 和模型权重，分别学习领域配置 $\Theta^{(d)}=(S^{(d)},M^{(d)},H^{(d)})$。通用长视频、第一视角生活视频和医疗手术视频由此形成不同的任务 Skill、Memory 和采样策略，最后在各自测试集上评价泛化性能。

## 4. 实验

本节依次回答四个问题：不同进化机制是否有效，如何通过进化得到领域 Agent，最终 Agent 能否达到有竞争力的任务性能，以及进化具体改变了 Agent 的哪些行为。

**当前优先完成的实验（成稿时删除）**

1. 固定三个数据集的数据划分、评价指标和统一预算，得到基础 VideoSpy 的结果。
2. 完成 Reflector 和 Agent Optimizer 的核心消融，确定后续使用的完整进化方案。
3. 使用该方案在三个领域独立进化，得到最终领域 Agent 及其进化曲线。
4. 将三个最终 Agent 与对应 Benchmark 上的代表性方法和 SOTA 进行比较。
5. 在主结果完成后，再补充行为案例、跨领域迁移、小模型泛化和成本分析。

### 4.1 实验设置

实验覆盖三个领域。LVBench 评估通用长视频理解，包含 103 个视频和 1,549 道多项选择题。EgoLifeQA 使用公开的 A1_JAKE 子集，共 500 道跨小时和跨天的第一视角生活理解题。MedVidBench 评估医疗与手术视频理解，包含 6,270 个公开 Ground Truth 样本和 6,245 个无标签 Leaderboard 样本，覆盖 8 类任务。

按照表 1，三个数据集均划分为训练集和测试集。LVBench 在视频层面保持互斥，EgoLifeQA 在日期层面保持互斥，MedVidBench 在来源视频层面保持互斥。训练集用于执行、反思和持续更新 Agent，测试集只用于最终评价。LVBench 的 Reflection 与 Dev 同样按视频隔离，候选只根据 Dev 选择；冻结后的最终版本才运行一次 Test，结果不反馈给 Reflector 或 Agent Optimizer。所有实验从相同的基础 VideoSpy $A_0$ 出发，并固定 Agent 框架、模型权重和工具接口。更完整的场景类型、任务定义、评价指标和实现设置见附录 B。

<!-- [P0 主文核心表] 表 1：Benchmark、数据规模与实验划分。此表必须最先完成。 -->

| Benchmark | 场景与官方规模 | 任务类型 | 训练集 | 测试集 | Leaderboard Test | 主要评价指标 |
|---|---|---|---:|---:|---:|---|
| LVBench | 103 个视频，1,549 题，约 115.5 小时 | 6 类长视频 MCQ 能力 | 83 视频 / 1,234 题 | 20 视频 / 315 题 | — | 总体 Accuracy、6 类 Accuracy |
| EgoLifeQA | A1_JAKE 500 题 | 5 类长时生活记忆 MCQ | Day 1–5 / 405 题 | Day 6–7 / 95 题 | — | 总体 Accuracy、5 类 Accuracy、Macro Average |
| MedVidBench | Ground Truth 6,270，Leaderboard 6,245 | 8 类医疗视频任务，11 个变体 | 5,146 | 1,124 | 6,245 | 10 项官方任务指标 |

### 4.2 进化机制消融

首先固定完整的三组件进化过程，比较不进行进化、Self-Reflector 和 External-Reflector，分析独立强模型能否产生质量更高的 Batch 反思。随后固定表现更好的 Reflector 和同一个完整进化版本，在测试推理时分别加载 Skill、Memory、工具与执行超参数，以及组合三类结果，分析不同进化组件的作用与互补性。

**表 2：Reflector 消融。** 报告各经验提炼方式的最终性能、相对基础 Agent 的提升和进化成本。

<!-- [P0 主文核心表] 表 2：先在 LVBench 完成全部设置，再补充另外两个领域的三个关键设置。 -->

| 设置 | 使用最终结果 | 使用完整轨迹 | 反思主体 | LVBench | EgoLifeQA | MedVidBench | 进化 Token / Cost |
|---|:---:|:---:|---|---:|---:|---:|---:|
| 基础 VideoSpy |  |  |  |  |  |  |  |
| Self-Reflector | ✓ | ✓ | 任务 Agent |  |  |  |  |
| External-Reflector | ✓ | ✓ | 外部 Agent |  |  |  |  |

**表 3：进化组件应用消融。** Skill、Memory、超参数在训练中共同进化，测试时使用三个加载开关构造不同组合，报告最终性能和推理成本。该实验衡量完整进化后各组件的推理贡献，不解释为单个 Optimizer 独立训练的因果效果。

<!-- [P0 主文核心表] 表 3：固定表现更好的 Reflector 和同一进化版本，验证三类结果的推理作用及其组合。 -->

| 设置 | 经验驱动 | Skill | Memory | 超参数 | LVBench | EgoLifeQA | MedVidBench | 平均提升 | 测试 Token | 工具调用数 |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 基础 VideoSpy |  |  |  |  |  |  |  |  |  |  |
| Skill only | ✓ | ✓ |  |  |  |  |  |  |  |  |
| Memory only | ✓ |  | ✓ |  |  |  |  |  |  |  |
| Hyperparameter only | ✓ |  |  | ✓ |  |  |  |  |  |  |
| Full evolution | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |

当前阶段先使用 Qwen3.5-9B，在 LVBench 上完成 Self-Reflector、External-Reflector 以及四种进化组件加载设置的消融。四种组件设置共享同一个完整进化版本。根据该实验确定的 Reflector，再用于三个领域的独立进化。模型规模实验在核心结论成立后再补充。

### 4.3 领域智能体的进化过程

采用 4.2 确定的完整方案，从同一个基础 VideoSpy 出发，分别在 LVBench、EgoLifeQA 和 MedVidBench 的训练集上按 Batch 更新 Agent。训练协议固定后，冻结最后一个 Batch 产生的最终版本，再在测试集上统一评价。LVBench 和 EgoLifeQA 直接计算测试集指标，MedVidBench 同时报告本地测试结果并生成 Leaderboard submission。测试结果不反馈到训练过程。

<!-- [P0 主文核心图] 图 3：三个领域的进化曲线。 -->

**图 3：三个领域的进化过程。** 默认离线协议下，固定基础轨迹的 Batch 分数不能解释为更新后 Agent 的性能曲线，因此横轴使用累计训练样本数，纵轴报告累计 Skill 更新数、Memory 数和超参数更新事件，并同时给出各 Batch 的反思样本构成。若补充在线协议实验，才单独绘制当前版本在下一 Batch 上的在线任务性能。测试集始终只评价冻结后的最终版本，不绘制逐步测试曲线。

**表 4：领域 Agent 的形成结果。** 对每个数据集报告基础性能、最终性能、绝对提升、训练样本数、Epoch 数、优化步骤数，以及最终 Agent 的测试时推理成本。

<!-- [P0 主文核心表] 表 4：回答如何从同一个基础 Agent 得到三个领域 Agent。 -->

| 领域 Agent | 基础性能 $A_0$ | 最终性能 $A^*$ | 绝对提升 | 使用训练样本数 | Epoch | 优化步骤数 | 测试 Token | 工具调用数 | 推理时延 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LVBench Agent |  |  |  |  |  |  |  |  |  |
| EgoLifeQA Agent |  |  |  |  |  |  |  |  |  |
| MedVidBench Agent |  |  |  |  |  |  |  |  |  |

这一节的主要目标是说明，同一个基础 VideoSpy 能否通过相同的进化流程，稳定形成三个不同领域的专业 Agent。

### 4.4 与 SOTA 方法比较

将三个领域进化后的最终 VideoSpy 分别与对应 Benchmark 上的代表性视频理解模型和非进化 Agent 比较，同时保留基础 VideoSpy 作为直接参照。三个领域都在本文固定的本地测试集上比较基础与进化 Agent。LVBench 和 EgoLifeQA 不直接混用公开全量结果；MedVidBench 另使用官方 Leaderboard 测试结果与公开方法比较。

LVBench 和 EgoLifeQA 使用多项选择准确率，但任务类别不同，分别设置结果表。MedVidBench 包含异构任务和 10 项官方指标，需要使用独立的宽表。所有表格均明确标注目标领域标签的使用情况，使进化后的 VideoSpy 只与数据条件相同的方法作直接比较。

<!-- [P0 主文核心表] 表 5a：LVBench SOTA 对比。现有方法必须在本文固定测试集上重新运行。 -->

| 方法 | 方法类型 | 底座模型 | 使用 LVBench 训练标签 | 视觉输入 / 工具预算 | Overall | Entity | Event | Retrieval | Reasoning | Summary | Temporal |
|---|---|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| 现有代表性方法 |  |  |  |  |  |  |  |  |  |  |  |
| 基础 VideoSpy | Agent |  |  |  |  |  |  |  |  |  |  |
| 进化后 VideoSpy | Evolved Agent |  | ✓ |  |  |  |  |  |  |  |  |

<!-- [P0 主文核心表] 表 5b：EgoLifeQA SOTA 对比。现有方法必须在本文固定的 Day 6–7 测试集上重新运行。 -->

| 方法 | 方法类型 | 底座模型 | 使用 EgoLifeQA 训练标签 | 输入模态 | Token / 检索预算 | Overall | EntityLog | EventRecall | HabitInsight | RelationMap | TaskMaster |
|---|---|---|:---:|---|---|---:|---:|---:|---:|---:|---:|
| 现有代表性方法 |  |  |  |  |  |  |  |  |  |  |  |
| 基础 VideoSpy | Agent |  |  |  |  |  |  |  |  |  |  |
| 进化后 VideoSpy | Evolved Agent |  | ✓ |  |  |  |  |  |  |  |  |

<!-- [P0 主文核心表] 表 5c：MedVidBench SOTA 对比。所有结果采用隐藏测试集官方评测。 -->

| 方法 | 底座模型 | 使用医疗训练数据 | TAG@0.3 | TAG@0.5 | STG | CVS | NAP | SA | DVC F1 | DVC LLM | VS LLM | RC LLM |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 现有代表性方法 |  |  |  |  |  |  |  |  |  |  |  |  |
| 基础 VideoSpy |  |  |  |  |  |  |  |  |  |  |  |  |
| 进化后 VideoSpy |  | ✓ |  |  |  |  |  |  |  |  |  |  |

### 4.5 结果与泛化分析

首先分析进化前后的行为变化，包括学到的 Skill 和 Memory、工具选择、时间定位、采样密度，以及被修复和仍然存在的错误。典型案例同时展示问题、Ground Truth、进化前后答案、关键工具调用和相关经验，使性能变化能够对应到具体行为变化。

在主结果完成后，进一步研究领域专业化与泛化能力。将三个领域 Agent 交叉应用于三个领域的最终评测数据，分析本领域收益、跨领域退化和经验迁移；在较小底座模型上重新执行进化流程，并测试大模型学到的 Skill 或 Memory 能否迁移到小模型。最后分别统计离线进化成本和测试时推理成本。

<!-- [P1 主结果完成后补充] 图 4：进化前后的典型轨迹对比。以时间轴展示工具调用、视频定位和关键证据，说明具体 Skill 或 Memory 如何改变 Agent 行为。 -->

<!-- [P1 主结果完成后补充] 表 6：错误类型与行为变化。

| 领域 | 错误类型 | 基础 Agent 错误数 | 进化后错误数 | 修复比例 | 主要相关更新 | 代表案例 |
|---|---|---:|---:|---:|---|---|
| LVBench | 时间定位 / 证据遗漏 / 推理错误 |  |  |  |  |  |
| EgoLifeQA | 长时检索 / 身份识别 / 多事件关联 |  |  |  |  |  |
| MedVidBench | 定位 / 描述 / 专业判断 |  |  |  |  |  |
-->

<!-- [P2 扩展实验] 图 5：跨领域评价矩阵。行表示经验来源领域，列表示测试领域，以热力图展示领域专业化和跨领域迁移。 -->

<!-- [P2 扩展实验] 表 7：小模型泛化与经验迁移。

| 模型规模 | 进化方式 | Skill 来源 | Memory 来源 | LVBench | EgoLifeQA | MedVidBench | 相对基础提升 |
|---|---|---|---|---:|---:|---:|---:|
| 小模型 | 不进化 |  |  |  |  |  |  |
| 小模型 | 独立进化 | 小模型 | 小模型 |  |  |  |  |
| 小模型 | 经验迁移 | 大模型 |  |  |  |  |  |
| 小模型 | 经验迁移 | 大模型 | 大模型 |  |  |  |  |
-->

<!-- [P2 扩展实验] 图 6：性能与成本关系。横轴分别使用测试 Token、工具调用次数和时延，纵轴为任务性能，比较基础版本、各单路径版本和完整进化版本。 -->

## 5. 结论

本文从持续进化的角度研究视频智能体。核心思想是保持 VideoSpy 的 Agent 框架代码、Agent Loop、基础 System Prompt 模板、Prompt 组装机制、工具接口和模型权重固定，将答案评估和轨迹分析视为经验反传，再由 Agent Optimizer 将经验转化为程序性的任务 Skill、情景性的 Memory，以及 Agent 与工具超参数的更新。通过在通用长视频、第一视角生活视频和医疗手术视频上持续训练，本文将验证同一个通用 Agent 能否逐步形成多个稳定、有效且具有一定泛化能力的领域专业智能体。


# 附录

## A. 局限性与未来工作

本文聚焦于 Agent 层面的进化，在所有进化轮次中固定底座模型和工具内部模型的权重。因此，Agent 的能力上限仍然受到初始模型能力的影响。未来可以将成功轨迹和经过纠正的失败轨迹转化为训练数据，通过监督微调、偏好优化或强化学习更新模型权重，进一步研究 Agent 配置进化与模型参数学习的协同作用。

## B. 实验设置补充

### B.1 数据集任务与评价指标

**LVBench。** 103 个视频总计约 115.5 小时，平均时长 67.3 分钟，覆盖动画、纪录片、直播、自拍视频、体育和电视剧六类场景。问题考查实体识别、事件理解、关键信息检索、推理、摘要和时间定位六类能力。官方数据没有预定义划分，本文使用随机种子 42，在视频级别划分为训练集 83 个视频、1,234 题，以及测试集 20 个视频、315 题。评价指标为总体准确率和六类能力准确率，现有方法需要在相同测试集上重新运行。

**EgoLifeQA。** 本文使用公开的 A1_JAKE 500 题，包含约 51.9 小时视频，五类问题分别为 EntityLog、EventRecall、HabitInsight、RelationMap 和 TaskMaster。按照日期确定性划分后，Day 1–5 的 405 题作为训练集，Day 6–7 的 95 题作为测试集。评价指标为总体准确率、五类准确率和宏平均准确率，所有对比方法都需要在同一测试集上重新评测。

**MedVidBench。** 公开 Ground Truth 数据包含 6,270 个样本和 8 个来源数据集。本文以来源视频为单位，使用随机种子 42 生成互斥候选划分，并选择来源数据集与任务分布最接近全量数据的划分，得到训练集 5,146 条和本地测试集 1,124 条。测试集覆盖全部 8 个来源数据集和 11 个任务变体，各来源与任务的测试占比相对全量占比偏差不超过 20%。官方无标签 Leaderboard Test 共 6,245 条，只用于最终提交。任务包括时间动作定位、时空定位、密集视频描述、下一动作预测、Critical View of Safety 评估、视频摘要、区域描述和手术技能评估。评价采用 CVS_acc、NAP_acc、SA_acc、STG_mIoU、TAG_mIoU@0.3、TAG_mIoU@0.5、DVC_F1、DVC_llm、VS_llm 和 RC_llm。

### B.2 统一实现设置

<!-- [P0 附录核心表] 表 B1：数值在首次完整实验前锁定，后续所有方法保持一致。 -->

| 设置类别 | 具体配置 | 固定范围或说明 |
|---|---|---|
| 基础 Agent | 决策模型、最终回答模型 | 所有进化版本一致 |
| 视频工具 | overview、clip_skim、frame_inspect 的模型与接口 | 工具模型和接口固定 |
| 训练数据 | 每轮样本数 $B$、总训练样本数 | 同一 Benchmark 的不同方法一致 |
| 候选选择 | 每轮一个候选、接受阈值 $\epsilon$ | 同一消融实验一致 |
| 执行预算 | 最大 Agent 步数、最大工具调用数 | 给出统一上限 |
| 可进化超参数 | `max_steps`、三类工具的抽帧或帧数、片段时长与字幕上下文窗口 | 通用框架支持；LVBench Offline 验证阶段完全冻结 |
| 版本验证 | 验证样本数、重复次数、随机种子 | 所有候选使用同一设置 |
| 成本统计 | Token、工具调用、推理时延 | 分别统计进化阶段和测试阶段 |
