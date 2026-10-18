# VideoSpy 双层 RSI

本实现与 `videospy/evolving/` 并存。基础 Agent、三个工具和已有 Skill/Memory 进化代码保持原样；新循环位于 `videospy_rsi/`，配置位于 `config/videospy_rsi/`，运行结果写入指定的独立目录。它复用基础 Agent 的函数调用、证据收集和最终回答协议，通过每个 worker 自己的工具 registry 装载精确候选版本，不修改基础模块的全局 registry，不读写旧进化数据库。

## 进化对象

| 层 | 可以进化 | 固定部分 |
|---|---|---|
| Task（T） | `overview.py`、`clip_skim.py`、`frame_inspect.py` 的工具实现；`runtime_policy.py` 的策略提示和工具选择顺序；`evidence_policy.py` 的工具证据文本整理；白名单采样参数、步数；显式 Skill 文本 | 工具名称和参数接口、Agent 回答协议、模型角色、评价器、数据划分、预算与权限 |
| Meta（M） | `diagnoser.py: diagnose(payload, request)` 和 `task_proposer.py: propose(payload, request)` 的诊断、调查请求、提案生成方法及提示词 | Task 验证器、选择规则、外层 G、Meta 验证器、模型和预算 |

Task 源码候选提交完整的文件替换内容；执行参数使用点分路径，例如 `tools.overview.num_frames`。每个候选必须携带精确 `parent_id` 和可解释的 `hypothesis`，无变化候选会被拒绝。源码、配置和 Skill 一起形成内容哈希版本。

初始 T0 复制基础三个工具，替换模型调用入口为固定角色网关；策略和证据处理初始为恒等行为。T0 使用 64/16/4 个帧的基础采样配置，最多 16 个 Agent 步骤，Memory 关闭。初始 M0 通过固定网关请求诊断和 Task 候选。模型权重不训练。

## 完整循环与信息流

1. 正式 T 在一个诊断批次上运行。运行时只收到问题、视频/帧路径、字幕、任务类型和合法时长等字段，收不到答案。
2. 固定评价器计算答案是否正确，并评估轨迹的证据充分性（0–5）。记录所有决策、工具内部请求、最终回答、重试、实际采样帧、模型图像输入大小与哈希。
3. M 的 diagnoser 接收当前 T、诊断案例、正确性、参考答案、证据评分和历史尝试。它返回假设、反例、未知项，以及可选的定向工具调查。调查只使用诊断数据；工具输入不带标签。
4. M 的 proposer 接收诊断、调查观察、案例、当前 T 源码/配置、历史尝试和固定限制，生成最多两个 Task 候选。候选经过结构/接口检查、诊断集局部执行、固定 gate 验证。只对结构、接口或局部执行错误提供最多一次修复机会。
5. gate 比较父 T 与候选的质量 Q、每问题推理 token 成本 C 和任务类别回归。默认接受规则是质量提高至少 0.001 且成本增长不超过 20%，或者质量不下降且成本减少至少 5%；任一任务类别下降都拒绝。正式 T 只在完整验证、成本已知且预算有效时改变。
6. 每三个正式 Task 轮次触发一次 Meta 更新；拒绝和无候选的轮次也计数。固定 G 接收当前 M/T、最近六个正式 Task 尝试的有界视图、历史 Meta 聚合反馈与固定限制，生成最多一个 M 候选。G 看不到 Meta episode 的样本、标签或报告轨迹。
7. 对每个 episode，旧 M 和新 M 从相同 T、空分支历史、同一数据/种子和预算开始独立 Task 搜索。每个分支最多两轮，仅在 probe 上诊断、在 inner_gate 上选择 T，随后在 report 上评估起点 T 和最终 T。分支不会触发 Meta 更新。
8. 固定 Meta 验证器聚合配对 report 的最终质量和推理成本，按同一 Q/C 规则选择 M。搜索成本、起点与增益另行记录。接受新 M 只更新正式 M；分支 T 不会成为正式 T。后续正式批次仍从原正式 T 出发，使用新 M。
9. 达到正式轮数或预算后冻结版本。可选 final test 只评价冻结 T，其结果不进入任何生成器输入。

默认最多六个正式 Task 轮次。达到第六轮后直接冻结，因此默认在第三轮之后进行一次 Meta 更新；末尾不做没有后续 Task 的更新。扩大实验时在独立配置中修改轮数、批次和预算。

## 离线验证

在仓库根目录执行：

```powershell
python -m videospy_rsi demo --output output/rsi/demo --final-test
python -m unittest tests.test_rsi -v
```

demo 只需要 NumPy 和 Pillow。子进程使用合成静态帧与确定性模拟响应，不连接模型服务。其 token 数被标记为 `synthetic`，只能验证循环行为，不能作为视频任务性能或真实成本结果。它会演示 T 接受/拒绝、M 接受、后续 T 使用新 M，以及冻结测试。

已有项目安装后也可使用 `videospy-rsi`；未重新安装时直接用 `python -m videospy_rsi` 即可。

## 真实视频运行

使用已有 VideoSpy 环境及项目依赖。默认配置从环境变量读取服务设置：

```powershell
$env:RSI_MODEL = 'your-provider/your-vision-model'
$env:RSI_API_BASE = 'https://your-service/v1'
$env:RSI_API_KEY = 'your-key'
python -m videospy_rsi run --dataset datasets/rsi.json --output output/rsi/experiment-01 --final-test
```

可以通过 `--config path/to/rsi.json` 覆盖默认 JSON。配置深度合并，七个模型角色分别是 `task`、`overview`、`clip_skim`、`frame_inspect`、`assessor`、`meta`、`outer`。每个角色在一次 run 内固定；凭证不写入版本和公开配置。可将某个模型字段写为 `env:VARIABLE_NAME`。原有 `config/videospy/default.yml` 中的服务凭证不会被 RSI 加载。

同样的命令和输出目录可恢复兼容的运行。配置、数据或实现指纹变化时必须使用新的输出目录；不能拿旧评分验证新代码。完整阶段使用 checkpoint 复用；被打断且未完成的模型调用可能重跑，已经记入账本的调用仍计成本。

推理时加载正式接受的 Task 版本：

```powershell
python -m videospy_rsi infer --run-dir output/rsi/experiment-01 --video-path video.mp4 --question 'Question with options' --output output/rsi/prediction.json
```

推理角色配置应与训练 run 一致；如训练使用了独立配置，推理也传同一个 `--config`。可用 `--task-version task-...` 指定档案中的版本。

## 数据格式与划分

第一版提供 LVBench 转换器和归一化 QA 接口，固定支持 `option`（单个大写选项）与 `exact`（小写化、合并空白后精确匹配）。这不是 MedVidBench 全套官方指标适配。

归一化数据的结构为：

```json
{
  "diagnostic_batches": [["sample objects"]],
  "gate": ["sample objects"],
  "episodes": [
    {"id": "initial", "start": "initial", "seed": 42,
     "probe": ["sample objects"], "inner_gate": ["sample objects"], "report": ["sample objects"]},
    {"id": "current", "start": "current", "seed": 43,
     "probe": ["sample objects"], "inner_gate": ["sample objects"], "report": ["sample objects"]}
  ],
  "test": ["sample objects"]
}
```

上面的字符串表示待替换的样本对象，不是可直接运行的数据。每个对象至少包含：

```json
{"id": "qa-001", "source_id": "video-001", "video_path": "videos/video-001.mp4",
 "question": "What happens?\nA. ...\nB. ...", "answer": "A", "metric": "option", "task_type": "action"}
```

也可使用 `frame_paths` 数组与 `fps` 表示帧序列。资源路径相对于数据 JSON 解析；`duration` 可限制可见时长。所有来自同一个视频的 QA 必须使用同一个 `source_id`。诊断、gate、Meta 和 test 按视频源互斥；每个 episode 内 probe/inner_gate/report 也互斥。至少提供两个 episode。划分校验基于显式 source_id，使用者必须确保它真实标识视频来源。

LVBench 可使用显式 UID 划分清单转换：

```powershell
python -m videospy_rsi prepare-lvbench --metadata path/to/video_info.meta.jsonl --video-dir path/to/videos --splits path/to/splits.json --output datasets/rsi.json
```

`splits.json` 与上述结构相同，各样本对象替换为 UID；同一个视频的 UID 不能跨数据作用域。请把转换结果写到自有目录；本实现不修改 `data/` 中的第三方数据。

## 档案与边界

```text
output/rsi/<run>/
  run.json, settings.json     配置/数据/实现指纹与公开设置
  state.json                  原子更新的正式 T/M 指针及轮次
  versions/task/<id>/         Task 源码、配置、Skill 与 manifest
  versions/meta/<id>/         Meta 源码与 manifest
  phases/                     精确输入哈希对应的阶段结果
  candidates/                 候选、源码 diff
  task_rounds/, meta_rounds/   完整尝试、生成输入、验证与拒绝原因
  records/                    每样本运行轨迹与调用引用
  calls.jsonl                 所有阶段逐请求事件，包括重试和失败
  workers/                    独立 worker 工作目录
  final/versions.json         冻结的正式版本
  final/test.json             可选最终测试聚合结果
```

全局搜索成本包含诊断推理、评价、调查、生成、局部/gate 验证和 Meta 分支；最终测试单独计数。episode 的搜索 token 配额不含 report 推理，但 report 仍属于全局预算。每问题 C 只统计该次 QA 的决策、工具内部与最终回答调用。未知 usage 不视为零成本，会阻止接受并停止搜索。最后一次调用超额也不能晋升版本。

候选使用 AST/import 白名单、受限 Reader、固定模型入口和独立进程执行。这些是合作式研究候选的结构边界，**不构成操作系统级的恶意 Python 沙箱**。进程继承运行环境，不能直接用它执行不可信第三方代码。定向调查已实现，同前缀多动作反事实分支尚未实现。首版不支持新工具名称、模型训练或在线事实 Memory 增长。

验证记录与基础测试现状见 [validation.md](validation.md)。流程规格见 [双层工作流](../docs/research/videospy_dual_layer_rsi_workflow.md)。
