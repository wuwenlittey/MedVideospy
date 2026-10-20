# 医学视觉案例记忆：第一版

这版将经核验的历史画面及正确语义答案作为参考，注入现有 `clip_skim` / `frame_inspect` 的同一次视觉模型调用。
默认关闭；不加载本地嵌入模型，不做 RSI，不修改当前视频的定位策略，也不自动向记忆库写入新预测。

可比较三组：无记忆、仅参考文字、参考图文。启用文件同时关闭上一版补看及旧的文本经验检索，便于分别验证。
当前帧的采样、VLM 模型及原提示保留；图文组会增加历史图片和上下文，必须一起记录 token、延迟和改错率。

## 1. 对接已有 WeMM-Embedding-2B 服务

编辑 `config/videospy/visual_memory_wemm.yml`，`model` 与服务器实际服务名一致。
默认 `protocol: vllm_chat`，以当前环境变量指定端点：

```bash
export WEMM_EMBEDDING_API_BASE=http://127.0.0.1:8001/v1
# 有鉴权时取消 YAML 中 api_key_env 注释，再设置环境变量：
# export WEMM_EMBEDDING_API_KEY=...
python -m videospy.visual_memory.cli probe \
  --embedding-config config/videospy/visual_memory_wemm.yml \
  --image /path/to/a/source_frame.png
```

PowerShell 使用 `$env:WEMM_EMBEDDING_API_BASE='http://127.0.0.1:8001/v1'`。
`probe` 仅发送一张图到嵌入服务，输出维度、范数、安全的协议指纹，不调用问答模型。

请求是 `POST <api_base>/embeddings`，每图一请求：

```json
{"model":"WeMM-Embedding-2B","encoding_format":"float","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}
```

响应默认读取 `data[i].embedding`。建库和查询必须使用同一个端点、模型、协议、模板及维度；不匹配会拒绝注入并记入审计。
默认检验 2048 维；如服务返回别的维度，核实服务配置后修改 YAML 并重新建库。
模型文件、服务进程由服务器管理，本代码不下载或自动启动模型。
协议依据：[WeMM 官方服务说明](https://github.com/Tencent/WeMM-Embedding)、[vLLM 图像嵌入示例](https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_online.py)。

若服务器是其他封装，`protocol: custom` 支持 `request_template` / `image_item_template` / `per_image` / `response_path` / `vector_key`。
模板替换只处理完整 JSON 值：`{model}`、`{images}`、`{image_urls}`，单图还支持 `{image_url}`。例如：

```yaml
tools:
  visual_memory:
    embedding:
      protocol: custom
      url: http://127.0.0.1:8001/your/image/embedding
      model: WeMM-Embedding-2B
      per_image: true
      request_template:
        model: "{model}"
        image: "{image_url}"
      response_path: data
      vector_key: embedding
      dimension: 2048
```

先通过 `probe` 再建库；HTTP 输入格式不统一，模型名字本身不能保证封装协议一致。

## 2. 从轨迹导出待审核案例

创建 `selections.jsonl`，每行明确指定一个已经成功执行的局部观察。答对和答错轨迹都可以选，但必须显式提供 GT。
成功执行工具只说明有观察结果，不代表模型的描述或答案正确。

```json
{"trajectory_path":"/runs/train/trajectories/1.json","tool_call_id":"call_abc","video_path":"/videos/train/case01.mp4","source_video_id":"case01","source_sample_id":"qa_1","task_type":"phase_recognition","split":"train","gold_option":"C","options":{"A":"Phase A","B":"Phase B","C":"Preparation","D":"Phase D"}}
```

也可以用 `answer_text: "Preparation"` 直接提供完整语义答案。两种写法并存时必须一致。
`tool_call_id` 可替换为从零开始的 `tool_message_index`，指向 `messages` 中工具的结果消息。
`source_video_id` 必须是原视频身份，不能用 QA ID 或裁剪片段 ID 替代。可选填写 `source_id`、`patient_id`、`procedure`。

若源视频在服务器以图片序列保存，用有序的 `frame_paths` 和 `fps` 替换 `video_path`：

```json
{"trajectory_path":"/runs/train/trajectories/2.json","tool_message_index":3,"frame_root":"/datasets/frames","frame_paths":["case02/000.png","case02/001.png","case02/002.png"],"fps":1.0,"source_video_id":"case02","task_type":"instrument_identification","split":"train","answer_text":"Atrial retractor"}
```

序列帧必须与原 runner 的顺序和 FPS 一致；带框任务可传与原 reader 相同的 `rc_info`。
输入路径相对于 selections 文件；`frame_paths` 相对于显式 `frame_root`。
有跨机器路径前缀时显式提供 `frame_path_prefix`，不会猜测或搜索其他磁盘。
旧轨迹没有模型输入记录时，需要提供历史 `num_frames` / `max_frames`，不会拿现在默认值冒充当时参数。

```bash
python -m videospy.visual_memory.cli curate \
  --selections /path/to/selections.jsonl --output output/visual_memory/candidates_v1
python -m videospy.visual_memory.cli validate \
  --cases output/visual_memory/candidates_v1/cases.jsonl
```

输出原始 RGB 采样帧、`cases.jsonl` 和来源报告；一次观察只产生一条案例，标签不广播到每帧。
原帧保留源分辨率，时间按历史工具的 floor/nearest、去重及边界规则还原；这不是历史缩图输入的像素级复刻。
每条包含轨迹、调用、采样位置、源图片及导出图片的 SHA256。失败产物会标注不完整，不能建库。

## 3. 审核后建库

逐条核对画面与标签的对应关系，必要时缩小证据时间范围、同步删改 `timestamps` / `image_paths`，或剔除无依据案例。
`question` 应仍然适用于所保留的画面。单帧用 `evidence_scope: frame`；有序多帧用 `clip`。
确认整组证据支持该问题的标签后，填写 `verified: true`、非空 `verification_note`。
正确选项不能自动证明每帧的可见事实，原错误轨迹的描述也不会自动被认定为纠正后的事实。

```bash
python -m videospy.visual_memory.cli validate \
  --cases output/visual_memory/candidates_v1/cases.jsonl --require-verified
python -m videospy.visual_memory.cli build \
  --cases output/visual_memory/candidates_v1/cases.jsonl \
  --output output/visual_memory/index_v1 \
  --embedding-config config/videospy/visual_memory_wemm.yml
```

保留待审核的候选时，另存一份只含已核验案例的 JSONL，且保留图片相对路径有效。
仅接收 `split: train/dev`；评测按独立视频/患者划分，不将旧测试答案换个 split 名称后用于同批评测。
源 JSONL、导出目录及索引都拒绝覆盖，换版本目录建新库。
索引由 `records.jsonl`、`vectors.npy`、`media/`、带完整性哈希的 `manifest.json` 组成，不需要 FAISS 或数据库服务。

当前向量是每帧独立编码、逐帧归一化后平均、再归一化；**不是 WeMM 原生视频嵌入，不编码动作顺序**。
图文参考保留案例原顺序，动作相反但外观相似仍是第一版的已知检索局限。
离线默认每例最多 16 帧，可用 `embedding.max_images_per_case` 调整，最大 64。

## 4. 启用与对照

在原推理命令上追加覆盖文件，先服务配置、后实验配置。每组使用独立结果目录。

| 组别 | 最后追加的配置 |
|---|---|
| 无记忆 | `--config config/videospy/visual_memory_off.yml` |
| 仅文字参考 | `--config config/videospy/visual_memory_wemm.yml --config config/videospy/visual_memory_text.yml` |
| 图文参考 | `--config config/videospy/visual_memory_wemm.yml --config config/videospy/visual_memory_image_text.yml` |

所有开启模式都从当前实际采样原图做视觉检索；仅文字组不向 VLM 提供参考图片。
默认 Top-1、每题最多 4 次检索，查询最多取当前已采样的 16 帧，不增加当前视频的新时刻。
`min_similarity: 0.0` 只是实验起点，不代表校准后的可信阈值；应在开发集调阈值并看命中率。

图文模式完整显示被审核的案例图组，默认每例上限 16 帧；超过 `reference_frames` 的案例显式跳过，**不会删掉关键帧后仍沿用整段标签**。
若希望参考最多 4 帧，需要审核时先精选并确认这 4 帧足以支持该问题，再设 `reference_frames: 4`。
参考图默认短边最多 256，不上采样。关键细节任务应核实这个分辨率是否仍足够，或提高 `reference_short_side`。

运行期必须提供与库相同的 `task_type`，并记录真实 `source_video_id`。MedVidBench 原 runner 会提供任务与视频上下文。
独立 CLI 或服务器自定义 runner 可以通过 YAML 指定单一任务类型；多任务 runner 应逐题传入 `_runtime.task_type`，不要固定成错误类型。

```yaml
tools:
  visual_memory:
    context:
      task_type: phase_recognition
      # 可选：procedure: ...
```

`source_video_id` 优先读取运行时 memory_context 的 video/source_video_id，再取显式 context，最后取视频文件名 stem。
跨机器必须统一该身份；若文件名不能唯一识别原视频，请显式提供。缺少有效任务身份会跳过检索并记录原因。
同视频、相同非空 source_id/patient_id/source_sample_id 被排除，任务类型严格相同；procedure 双方都有值时必须相同。
每个来源视频最多取一个案例。阈值不合适、类型不匹配或超参考预算时允许零参考。

单题示例，最后的 task.yml 填上上面的任务类型：

```bash
python -m videospy.cli --video_path /videos/eval/case99.mp4 \
  --user_query 'Which phase is visible in this interval?' \
  --config /path/to/vlm.yml \
  --config config/videospy/visual_memory_wemm.yml \
  --config config/videospy/visual_memory_image_text.yml \
  --config /path/to/task.yml --output_dir output/memory_image_text
```

本地没有 MedHorizon 专用 runner，不虚构它的命令；只要服务器 runner 使用本仓库 VideoSpyAgent 并合并 YAML，就能使用相同配置。
嵌入失败默认 `fail_open: true`，保留原观察；接口调试阶段可设 false，让错误显式成为工具失败。
每个被查询的帧至少一次嵌入 HTTP 请求，重试另外计数；图文上下文可能明显增加 VLM 成本。

## 5. 审计、验证与回退

工具消息的 `visual_memory` 和轨迹 `diagnostics.visual_memory.inspections` 包含：
当前查询帧时间/像素哈希、索引哈希、嵌入身份、检索分数、来源与案例 ID、参考图片哈希/尺寸、实际注入状态、排除与失败原因、请求内容指纹及嵌入请求次数/耗时。
密钥不入审计；旁路审计字段不会发送给主 Agent。`model_call` 保留实际 VLM 输入（默认图像 base64 省略）。

评测同时记录原错→对、原对→错、净变化、各任务准确率、命中率及成本。先小规模比较精选案例与自动检索，定位问题在案例质量、检索还是模型利用示例的能力。
当前离线测试使用合成图片、临时索引和假服务；尚无真实医疗数据增益结论，也未连接服务器实测。

```bash
python -m unittest tests.test_visual_memory_store tests.test_visual_memory_curation tests.test_visual_memory_integration tests.test_clip_refinement -v
```

关闭：移除新配置，或将 `visual_memory_off.yml` 放到最后。无需删除原视频或修改已有记忆库。
源码修改前的备份及测试记录：`../artifacts/visual_memory_baseline_20261020_192808/`。
备份目录提供文件哈希及只恢复本次改动的 `restore_visual_memory.py`；默认检查，显式 `--apply` 才执行，后续有改动会拒绝覆盖。
