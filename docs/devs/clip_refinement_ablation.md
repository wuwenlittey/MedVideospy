# VideoSpy：细节与时间补看实验

本轮仅修改基础 `videospy-main` 的可选观察流程，不启用检索、RSI 或训练。
默认关闭，原始 `clip_skim.py`、默认配置和三个工具的公开接口保持不变。

## 四种运行方式

在 `videospy-main` 根目录，给原本的推理/benchmark 命令追加配置即可。
先传自己的服务配置，再传实验配置。不要复用同一输出目录。

| 模式 | 追加参数 | 首次成功 clip_skim 后发生什么 |
|---|---|---|
| 原版 | 不加实验配置，或最后加 `--config config/videospy/refinement_off.yml` | 无补看 |
| 细节 | `--config config/videospy/refinement_detail.yml` | 重看原采样中的 4 帧，使用原始分辨率 |
| 时间 | `--config config/videospy/refinement_temporal.yml` | 在原采样间隙补 4 帧，短边仍为 256 |
| 同时开启 | `--config config/videospy/refinement_both.yml` | 一次补看调用中提供 4 张细节帧和 4 张新时间帧 |

也可以依次叠加 detail 与 temporal 两个配置，效果等同于 both。
detail 文件只开启细节开关，temporal 文件只开启时间开关，**不会清除先前配置的另一项**。
需要明确关闭两项时，请将 off 配置放在最后。

完整单题示例（视频、问题及服务配置替换为实际值）：

```bash
python -m videospy.cli --video_path /path/to/video.mp4 \
  --user_query "Your medical video question" \
  --config /path/to/model.yml \
  --config config/videospy/refinement_detail.yml \
  --output_dir output/refinement_detail
```

已有接受重复 `--config` 的 benchmark 入口也可使用同样覆盖文件，无需新增 CLI 参数。
本地没有 MedHorizon runner，因此此处不虚构其运行命令；如果服务器 runner 已接受并合并 Agent YAML，
给它追加同样配置即可。仍应检查实际生效的配置与逐题审计日志。

## 触发与预算

先执行完整原始粗看，再自动追加观察，Agent 不需要主动申请补看。
默认仅补看每题**第一次成功执行的 clip_skim**，不是每次调用都追加；
仅 overview 或 frame_inspect 的题目不会触发。失败、被拦截的粗看不会消耗这个触发额度。
本功能不会修正时间转换，也不会判断选定区间是否正确。建议先在正确定位的样本上验证。

额外观察使用 **clip_skim 相同的视觉模型**，不调用 frame_inspect 的不同模型或提示词。
补看共用固定观察提示词和原始问题，不接收参考答案，也不接收“上一版回答错误”的提示。
原始粗看观察和补看观察分别保留，再交给原 Agent 继续推理。
本轮不改变后续自主调用和最终回答流程，不是固定的一次补看后强制停止的回放实验。

补看最多增加一次视觉模型调用；同时开启时为同一次调用输入两组共最多 8 张图片。
这不额外占用主 Agent 的工具决策轮数，因此比较成本时必须看实际模型调用/token，不能只看 total_steps。
若服务调用失败，保留成功粗看，在审计中记录失败；不偷偷回退到另一模型，也不重新执行粗看。
ModelClient 自身配置的 API 重试仍有效，每次尝试保留在 model_calls。

## 帧选择与控制变量

- 细节帧：将原始采样序列等分，取各分组中心位置，默认选 4 个；
  只重读相同帧 ID，不添加新时刻，不做 ROI、不做生成式增强、不人工放大低分辨率源图。
- 时间帧：对原始相邻采样帧的 ID 求中点，从有新帧的间隙中等距选取，默认 4 个；
  新帧 ID 与原采样不重合，并保留在原观察区间内，仍使用原粗看短边 256 的预处理。
- 实际可选帧不足时使用已有数量；没有间隙新帧则记录 skipped，不制造重复补帧。
- 补看字幕固定为首次粗看所见字幕，时间补帧不会意外增加字幕证据。
- 两组共用补看提示词；同时开启时显式标明两组来源，不把它们当成一段连续视频。

第一版通过独立运行实现消融，不保证不同运行的服务响应完全相同。
应核对每组首次粗看的帧 ID 和区间；如果它们不同，不能称为同一状态下的严格配对回放。

## 配置项

```yaml
tools:
  clip_skim:
    refinement:
      high_resolution: false
      temporal_fill: false
      high_resolution_frames: 4
      temporal_frames: 4
      high_resolution_short_side: null
      max_clips_per_question: 1
```

`high_resolution_short_side: null` 保留源图尺寸。
设置正整数可限制短边，但不会将源图放大；例如 512 表示大源图缩到短边 512，小源图保持原样。
两个帧数和每题额度均须为正整数；使用布尔开关关闭功能。

若原图短边不大于 256，细节分支没有真正增加空间细节，日志会明确标记。
客户端只能记录发送尺寸，无法保证服务端视觉处理器没有进一步缩图；请核查服务器的图像 token/像素配置。

另提供 `refinement_reobserve_control.yml`：与细节分支取完全相同的帧，但短边限制为 256。
它用于区分“再次观察/生成”和“更高分辨率”的收益；源图短边低于 256 的样本不适合该对照。

## 日志与评测

正常轨迹 JSON 中：

- `diagnostics.clip_refinement.settings`：实际配置；
- `diagnostics.clip_refinement.inspections`：每次触发的区间、原采样 ID、两组补看帧 ID/时间、
  源尺寸、发送尺寸、源像素 SHA256、completed/skipped/failed 状态；
- 对应工具消息的 `clip_refinement`：同样的审计记录；
- `model_call`：保留旧字段，兼容旧读取代码；
- `model_calls`：有多次模型尝试/调用时，记录全部（包含 model_call，**不要重复累加**）。

审计元数据保存在轨迹中，不作为多余上下文传给主 Agent。
补看失败、未触发和源图不够清晰应单独统计，不能都视为“高分辨率无效”。
建议记录原错→对、原对→错、净修复及完整成本，包含原本答对的同类样本。
本实现仅验证机制及开关，不包含真实数据性能结论。

## 回退

最简单回退：移除实验覆盖文件，或将 `refinement_off.yml` 放在最后；不需要重建视频索引。
本地是源码归档，没有 Git 元数据，改动前的 agent.py / README.md / clip_skim.py 已备份至：
`../artifacts/clip_refinement_baseline_20261018_164724/`。
原始 clip_skim.py 最终未修改；另附 manifest、差异补丁和哈希校验回退脚本。
源码恢复脚本默认只检查，显式 `--apply` 才恢复，并拒绝覆盖发生后续修改的文件。

## 离线检查

```bash
python -m unittest tests.test_clip_refinement -v
```

测试使用合成帧和 mock 模型，不读取真实视频、不调用真实服务。

本次新增 12 项测试通过。原有 `tests.test_videospy` 共 73 项测试中有 5 个失败项，
与修改前备份复跑的失败项完全相同（预算提示、overview 失败重试、三个字幕窗口默认值断言）。
对比记录在备份目录的 `base_test_comparison.json`，未将这些既有差异纳入本次修复。
