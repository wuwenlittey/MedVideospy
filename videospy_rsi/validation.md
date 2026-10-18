# 本地验证记录

2026-10-09，未调用真实模型服务，未运行真实视频性能实验，未上传 GitHub。

## RSI 验证

```powershell
python -m unittest tests.test_rsi -v
```

在已有完整 VideoSpy 环境中，23 项行为测试全部通过（27.636 秒）。覆盖候选边界、模型固定、源码装载、数据源划分、标签视图、类别回归、未知 usage、全部重试记录、最终调用超额保护、版本快照防篡改、精确输入 checkpoint、两层接受/拒绝、Meta 分支不晋升正式 T、最终测试不回流、完成后恢复与中途提交前恢复。新增包和测试的 `compileall` 也通过。

`BaselineParityTest` 需要完整基础 VideoSpy 环境；只有 NumPy/Pillow 的轻量环境会跳过该项。在已有 `D:/miniforge/envs/videospy` 环境中该项通过：T0 与基础 Agent 使用相同 reader、配置和模拟响应时，决策模型请求、工具模型请求以及最终回答协议一致。

demo 使用合成帧和模拟模型，可验证控制逻辑与档案，不能证明真实任务上的性能收益。默认流程应得到六个正式 Task 轮次、一个 Meta 更新，以及可选的冻结最终测试报告。

最终完整演示已成功运行，档案位于 `output/rsi/demo-20261009-final/`。状态为 `complete`，六个正式 Task 轮次与一次 Meta 更新均完成；冻结测试的执行和 usage 记录完整。该目录被仓库已有 `output` 忽略规则排除，不会作为代码一同上传。

## 基础版独立回归

执行 `python -m unittest tests.test_videospy tests.test_evolved_memory_injection -q` 共 78 项测试，出现 5 个失败断言：

- `test_budget_decreases_after_failed_and_duplicate_inspections`：测试期待轮次提示，但基础 Agent 的当前问题消息没有该提示。
- `test_failed_tool_request_can_be_retried`：模拟的第二次执行未返回成功。
- `test_visual_tools_default_to_sampled_frame_subtitles`：三个子测试期待 overview/clip_skim/frame_inspect 字幕窗口为 60/120/30 秒，当前基础默认配置均为 5 秒。

这两个基础测试模块及其执行路径没有引入 `videospy_rsi`。本次未修改 `videospy/` 或 `config/videospy/` 来解决这些差异，保留与原自进化实现的隔离。仓库原有测试全部通过这一条件尚未满足。

## 首版边界

- 候选检查为合作式结构限制，独立子进程不等于恶意代码安全沙箱。
- 记录实际帧索引/时间、原图 shape、各模型图像输入尺寸和哈希；不保存完整 base64，也没有逐裁剪区域的可视化溯源界面。
- 支持固定 `option`/`exact` QA 指标和 LVBench 转换，未声称接入全部医学视频指标。
- 正式训练与性能比较需要使用视频源互斥的真实数据、可用的固定模型配置和预先确定的预算。
