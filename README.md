# 多模态视频理解智能体

本项目包含两个长视频理解 Agent。VideoSeek 是来自原论文的 baseline 实现，VideoSpy 是我们在此基础上开发的方法。两者共享 benchmark 和基础依赖，但代码、配置与文档彼此独立。

## 方法概览

| | VideoSeek | VideoSpy |
|---|---|---|
| 定位 | 外部 baseline | 本项目开发的方法 |
| 工具 | `overview`、`skim`、`focus`、`answer` | `overview`、`clip_skim`、`frame_inspect` |
| Agent 交互 | 自定义动作与观察协议 | 原生函数调用与标准工具消息 |
| 轨迹 | 自定义轨迹结构 | 完整消息、工具调用及内部模型调用 |

## 项目结构

```text
.
├── videoseek/              # VideoSeek Agent、工具和 CLI
├── videospy/               # VideoSpy Agent、工具和 CLI
├── config/
│   ├── videoseek/          # VideoSeek 配置与提示词
│   └── videospy/           # VideoSpy 配置与提示词
├── benchmark/
│   ├── lvbench/            # LVBench 运行与评测
│   └── medvidbench/        # MedVidBench 运行与评测
├── data/                   # 数据集与第三方资源
├── output/                 # Agent 和 benchmark 输出
├── tests/                  # 单元测试
└── docs/                   # 图片等文档资源
```

## 文档

- [VideoSeek](videoseek/README.md)
- [VideoSpy](videospy/README.md)
- [VideoSpy 双层 RSI（独立实现）](videospy_rsi/README.md)
- [VideoSpy-Evolve 实验](experiments/README.md)
- [VideoSpy-Evolve 开发设计](docs/research/evolving_video_agent/dev_plan.md)
- [LVBench](benchmark/lvbench/README.md)
- [MedVidBench](benchmark/medvidbench/README.md)
- [StreamEQA](benchmark/streameqa/README.md)
- [StreamingBench](benchmark/streamingbench/README.md)

## 安装

要求 Python 3.10 及以上版本，并确保系统已安装 `ffmpeg`。

```bash
pip install -e .
```

安装后可使用：

```bash
videoseek-cli -h
videospy-cli -h
videospy-rsi -h
```

## 测试

```bash
python -m pytest tests -q
```

## License

[MIT](https://opensource.org/licenses/MIT)

## 医学视频观察消融版本

本版本包含独立、默认关闭的高分辨率补看和时间间隙补帧。
在原命令末尾追加下列配置之一，分别使用独立输出目录：

- 细节：`--config config/videospy/refinement_detail.yml`
- 时间：`--config config/videospy/refinement_temporal.yml`
- 同时开启：`--config config/videospy/refinement_both.yml`
- 关闭：最后追加 `--config config/videospy/refinement_off.yml`

详见 [实验说明](docs/devs/clip_refinement_ablation.md)。
新增 12 项离线机制测试通过；尚未运行真实医学视频效果评测。
本次发布保留基础源码、配置、测试、文档与许可证，不包含数据、模型权重、运行输出或真实 API Key。
