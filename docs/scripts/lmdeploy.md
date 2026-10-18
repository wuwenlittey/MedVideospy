## 本地模型部署和验证

### 模型服务部署

本文基于 LMDeploy 0.14.0。Qwen3.5 和 Qwen3-VL 都是多模态模型，下面显式使用 PyTorch 后端，以保留图片和视频输入能力。

```bash
conda activate llm_deploy

# 或者重新安装环境。
# python -m pip install "lmdeploy==0.14.0"
```

根据需要选择一个模型启动服务。两个命令使用相同端口，不能同时运行。

```bash
# Qwen3.5 系列，Qwen3.8-27B
lmdeploy serve api_server \
  /myworkspace/models/Qwen/Qwen3.5-4B \
  --backend pytorch \
  --server-name 0.0.0.0 \
  --server-port 23333 \
  --model-name Qwen3.5-4B \
  --max_batch_size 8 \
  --cache-max-entry-count 0.8 \
  --trust-remote-code \
  --reasoning-parser default \
  --tool-call-parser qwen3coder \
  --log-level REQUEST

# Qwen3-VL 系列
lmdeploy serve api_server \
  /myworkspace/models/Qwen/Qwen3-VL-4B-Instruct \
  --backend pytorch \
  --server-name 0.0.0.0 \
  --server-port 23333 \
  --model-name Qwen3-VL-4B-Instruct \
  --cache-max-entry-count 0.8 \
  --trust-remote-code \
  --tool-call-parser qwen3 \
  --log-level REQUEST
```

如果 Qwen3.5 只处理文本，可以增加 `--language-model-only`，避免加载视觉编码器并节省显存。出现显存不足时，可以将 `--cache-max-entry-count` 降到 `0.6` 或 `0.7`。

### 模型服务测试

测试调用（OpenAI 兼容接口）：

```bash
# Qwen3.5 系列
curl http://127.0.0.1:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen3.5-4B",
    "messages":[{"role":"user","content":"你好"}],
    "max_completion_tokens":200
  }'

curl https://member.aicloud.szu.edu.cn:11443/inf-app/a21068095973355520664329/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen3.5-4B",
    "messages":[{"role":"user","content":"你好"}],
    "max_completion_tokens":200
  }'

curl https://member.aicloud.szu.edu.cn:11443/inf-app/a21161580353548288613269/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen3.5-9B",
    "messages":[{"role":"user","content":"你好"}],
    "max_completion_tokens":200
  }'

# 智算中心内网访问用这个地址（比如提交训练任务）： http://qwen38x27b:23333/v1/chat/completions
# 外网访问地址： https://member.aicloud.szu.edu.cn:11443/inf-app/a21161560921862144655310/v1/chat/completions
curl https://member.aicloud.szu.edu.cn:11443/inf-app/a21161560921862144655310/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen3.8-27B",
    "messages":[{"role":"user","content":"你好"}],
    "max_completion_tokens":200
  }'


# Qwen3-VL-4B-Instruct 图片输入
curl http://127.0.0.1:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen3-VL-4B-Instruct",
    "messages":[{
      "role":"user",
      "content":[
        {"type":"text","text":"请描述这张图片"},
        {
          "type":"image_url",
          "image_url":{
            "url":"https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
          }
        }
      ]
    }],
    "max_completion_tokens":200
  }'
```

### 模型配置

```yaml
model_name: "openai/Qwen3-VL-4B-Instruct"
api_base: "http://127.0.0.1:23333/v1"
api_key: "dummy"
```

## Todos

智算中心部署模型，暴露访问地址 @黄国林
