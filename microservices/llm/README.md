# LLM Microservice

LLM微服务提供了文本生成和聊天功能的API接口，支持多种LLM提供商。

## 功能

- 文本生成：根据提示生成文本
- 聊天：生成对话响应
- 健康检查：检查服务和提供商的可用性
- 提供商列表：获取可用的LLM提供商

## API接口

### 健康检查

```
GET /health
```

返回服务状态和各提供商的可用性。

**响应示例：**

```json
{
  "status": "ok",
  "providers": {
    "openai": true,
    "anthropic": true,
    "huggingface": false
  }
}
```

### 文本生成

```
POST /generate
```

根据提示生成文本。

**请求参数：**

```json
{
  "provider": "openai",
  "prompt": "Once upon a time",
  "model_name": "gpt-3.5-turbo-instruct",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**响应示例：**

```json
{
  "text": "Once upon a time in a faraway kingdom, there lived a brave knight who..."
}
```

### 聊天

```
POST /chat
```

生成对话响应。

**请求参数：**

```json
{
  "provider": "openai",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "model_name": "gpt-3.5-turbo",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**响应示例：**

```json
{
  "message": {
    "role": "assistant",
    "content": "I'm doing well, thank you for asking! How can I assist you today?"
  }
}
```

### 获取提供商列表

```
GET /providers
```

返回可用的LLM提供商列表。

**响应示例：**

```json
[
  "openai",
  "anthropic",
  "huggingface"
]
```

## 使用方式

### 启动服务

```bash
# 设置环境变量
export PORT=8001  # 可选，默认为8001

# 启动服务
python -m backend.microservices.llm.app
```

### 调用示例

```python
import requests

# 文本生成
response = requests.post(
    "http://localhost:8001/generate",
    json={
        "provider": "openai",
        "prompt": "Write a poem about AI",
        "model_name": "gpt-3.5-turbo-instruct",
        "max_tokens": 100,
    },
)
print(response.json())

# 聊天
response = requests.post(
    "http://localhost:8001/chat",
    json={
        "provider": "openai",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms"},
        ],
        "model_name": "gpt-3.5-turbo",
    },
)
print(response.json())
```

## 配置

服务配置通过环境变量进行设置：

- `PORT`：服务端口，默认为8001
- 各提供商的API密钥通过相应的环境变量设置，例如：
  - `OPENAI_API_KEY`：OpenAI API密钥
  - `ANTHROPIC_API_KEY`：Anthropic API密钥