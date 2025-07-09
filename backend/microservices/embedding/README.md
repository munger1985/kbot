# 嵌入服务 (Embedding Service)

## 概述

嵌入服务是一个高性能的文本嵌入生成微服务，支持多种嵌入模型，提供批处理、自动健康检查和模型生命周期管理功能。该服务可以作为独立微服务运行，也可以集成到更大的应用程序中。

## 功能特点

- **多模型支持**：可同时支持多种嵌入模型，如text2vec、BGE、M3E等
- **动态模型加载**：按需加载模型，减少资源占用
- **批处理优化**：自动批处理请求，提高吞吐量
- **健康监控**：定期检查模型健康状态，自动恢复异常模型
- **资源管理**：自动卸载长时间未使用的模型，优化资源使用
- **RESTful API**：提供简洁易用的API接口
- **可扩展性**：易于添加新的模型实现

## 安装与配置

### 依赖项

- Python 3.8+
- FastAPI
- Uvicorn
- NumPy
- psutil

### 配置

1. 复制示例配置文件并根据需要修改：

```bash
cp example.env .env
```

2. 编辑`.env`文件，配置服务参数和模型参数

### 启动服务

```bash
python -m backend.microservices.embedding.main
```

或者通过应用程序入口点启动：

```bash
python -m backend.main --service embedding
```

## API 接口

### 生成嵌入向量

```
POST /api/embedding/embed
```

请求体：
```json
{
  "texts": ["这是第一段文本", "这是第二段文本"],
  "model_id": "text2vec"
}
```

响应：
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "model_id": "text2vec",
  "dimensions": 768
}
```

### 更新模型配置

```
PUT /api/embedding/models/{model_id}/config
```

请求体：
```json
{
  "config": {
    "api_url": "https://new-api-endpoint.com/embed",
    "api_key": "new_api_key",
    "dimensions": 768
  },
  "version": "1.0"
}
```

响应：
```json
{
  "updated": true,
  "model_id": "text2vec"
}
```

### 健康检查

```
GET /api/embedding/health
```

响应：
```json
{
  "status": "healthy",
  "timestamp": "2023-06-01T12:00:00Z",
  "models": {
    "text2vec": {
      "status": "connected",
      "last_used": "2023-06-01T11:55:00Z"
    },
    "bge": {
      "status": "connected",
      "last_used": "2023-06-01T11:50:00Z"
    }
  }
}
```

### 获取统计信息

```
GET /api/embedding/stats
```

响应：
```json
{
  "models": {
    "text2vec": {
      "request_count": 1250,
      "last_used": 1685620800
    },
    "bge": {
      "request_count": 850,
      "last_used": 1685620500
    }
  },
  "instance_id": "550e8400-e29b-41d4-a716-446655440000",
  "cpu_usage": 25.5,
  "memory_usage": 15.2
}
```

## 模型配置

### 通过环境变量配置

可以通过环境变量为每个模型配置参数，格式为：

```
EMBEDDING_MODEL_<MODEL_ID>_<PARAM_NAME>=<VALUE>
```

例如：

```
EMBEDDING_MODEL_TEXT2VEC_API_URL=https://api.example.com/embedding
EMBEDDING_MODEL_TEXT2VEC_API_KEY=your_api_key_here
```

### 常用模型参数

| 参数名 | 描述 | 默认值 |
|--------|------|--------|
| api_url | API端点URL | - |
| api_key | API密钥 | - |
| dimensions | 嵌入向量维度 | 模型相关 |
| max_tokens | 最大token数 | 512 |
| timeout | 请求超时时间(秒) | 30 |
| headers | 额外的HTTP头 | {} |

## 性能优化

1. **调整批处理参数**：
   - 增大`max_batch_size`可提高吞吐量，但可能增加延迟
   - 减小`max_wait_time`可降低延迟，但可能降低吞吐量

2. **资源管理**：
   - 调整`max_idle_time`控制模型卸载时机
   - 增加工作进程数(`workers`)以处理更多并发请求

3. **健康检查**：
   - 调整`health_check_interval`以平衡监控频率和性能开销

## 扩展模型支持

要添加新的模型实现，需要在`backend/services/embedding/models`目录下创建新的模型类，并实现`EmbeddingModel`接口。

## 故障排除

1. **模型加载失败**：
   - 检查API密钥和URL是否正确
   - 确认网络连接是否正常

2. **性能问题**：
   - 检查批处理参数是否合理
   - 监控系统资源使用情况

3. **健康检查失败**：
   - 查看日志了解具体错误
   - 检查模型服务是否可用