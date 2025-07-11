# 嵌入微服务

这是一个提供文本嵌入功能的微服务，可以将文本转换为向量表示。

## 功能

- 将文本列表转换为嵌入向量
- 支持多种嵌入模型
- 提供健康检查端点
- 提供可用模型列表端点

## 安装和运行

### 使用 Docker Compose（推荐）

1. 确保已安装 Docker 和 Docker Compose
2. 在项目根目录下运行以下命令：

```bash
cd backend/microservices/embedding
docker-compose up -d
```

### 直接运行

1. 确保已安装 Python 3.9 或更高版本
2. 安装依赖项：

```bash
pip install -r backend/microservices/embedding/requirements.txt
```

3. 运行微服务：

```bash
python backend/microservices/embedding/app.py
```

## 依赖项

主要依赖项包括：

- fastapi: Web 框架
- uvicorn: ASGI 服务器
- pydantic: 数据验证
- numpy: 数学计算
- torch: 深度学习框架
- transformers: 自然语言处理模型
- sentence-transformers: 文本嵌入模型
- sqlalchemy: 数据库 ORM
- asyncpg: 异步 PostgreSQL 客户端

## API 端点

### 1. 文本嵌入

将文本列表转换为嵌入向量。

- **URL**: `/embed`
- **方法**: `POST`
- **请求体**:
  ```json
  {
    "model_id": 1,
    "texts": ["string"],
    "batch_size": 32
  }
  ```
- **响应**:
  ```json
  {
    "embeddings": [[0.1, 0.2, ...]],
    "model_id": 1,
    "dimensions": 0
  }
  ```

### 2. 获取可用模型列表

获取可用的嵌入模型列表。

- **URL**: `/models`
- **方法**: `GET`
- **响应**:
  ```json
  {
    "models": [1, 2, 3]
  }
  ```

### 3. 健康检查

检查服务是否正常运行。

- **URL**: `/health`
- **方法**: `GET`
- **响应**:
  ```json
  {
    "status": "healthy",
    "service": "embedding-service"
  }
  ```

## 示例

### 嵌入文本

```bash
curl -X POST "http://localhost:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{
           "model_id": 1,
           "texts": ["这是一个测试文本", "这是另一个测试文本"]
         }'
```

### 获取可用模型列表

```bash
curl -X GET "http://localhost:8000/models"
```

### 健康检查

```bash
curl -X GET "http://localhost:8000/health"
```

## 注意事项

- 模型 ID 是整数类型，需要从数据库中获取有效的模型 ID
- 嵌入服务在应用启动时初始化，在应用关闭时关闭
- 嵌入向量的维度取决于所使用的模型