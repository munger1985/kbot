# 嵌入服务部署指南

本文档提供了部署和运行嵌入服务的详细说明，包括本地开发环境、Docker容器和生产环境的部署方法。

## 目录

- [前提条件](#前提条件)
- [本地部署](#本地部署)
- [Docker部署](#docker部署)
- [生产环境配置](#生产环境配置)
- [扩展和负载均衡](#扩展和负载均衡)
- [监控和日志](#监控和日志)
- [故障排除](#故障排除)

## 前提条件

- Python 3.8+
- pip 包管理器
- Docker 和 Docker Compose (可选，用于容器化部署)
- 访问嵌入模型API的凭证

## 本地部署

### 1. 安装依赖

```bash
# 在项目根目录下
pip install -r backend/microservices/embedding/docker/requirements.txt
```

### 2. 配置环境变量

复制示例环境文件并根据需要修改：

```bash
cp backend/microservices/embedding/example.env .env
```

编辑 `.env` 文件，设置必要的配置参数，特别是模型API凭证。

### 3. 运行服务

```bash
# 方法1：直接运行微服务
python -m backend.microservices.embedding.main

# 方法2：通过主应用运行
python -m backend.main --service embedding
```

服务将在配置的主机和端口上启动（默认为 http://localhost:8000）。

### 4. 验证部署

使用curl或其他HTTP客户端测试服务是否正常运行：

```bash
# 检查健康状态
curl http://localhost:8000/api/embedding/health

# 测试嵌入生成
curl -X POST http://localhost:8000/api/embedding/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["测试文本"], "model_id": "text2vec"}'
```

## Docker部署

### 1. 构建和启动容器

```bash
# 在项目根目录下
cd backend/microservices/embedding/docker
docker-compose up -d
```

这将构建Docker镜像并启动容器。服务将在配置的端口上可用（默认为8000）。

### 2. 配置环境变量

可以通过以下方式设置环境变量：

- 在`docker-compose.yml`文件中直接编辑
- 创建`.env`文件（与`docker-compose.yml`位于同一目录）
- 使用Docker Compose的环境变量文件

示例`.env`文件：

```
TEXT2VEC_API_KEY=your_api_key_here
```

### 3. 验证部署

```bash
# 检查容器状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 测试API
curl http://localhost:8000/api/embedding/health
```

## 生产环境配置

### 安全配置

1. **API认证**：在生产环境中，应启用API认证：

   ```python
   # 在FastAPI应用中添加认证中间件
   from fastapi.security import APIKeyHeader
   
   api_key_header = APIKeyHeader(name="X-API-Key")
   
   @app.middleware("http")
   async def authenticate(request, call_next):
       if request.url.path.startswith("/api/"):
           api_key = request.headers.get("X-API-Key")
           if not api_key or api_key != os.getenv("API_SECRET_KEY"):
               return JSONResponse(
                   status_code=401,
                   content={"detail": "Invalid API Key"}
               )
       return await call_next(request)
   ```

2. **HTTPS**：使用反向代理（如Nginx）配置HTTPS：

   ```nginx
   server {
       listen 443 ssl;
       server_name embedding-api.example.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### 性能优化

1. **工作进程**：增加工作进程数以处理更多并发请求：

   ```
   EMBEDDING_WORKERS=4
   ```

2. **批处理参数**：根据负载调整批处理参数：

   ```
   EMBEDDING_MAX_BATCH_SIZE=128
   EMBEDDING_MAX_WAIT_TIME=0.05
   ```

3. **内存限制**：在Docker环境中设置内存限制：

   ```yaml
   services:
     embedding-service:
       # ...其他配置...
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
   ```

## 扩展和负载均衡

### 水平扩展

使用Docker Swarm或Kubernetes进行水平扩展：

#### Docker Swarm

```bash
# 初始化Swarm
docker swarm init

# 部署服务
docker stack deploy -c docker-compose.yml embedding-stack

# 扩展服务
docker service scale embedding-stack_embedding-service=3
```

#### Kubernetes

创建Kubernetes部署配置（`deployment.yaml`）：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding-service
        image: your-registry/embedding-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: EMBEDDING_HOST
          value: "0.0.0.0"
        # 其他环境变量...
---
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
spec:
  selector:
    app: embedding-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

应用配置：

```bash
kubectl apply -f deployment.yaml
```

### 负载均衡

使用Nginx作为负载均衡器：

```nginx
upstream embedding_servers {
    server embedding1:8000;
    server embedding2:8000;
    server embedding3:8000;
}

server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://embedding_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 监控和日志

### Prometheus监控

1. 添加Prometheus客户端支持：

```python
from prometheus_client import Counter, Histogram, start_http_server

# 定义指标
embedding_requests = Counter('embedding_requests_total', 'Total embedding requests', ['model_id'])
embedding_latency = Histogram('embedding_latency_seconds', 'Embedding request latency', ['model_id'])

# 在应用启动时启动Prometheus HTTP服务器
start_http_server(9090)
```

2. 配置Prometheus抓取目标：

```yaml
scrape_configs:
  - job_name: 'embedding-service'
    scrape_interval: 15s
    static_configs:
      - targets: ['embedding-service:9090']
```

### 日志配置

使用结构化日志：

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, 'request_id'):
            log_record["request_id"] = record.request_id
        return json.dumps(log_record)

# 配置日志
logger = logging.getLogger("embedding_service")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### 日志聚合

使用ELK栈（Elasticsearch, Logstash, Kibana）或类似工具聚合和分析日志。

## 故障排除

### 常见问题

1. **服务无法启动**
   - 检查端口是否被占用
   - 验证环境变量是否正确设置
   - 检查Python版本是否兼容

2. **模型加载失败**
   - 验证API凭证是否正确
   - 检查网络连接是否正常
   - 查看详细错误日志

3. **性能问题**
   - 检查系统资源使用情况
   - 调整批处理参数
   - 考虑增加工作进程或水平扩展

### 诊断命令

```bash
# 检查服务日志
docker-compose logs -f embedding-service

# 检查系统资源
docker stats

# 进入容器进行调试
docker-compose exec embedding-service bash
```

### 联系支持

如果遇到无法解决的问题，请联系技术支持团队：

- 提交GitHub Issue: [项目Issues页面](https://github.com/your-org/your-repo/issues)
- 发送邮件至: support@example.com