FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV EMBEDDING_HOST=0.0.0.0
ENV EMBEDDING_PORT=8000
ENV EMBEDDING_WORKERS=1
ENV EMBEDDING_LOG_LEVEL=info

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "backend.microservices.embedding.main"]