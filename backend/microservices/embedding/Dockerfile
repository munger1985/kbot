# 使用 Python 3.9 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 安装微服务依赖项
RUN pip install --no-cache-dir -r backend/microservices/embedding/requirements.txt

# 设置 Python 路径
ENV PYTHONPATH=/app

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=8000

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "backend/microservices/embedding/app.py"]