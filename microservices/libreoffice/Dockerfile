FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libuno-cppu3 \
    libuno-cppuhelpergcc3-3 \
    libuno-sal3 \
    libuno-salhelpergcc3-3 \
    uno-libs3 \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app /app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:${API_PORT}", "--workers", "4", "main:app"]