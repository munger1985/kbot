#!/bin/bash

# 初始化 conda 环境
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate kbot3

# 定义微服务目录
MICROSERVICES_DIR="$(dirname "$0")/microservices"

# 启动四个微服务
cd "${MICROSERVICES_DIR}/embedding" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/llm" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/reranker" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/vlm" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/synonym" && python app.py >/dev/null 2>&1 &

# 启动 main.py
cd "$(dirname "$0")" && python main.py >/dev/null 2>&1 &

echo "所有服务已启动。"
