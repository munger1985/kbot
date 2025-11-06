#!/bin/bash

# 初始化 conda 环境
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate kbot3

# 启动 main.py
echo "正在启动 KBot 主程序..."

cd "$(dirname "$0")" && python kbot_main.py >/dev/null 2>&1 &

for i in {1..10}; do
  echo -n "*"
  sleep 1
done
echo
echo "Kbot 主程序已启动。"

# 定义微服务目录
MICROSERVICES_DIR="$(dirname "$0")/microservices"

# 启动四个微服务
echo "正在启动微服务..."

# 启动 Embedding 微服务
cd "${MICROSERVICES_DIR}/embedding" && python app.py >/dev/null 2>&1 &
for i in {1..3}; do
  echo -n "*"
  sleep 1
done
echo
echo "Embedding 微服务已启动。"

# 启动 LLM 微服务
cd "${MICROSERVICES_DIR}/llm" && python app.py >/dev/null 2>&1 &
echo "LLM 微服务已启动。"

# 启动 Reranker 微服务
cd "${MICROSERVICES_DIR}/reranker" && python app.py >/dev/null 2>&1 &
for i in {1..3}; do
  echo -n "*"
  sleep 1
done
echo "Reranker 微服务已启动。"

# 启动 VLM 微服务
cd "${MICROSERVICES_DIR}/vlm" && python app.py >/dev/null 2>&1 &
echo "VLM 微服务已启动。"
echo
echo "微服务已启动。"
echo "所有服务已启动。"
