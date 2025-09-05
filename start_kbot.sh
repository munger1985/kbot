#!/bin/bash

# 初始化 conda 环境
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate kbot3

# 定义微服务目录
MICROSERVICES_DIR="$(dirname "$0")/microservices"

# 启动四个微服务
echo "正在启动微服务..."

cd "${MICROSERVICES_DIR}/embedding" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/llm" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/reranker" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/vlm" && python app.py >/dev/null 2>&1 &
cd "${MICROSERVICES_DIR}/synonym" && python app.py >/dev/null 2>&1 &

for i in {1..10}; do
  echo -n "*"
  sleep 1
done
echo
echo "微服务已启动。"
# 启动 main.py
echo "正在启动 KBot 主程序..."
cd "$(dirname "$0")" && python main.py >/dev/null 2>&1 &
for i in {1..10}; do
  echo -n "*"
  sleep 1
done
echo
echo "所有服务已启动。"
