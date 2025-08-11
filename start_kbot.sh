#!/bin/bash

# 激活 conda 环境
source /home/chris/miniconda3/bin/activate kbot

# 启动四个微服务
cd /home/chris/kbot/microservices/embedding && python app.py &
cd /home/chris/kbot/microservices/llm && python app.py &
cd /home/chris/kbot/microservices/reranker && python app.py &
cd /home/chris/kbot/microservices/vlm && python app.py &

# 启动 main.py
cd /home/chris/kbot && python main.py &

echo "所有服务已启动。"