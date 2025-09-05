#!/bin/bash

# 设置循环次数
COUNT=100

# 循环执行curl命令
for ((i=1; i<=COUNT; i++))
do
  echo "执行第 $i 次请求..."
  
  # 执行curl命令
  curl -X POST "http://localhost:8000/api/agent/chat" \
    -H "Content-Type: application/json" \
    -d '{"session_id": "session_1754010982.804123", "by":"chris", "agent_id":2, "security_level":3, "request_time":"2025-08-01 18:01:01", "question": "黎曼几何是什么?"}'
  
  echo -e "\n"
  
  # 可选：添加延迟以防止服务器过载
  sleep 0.1
done

echo "已完成 $COUNT 次请求"