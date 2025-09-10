#!/bin/bash

# 获取所有微服务和 main.py 的进程ID
PIDS=$(pgrep -f "python app.py")
PIDS+=" $(pgrep -f "python kbot_main.py")"

if [ -z "$PIDS" ]; then
    echo "没有找到运行中的服务。"
    exit 0
fi

# 发送 SIGTERM 信号优雅关闭
for PID in $PIDS; do
    kill -SIGTERM $PID
    echo "已发送关闭信号到进程 $PID"
done

# 等待最多 10 秒，确保服务退出
TIMEOUT=10
for PID in $PIDS; do
    while kill -0 $PID 2>/dev/null; do
        if [ $TIMEOUT -le 0 ]; then
            echo "进程 $PID 未在超时时间内退出，强制终止。"
            kill -9 $PID
            break
        fi
        sleep 1
        TIMEOUT=$((TIMEOUT - 1))
    done
done

echo "所有服务已优雅关闭。"