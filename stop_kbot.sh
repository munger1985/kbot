#!/bin/bash

# 定义服务根目录（根据您的启动脚本调整）
SERVICE_ROOT="$(dirname "$0")"
MICROSERVICES_DIR="${SERVICE_ROOT}/microservices"

# 函数：安全获取指定目录下特定Python脚本的进程PID
get_service_pid() {
    local script_dir="$1"
    local script_name="$2"
    
    # 使用pgrep查找进程，并结合pwdx检查进程的工作目录是否匹配
    # 这确保了只杀死在指定目录下运行的特定脚本
    pgrep -f "python.*${script_name}" | while read pid; do
        # 检查进程的工作目录是否在服务根目录下
        if pwdx "$pid" 2>/dev/null | grep -q "${SERVICE_ROOT}"; then
            echo "$pid"
        fi
    done
}

# 收集需要关闭的进程PID
PIDS=""

# 获取主程序PID（确保只在当前项目目录下运行的那个）
MAIN_PID=$(get_service_pid "${SERVICE_ROOT}" "kbot_main.py")
if [ -n "$MAIN_PID" ]; then
    PIDS+=" $MAIN_PID"
    echo "找到主程序进程: $MAIN_PID"
fi

# 获取各个微服务的PID
for service_dir in "embedding" "llm" "reranker" "vlm"; do
    SERVICE_PID=$(get_service_pid "${MICROSERVICES_DIR}/${service_dir}" "app.py")
    if [ -n "$SERVICE_PID" ]; then
        PIDS+=" $SERVICE_PID"
        echo "找到${service_dir}微服务进程: $SERVICE_PID"
    fi
done

# 检查是否有进程需要关闭
if [ -z "$PIDS" ]; then
    echo "没有找到运行中的KBot服务。"
    exit 0
fi

echo "即将关闭以下进程: $PIDS"
# read -p "确认关闭这些服务? (y/n): " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "操作已取消。"
#     exit 0
# fi

# 发送SIGTERM信号优雅关闭
for PID in $PIDS; do
    if kill -0 $PID 2>/dev/null; then
        kill -SIGTERM $PID
        echo "已向进程 $PID 发送优雅关闭信号(SIGTERM)。"
    else
        echo "进程 $PID 已不存在，跳过。"
    fi
done

# 等待进程优雅退出（最长10秒）
TIMEOUT=10
for PID in $PIDS; do
    COUNT=0
    while kill -0 $PID 2>/dev/null && [ $COUNT -lt $TIMEOUT ]; do
        sleep 1
        COUNT=$((COUNT + 1))
    done
    
    # 检查进程是否仍在运行
    if kill -0 $PID 2>/dev/null; then
        echo "进程 $PID 未在 ${TIMEOUT} 秒内退出，将强制终止(SIGKILL)。"
        kill -SIGKILL $PID 2>/dev/null || echo "强制终止进程 $PID 失败。"
    else
        echo "进程 $PID 已正常退出。"
    fi
done

echo "所有KBot服务关闭完成。"