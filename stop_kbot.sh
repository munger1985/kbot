#!/bin/bash

# 定义服务根目录（根据您的启动脚本调整）
SERVICE_ROOT="$(dirname "$0")"

# 函数：安全获取指定目录下特定Python脚本的进程PID
get_service_pid() {
    local script_dir="$1"
    local script_name="$2"
    
    # 使用pgrep查找进程，并结合pwdx检查进程的工作目录是否匹配
    # 这确保了只杀死在指定目录下运行的特定脚本
    local pids=""
    pgrep -f "python.*${script_name}" | while read pid; do
        # 检查进程的工作目录是否在服务根目录下
        if pwdx "$pid" 2>/dev/null | grep -q "${script_dir}"; then
            echo "$pid"
        fi
    done
}

# 收集需要关闭的进程PID
declare -A PID_MAP  # 使用关联数组来去重

# 获取主程序PID（确保只在当前项目目录下运行的那个）
MAIN_PIDS=$(get_service_pid "${SERVICE_ROOT}" "kbot_main.py")
if [ -n "$MAIN_PIDS" ]; then
    while read pid; do
        if [ -n "$pid" ]; then
            PID_MAP["$pid"]=1
            echo "找到主程序进程: $pid"
        fi
    done <<< "$MAIN_PIDS"
fi

# 获取各个微服务的PID（现在都在项目根目录下）
MICROSERVICES=(
    "kbot_app_embedding.py"
    "kbot_app_llm.py" 
    "kbot_app_vlm.py"
    "kbot_app_reranker.py"
    "kbot_app_parser.py"
)

for service_script in "${MICROSERVICES[@]}"; do
    SERVICE_PIDS=$(get_service_pid "${SERVICE_ROOT}" "${service_script}")
    if [ -n "$SERVICE_PIDS" ]; then
        while read pid; do
            if [ -n "$pid" ]; then
                PID_MAP["$pid"]=1
                echo "找到${service_script}微服务进程: $pid"
            fi
        done <<< "$SERVICE_PIDS"
    fi
done

# 检查是否有进程需要关闭
if [ ${#PID_MAP[@]} -eq 0 ]; then
    echo "没有找到运行中的KBot服务。"
    exit 0
fi

# 将去重后的PID转换为空格分隔的字符串
PIDS="${!PID_MAP[@]}"

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