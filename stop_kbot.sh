#!/bin/bash

# 服务根目录
SERVICE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CURRENT_PID=$$

# 统一服务列表（与启动脚本保持一致）
SERVICES=(
    "apps/ai_models_embedding/main.py"
    "apps/ai_models_llm/main.py"
    "apps/ai_models_vlm/main.py"
    "apps/ai_models_visual/main.py"
    "apps/knowledge_core_api/main.py"
    "apps/knowledge_core_parser/main.py"
    "apps/knowledge_core_projection/main.py"
)

# 安全查找在指定目录运行的 Python 服务进程
get_service_pid() {
    local script_dir="$1"
    local script_name="$2"
    
    # 先按脚本名查找，再通过工作目录限制进程范围
    pgrep -f "python.*${script_name}" | grep -v "$CURRENT_PID" | while read pid; do
        if [ -d "/proc/$pid" ]; then
            if pwdx "$pid" 2>/dev/null | grep -q "${script_dir}"; then
                echo "$pid"
            fi
        fi
    done
}

# 收集待停止的进程号
declare -A PID_MAP

# 遍历所有服务获取PID
for service_script in "${SERVICES[@]}"; do
    SERVICE_PIDS=$(get_service_pid "${SERVICE_ROOT}" "${service_script}")
    if [ -n "$SERVICE_PIDS" ]; then
        while read pid; do
            if [ -n "$pid" ]; then
                PID_MAP["$pid"]=1
                echo "发现 ${service_script} 进程：$pid"
            fi
        done <<< "$SERVICE_PIDS"
    fi
done

# 没有可停止的进程时直接退出
if [ ${#PID_MAP[@]} -eq 0 ]; then
    echo "未发现正在运行的 KBot 服务。"
    exit 0
fi

# 将去重后的进程号转换为空格分隔列表
PIDS="${!PID_MAP[@]}"

echo "即将停止以下进程：$PIDS"

# 如果需要确认，取消注释下面的代码
# read -p "确认停止这些服务吗？(y/n): " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "操作已取消。"
#     exit 0
# fi

# 发送 SIGTERM 以触发优雅退出
for PID in $PIDS; do
    if kill -0 $PID 2>/dev/null; then
        kill -SIGTERM $PID
        echo "已向进程 $PID 发送优雅退出信号（SIGTERM）。"
    else
        echo "进程 $PID 已不存在，跳过。"
    fi
done

# 最多等待 10 秒让进程完成优雅退出
TIMEOUT=10
for PID in $PIDS; do
    COUNT=0
    while kill -0 $PID 2>/dev/null && [ $COUNT -lt $TIMEOUT ]; do
        sleep 1
        COUNT=$((COUNT + 1))
    done
    
    # 超时后再次检查进程状态
    if kill -0 $PID 2>/dev/null; then
        echo "进程 $PID 在 ${TIMEOUT} 秒内未退出，将发送 SIGKILL。"
        kill -SIGKILL $PID 2>/dev/null || echo "无法强制停止进程 $PID。"
    else
        echo "进程 $PID 已正常退出。"
    fi
done

echo "全部 KBot 服务已停止。"
