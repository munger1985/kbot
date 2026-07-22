#!/bin/bash

# Define service root directory
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

# Function: Safely get PID of specific Python script running in specified directory
get_service_pid() {
    local script_dir="$1"
    local script_name="$2"
    
    # Use pgrep to find processes and pwdx to check working directory
    pgrep -f "python.*${script_name}" | grep -v "$CURRENT_PID" | while read pid; do
        if [ -d "/proc/$pid" ]; then
            if pwdx "$pid" 2>/dev/null | grep -q "${script_dir}"; then
                echo "$pid"
            fi
        fi
    done
}

# Collect PIDs of processes to be terminated
declare -A PID_MAP

# 遍历所有服务获取PID
for service_script in "${SERVICES[@]}"; do
    SERVICE_PIDS=$(get_service_pid "${SERVICE_ROOT}" "${service_script}")
    if [ -n "$SERVICE_PIDS" ]; then
        while read pid; do
            if [ -n "$pid" ]; then
                PID_MAP["$pid"]=1
                echo "Found ${service_script} process: $pid"
            fi
        done <<< "$SERVICE_PIDS"
    fi
done

# Check if there are processes to terminate
if [ ${#PID_MAP[@]} -eq 0 ]; then
    echo "No running KBot services found."
    exit 0
fi

# Convert deduplicated PIDs to space-separated string
PIDS="${!PID_MAP[@]}"

echo "About to terminate the following processes: $PIDS"

# 如果需要确认，取消注释下面的代码
# read -p "Confirm termination of these services? (y/n): " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Operation cancelled."
#     exit 0
# fi

# Send SIGTERM signal for graceful shutdown
for PID in $PIDS; do
    if kill -0 $PID 2>/dev/null; then
        kill -SIGTERM $PID
        echo "Sent graceful shutdown signal (SIGTERM) to process $PID."
    else
        echo "Process $PID no longer exists, skipping."
    fi
done

# Wait for processes to exit gracefully (max 10 seconds)
TIMEOUT=10
for PID in $PIDS; do
    COUNT=0
    while kill -0 $PID 2>/dev/null && [ $COUNT -lt $TIMEOUT ]; do
        sleep 1
        COUNT=$((COUNT + 1))
    done
    
    # Check if process is still running
    if kill -0 $PID 2>/dev/null; then
        echo "Process $PID did not exit within ${TIMEOUT} seconds, will force terminate (SIGKILL)."
        kill -SIGKILL $PID 2>/dev/null || echo "Failed to force terminate process $PID."
    else
        echo "Process $PID exited normally."
    fi
done

echo "All KBot services have been shut down completely."
