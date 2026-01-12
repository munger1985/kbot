#!/bin/bash

# 初始化 conda 环境
eval "$(conda shell.bash hook)"
conda activate kbot3

# 使用 /tmp 目录存储启动日志
LOG_DIR="/tmp/kbot3_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 函数：启动服务并检查状态
start_service() {
    local service_name=$1
    local directory=$2
    local script=$3
    local wait_for_ready=$4
    local log_file="${LOG_DIR}/${service_name}_${TIMESTAMP}.log"
    
    echo "正在启动 ${service_name}..."
    
    # 切换到目录并启动服务
    cd "$directory" && python "$script" >"$log_file" 2>&1 &
    local pid=$!
    
    sleep 1
    if kill -0 $pid 2>/dev/null; then
        echo "✅ ${service_name} 已启动 (PID: $pid)"
        
        return 0
    else
        echo "❌ ${service_name} 启动失败！错误信息："
        cat "$log_file"
        return 1
    fi
}

# 启动主程序（并等待其完全启动）
start_service "Nexus Cube 主程序" "$(dirname "$0")" "kbot_main.py" "true" || exit 1


# 启动微服务数组
declare -A services=(
    ["Embedding"]="kbot_app_embedding.py"
    ["LLM"]="kbot_app_llm.py" 
    ["Reranker"]="kbot_app_reranker.py"
    ["VLM"]="kbot_app_vlm.py"
    ["Parser"]="kbot_app_parser.py"
)

# 遍历启动所有微服务
for service_name in "${!services[@]}"; do
    start_service "${service_name}微服务" "$(dirname "${services[$service_name]}")" "$(basename "${services[$service_name]}")" "false" || exit 1
done

echo
echo "🎉 所有服务已成功启动！"
echo "📋 本次启动日志位置: $LOG_DIR"
