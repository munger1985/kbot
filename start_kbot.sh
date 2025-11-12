#!/bin/bash

# 初始化 conda 环境
eval "$(conda shell.bash hook)"
conda activate kbot3

# 使用 /tmp 目录存储启动日志
LOG_DIR="/tmp/kbot_startup_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 函数：等待主程序完全启动
wait_for_main_program() {
    local service_name=$1
    local pid=$2
    local max_attempts=30
    local wait_seconds=2
    
    echo "等待 ${service_name} 完全启动..."
    
    for ((i=1; i<=max_attempts; i++)); do
        # 检查进程是否仍在运行
        if ! kill -0 $pid 2>/dev/null; then
            echo "❌ ${service_name} 进程已退出"
            return 1
        fi
        
        # 检查日志中是否有启动成功的标志（根据您的实际日志调整）
        local log_file="${LOG_DIR}/${service_name}_${TIMESTAMP}.log"
        if grep -q "启动成功\|启动完成\|ready\|started" "$log_file" 2>/dev/null; then
            echo "✅ ${service_name} 已完全启动"
            return 0
        fi
        
        if [ $i -eq 1 ]; then
            echo "正在等待 ${service_name} 初始化..."
        elif [ $((i % 5)) -eq 0 ]; then
            echo "已等待 $((i * wait_seconds)) 秒，${service_name} 仍在初始化..."
        fi
        
        sleep $wait_seconds
    done
    
    echo "⚠️ ${service_name} 在 $((max_attempts * wait_seconds)) 秒内未显示完全启动，但进程仍在运行"
    return 0
}

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
        
        # 如果需要等待完全就绪
        if [ "$wait_for_ready" = "true" ]; then
            wait_for_main_program "$service_name" "$pid" || return 1
        fi
        
        return 0
    else
        echo "❌ ${service_name} 启动失败！错误信息："
        cat "$log_file"
        return 1
    fi
}

# 启动主程序（并等待其完全启动）
start_service "KBot主程序" "$(dirname "$0")" "kbot_main.py" "true" || exit 1


# 启动微服务数组
declare -A services=(
    ["Embedding"]="kbot_app_embedding.py"
    ["LLM"]="kbot_app_llm.py" 
    ["Reranker"]="kbot_app_reranker.py"
    ["VLM"]="kbot_app_vlm.py"
)

# 遍历启动所有微服务
for service_name in "${!services[@]}"; do
    start_service "${service_name}微服务" "$(dirname "${services[$service_name]}")" "$(basename "${services[$service_name]}")" "false" || exit 1
done

echo
echo "🎉 所有服务已成功启动！"
echo "📋 本次启动日志位置: $LOG_DIR"