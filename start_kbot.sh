#!/bin/bash

# Conda 环境配置
CONDA_BIN_PATH=$(which conda 2>/dev/null)
if [ -z "$CONDA_BIN_PATH" ]; then
    POSSIBLE_PATHS=("$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" "/opt/miniconda3/bin/conda")
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$path" ]; then
            CONDA_BIN_PATH="$path"
            break
        fi
    done
fi

if [ -z "$CONDA_BIN_PATH" ]; then
    echo "❌ Error: Conda command not found. Please ensure conda is installed."
    exit 1
fi

CONDA_ROOT=$(dirname "$(dirname "$CONDA_BIN_PATH")")
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate kbot3

# 统一服务列表：格式 "服务名:脚本路径:启动目录"
# 注意：脚本路径支持相对路径和绝对路径
SERVICES=(
    "主程序:kbot_main.py:."
    "Embedding:kbot_app_embedding.py:."
    "LLM:kbot_app_llm.py:."
    "VLM:kbot_app_vlm.py:."
    "Visual:kbot_app_visual.py:."
    "Parser:kbot_app_parser.py:."
    "MCP:kbot_mcp_server.py:."
    "DB Executor:kbot_db_executor.py:."
)

# 启动服务函数
start_service() {
    local service_name="$1"
    local script="$2"
    local directory="$3"
    
    echo "🚀 Starting ${service_name}..."
    cd "$directory" && python "$script" >/dev/null 2>&1 &
    local pid=$!
    
    sleep 1
    if kill -0 $pid 2>/dev/null; then
        echo "✅ ${service_name} started successfully (PID: $pid)"
        return 0
    else
        echo "❌ Failed to start ${service_name}!"
        return 1
    fi
}

# 启动所有服务
echo "Starting all KBot services..."
for service in "${SERVICES[@]}"; do
    IFS=':' read -r name script dir <<< "$service"
    start_service "$name" "$script" "$dir" || exit 1
done

echo
echo "🎉 All KBot services started successfully!"