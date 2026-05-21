#!/bin/bash

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

# ------------------------------

start_service() {
    local service_name=$1
    local directory=$2
    local script=$3
    local wait_for_ready=$4
    
    echo "正在启动 ${service_name}..."

    local python_exec=$(which python)
    
    cd "$directory" && python "$script" >/dev/null 2>&1 &
    local pid=$!
    
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        echo "✅ ${service_name} started successfully (PID: $pid)" 
        return 0
    else
        echo "❌ ${service_name} start failed!"
        return 1
    fi
}


# Function: start service and check status
start_service() {
    local service_name=$1
    local directory=$2
    local script=$3
    local wait_for_ready=$4

    echo "Starting ${service_name}..."

    # Switch to directory and start service, discard all output
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

# Start main program (and wait for full initialization)
start_service "KBOT3 Main Program" "$(dirname "$0")" "kbot_main.py" "true" || exit 1


# Microservices array
declare -A services=(
    ["Embedding"]="kbot_app_embedding.py"
    ["LLM"]="kbot_app_llm.py" 
    ["Reranker"]="kbot_app_reranker.py"
    ["VLM"]="kbot_app_vlm.py"
    ["Parser"]="kbot_app_parser.py"
    # ["MCP"]="kbot_mcp_server.py"
)

# Iterate and start all microservices
for service_name in "${!services[@]}"; do
    start_service "${service_name} Microservice" "$(dirname "${services[$service_name]}")" "$(basename "${services[$service_name]}")" "false" || exit 1
done

echo
echo "🎉 All KM services started successfully!"