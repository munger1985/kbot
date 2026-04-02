#!/bin/bash

# Initialize conda environment
eval "$(conda shell.bash hook)"
conda activate kbot3

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
)

# Iterate and start all microservices
for service_name in "${!services[@]}"; do
    start_service "${service_name} Microservice" "$(dirname "${services[$service_name]}")" "$(basename "${services[$service_name]}")" "false" || exit 1
done

echo
echo "🎉 All services started successfully!"