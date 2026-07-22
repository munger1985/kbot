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

# 日志目录（服务自身的 loguru 日志配置由各服务控制，这里只存启动 stderr）
STARTUP_LOG_DIR="logs/startup"
mkdir -p "$STARTUP_LOG_DIR"

# 统一服务列表：格式 "服务名:脚本路径:启动目录"
# 注意：脚本路径支持相对路径和绝对路径
SERVICES=(
    "Embedding:apps/ai_models_embedding/main.py:."
    "LLM:apps/ai_models_llm/main.py:."
    "VLM:apps/ai_models_vlm/main.py:."
    "Visual:apps/ai_models_visual/main.py:."
    "Knowledge Core:apps/knowledge_core_api/main.py:."
    "KC Index Worker:apps/knowledge_core_projection/main.py:."
    "Parser:apps/knowledge_core_parser/main.py:."
)

# 启动服务函数
start_service() {
    local service_name="$1"
    local script="$2"
    local directory="$3"
    local safe_script="${script//\//_}"
    local log_file="${STARTUP_LOG_DIR}/${safe_script%.py}.log"
    
    echo "🚀 Starting ${service_name}..."

    # 清空上次的启动日志，便于阅读
    : > "$log_file"

    # stdout → /dev/null（抑制 uvicorn access 日志）
    # stderr → 启动日志文件（捕获 EADDRINUSE 等错误）
    cd "$directory" && python "$script" >/dev/null 2>>"$log_file" &
    local pid=$!

    # Background workers do not expose an HTTP port; verify the process stays
    # alive briefly and then let the launcher continue.
    if [ "$script" = "apps/knowledge_core_projection/main.py" ]; then
        sleep 1
        if ! kill -0 $pid 2>/dev/null; then
            echo "  ❌ ${service_name} worker exited during startup"
            [ -s "$log_file" ] && sed 's/^/    | /' "$log_file"
            return 1
        fi
        echo "✅ ${service_name} worker started（PID: $pid）"
        return 0
    fi

    # 重试循环：最多等待 15 秒，每 2 秒检查一次 PID
    local waited=0
    local max_wait=15
    while [ $waited -lt $max_wait ]; do
        sleep 2
        waited=$((waited + 2))
        if ! kill -0 $pid 2>/dev/null; then
            # 进程已退出，检查日志
            if [ -s "$log_file" ]; then
                echo "  ⚠  ${service_name} 启动失败，错误日志如下:"
                sed 's/^/    | /' "$log_file"
            else
                echo "  ❌ ${service_name} 启动失败（无额外错误日志）"
            fi
            return 1
        fi

        # 额外检查：端口是否已在监听（对于已知端口的服务）
        local port=""
        case "$script" in
            apps/ai_models_embedding/main.py)   port="18091" ;;
            apps/ai_models_llm/main.py)         port="18092" ;;
            apps/ai_models_vlm/main.py)         port="18094" ;;
            apps/ai_models_visual/main.py)      port="18093" ;;
            apps/knowledge_core_parser/main.py)      port="18095" ;;
            apps/knowledge_core_api/main.py)   port="18090" ;;
        esac

        if [ -n "$port" ] && ss -tlnp 2>/dev/null | grep -q ":$port "; then
            echo "✅ ${service_name} 启动成功（PID: $pid，端口: $port）"
            return 0
        fi

        # 端口还没起来，继续等待
    done

    # 超时 — 进程还在但端口没起来
    echo "  ⚠  ${service_name} 进程仍在运行（PID: $pid）但端口尚未就绪（${max_wait}s 超时），可能仍在初始化"
    return 0
}

# 启动所有服务
echo "Starting all KBot services..."
echo "  启动日志目录: $(pwd)/${STARTUP_LOG_DIR}/"
echo
for service in "${SERVICES[@]}"; do
    IFS=':' read -r name script dir <<< "$service"
    start_service "$name" "$script" "$dir" || exit 1
done

echo
echo "🎉 All KBot services started successfully!"
