#!/bin/bash

# 始终从仓库根目录启动，避免调用方当前目录影响相对路径。
SERVICE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SERVICE_ROOT" || exit 1

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
    echo "❌ 未找到 Conda 命令，请先安装并配置 Conda。"
    exit 1
fi

CONDA_ROOT=$(dirname "$(dirname "$CONDA_BIN_PATH")")
source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda_env_exists() {
    conda env list 2>/dev/null \
        | awk 'NF > 1 && $1 !~ /^#/ {print $1}' \
        | grep -Fxq "$1"
}

if [ -z "${KBOT_CONDA_ENV:-}" ]; then
    if conda_env_exists "kbot3"; then
        KBOT_CONDA_ENV="kbot3"
    elif conda_env_exists "cube"; then
        KBOT_CONDA_ENV="cube"
        echo "⚠ 未找到 kbot3 环境，本地启动自动使用 cube。"
    else
        echo "❌ 未找到 kbot3 或 cube Conda 环境。"
        echo "   请创建环境，或通过 KBOT_CONDA_ENV 显式指定。"
        exit 1
    fi
fi

if ! conda_env_exists "$KBOT_CONDA_ENV"; then
    echo "❌ Conda 环境不存在：${KBOT_CONDA_ENV}"
    exit 1
fi
if ! conda activate "$KBOT_CONDA_ENV"; then
    echo "❌ 无法激活 Conda 环境：${KBOT_CONDA_ENV}"
    exit 1
fi
echo "使用 Conda 环境：${KBOT_CONDA_ENV}（$(command -v python)）"

# UI 测试服务器开关：开发环境默认开启，其他环境强制关闭。
ENVIRONMENT="${ENVIRONMENT:-development}"
KBOT_UI_ENABLED="${KBOT_UI_ENABLED:-true}"

# 日志目录（服务自身的 loguru 日志配置由各服务控制，这里只存启动 stderr）
STARTUP_LOG_DIR="logs/startup"
mkdir -p "$STARTUP_LOG_DIR"

# 统一服务列表：格式 "服务分组:进程名:脚本路径:启动目录"
# 注意：脚本路径支持相对路径和绝对路径
SERVICES=(
    "Model Serving:Embedding:apps/ai_models_embedding/main.py:."
    "Model Serving:LLM:apps/ai_models_llm/main.py:."
    "Model Serving:VLM:apps/ai_models_vlm/main.py:."
    "Model Serving:Visual:apps/ai_models_visual/main.py:."
    "Knowledge Core:API:apps/knowledge_core_api/main.py:."
    "Knowledge Core:Index Worker:apps/knowledge_core_projection/main.py:."
    "Knowledge Core:Parser:apps/knowledge_core_parser/main.py:."
    "Agent Runtime:API:apps/agent_runtime_api/main.py:."
    "Agent Runtime:Worker:apps/agent_runtime_worker/main.py:."
    "AIOps Agent:API:apps/aiops_api/main.py:."
    "AIOps Agent:Worker:apps/aiops_worker/main.py:."
    "AIOps Agent:Scheduler:apps/aiops_scheduler/main.py:."
    "AIOps Agent:DB Executor:apps/aiops_db_executor/main.py:."
    "Main API:API:apps/main_api/main.py:."
)

# 启动服务函数
start_service() {
    local service_name="$1"
    local script="$2"
    local directory="$3"
    local module="${script%.py}"
    module="${module//\//.}"
    local safe_script="${script//\//_}"
    local log_file="${STARTUP_LOG_DIR}/${safe_script%.py}.log"
    
    echo "🚀 正在启动 ${service_name}..."

    # 清空上次的启动日志，便于阅读
    : > "$log_file"

    # 通过模块入口启动，确保仓库根目录位于 Python import path。
    # 使用 exec 使后台 PID 直接对应 Python 进程，便于健康检查与优雅停止。
    (
        cd "$directory" || exit 1
        exec python -m "$module"
    ) >/dev/null 2>>"$log_file" &
    local pid=$!

    # 后台 Worker 不监听 HTTP 端口，仅确认进程没有在启动后立即退出。
    if [ "$script" = "apps/knowledge_core_projection/main.py" ] \
        || [ "$script" = "apps/agent_runtime_worker/main.py" ]; then
        sleep 1
        if ! kill -0 $pid 2>/dev/null; then
            echo "  ❌ ${service_name} Worker 在启动期间退出"
            [ -s "$log_file" ] && sed 's/^/    | /' "$log_file"
            return 1
        fi
        echo "✅ ${service_name} Worker 已启动（PID: $pid）"
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
            apps/agent_runtime_api/main.py)   port="18100" ;;
            apps/aiops_api/main.py)            port="18110" ;;
            apps/aiops_db_executor/main.py)    port="18111" ;;
            apps/aiops_worker/main.py)         port="18112" ;;
            apps/aiops_scheduler/main.py)      port="18113" ;;
            apps/main_api/main.py)             port="18099" ;;
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

start_development_ui() {
    if [ "$ENVIRONMENT" != "development" ]; then
        echo "跳过 UI 测试服务器：当前环境为 ${ENVIRONMENT}。"
        return 0
    fi
    case "${KBOT_UI_ENABLED,,}" in
        1|true|yes|on) ;;
        *)
            echo "跳过 UI 测试服务器：KBOT_UI_ENABLED=${KBOT_UI_ENABLED}。"
            return 0
            ;;
    esac

    local port="8080"
    local log_file="${STARTUP_LOG_DIR}/ui_http_server.log"
    if ss -tln 2>/dev/null | grep -q ":${port} "; then
        echo "❌ UI 测试服务器端口 ${port} 已被占用。"
        return 1
    fi

    echo "🚀 正在启动 UI 测试服务器..."
    : > "$log_file"
    (
        cd "$SERVICE_ROOT" || exit 1
        exec python3 -m http.server "$port" -d ui
    ) >/dev/null 2>>"$log_file" &
    local pid=$!
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  ❌ UI 测试服务器启动失败"
        [ -s "$log_file" ] && sed 's/^/    | /' "$log_file"
        return 1
    fi
    if ! ss -tln 2>/dev/null | grep -q ":${port} "; then
        echo "  ❌ UI 测试服务器未监听端口 ${port}"
        kill -SIGTERM "$pid" 2>/dev/null || true
        return 1
    fi
    echo "✅ UI 测试服务器启动成功（PID: $pid，地址: http://127.0.0.1:${port}）"
}

# 启动所有服务
echo "正在启动全部 KBot 服务..."
echo "  启动日志目录: $(pwd)/${STARTUP_LOG_DIR}/"
current_group=""
for service in "${SERVICES[@]}"; do
    IFS=':' read -r group name script dir <<< "$service"
    if [ "$group" != "$current_group" ]; then
        echo
        echo "━━━━━━━━━━ ${group} ━━━━━━━━━━"
        current_group="$group"
    fi
    start_service "$name" "$script" "$dir" || exit 1
done

if [ "$ENVIRONMENT" = "development" ]; then
    echo
    echo "━━━━━━━━━━ Development Tools ━━━━━━━━━━"
fi
start_development_ui || exit 1

echo
echo "🎉 全部 KBot 服务启动完成！"
