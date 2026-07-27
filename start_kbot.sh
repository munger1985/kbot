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

# native crash 时把 Python 调用栈写入所属服务运行日志。
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
# 统一脚本已把 stderr 接入服务 runtime.log，关闭 Loguru 控制台副本避免重复。
export KBOT_LOG_CONSOLE="false"

# 源码仓库启动使用明确的包搜索路径；正式部署应安装各服务 Wheel。
export PYTHONPATH="${SERVICE_ROOT}/packages/platform_core/src:${SERVICE_ROOT}/packages/platform_clients/src:${SERVICE_ROOT}/services/main_api/src:${SERVICE_ROOT}/services/agent_runtime/src:${SERVICE_ROOT}/services/knowledge_core/src:${SERVICE_ROOT}/services/aiops_agent/src:${SERVICE_ROOT}/services/model_serving/src${PYTHONPATH:+:${PYTHONPATH}}"
export KBOT_RESOURCE_DIR="${KBOT_RESOURCE_DIR:-${SERVICE_ROOT}/resources}"

# 启动脚本与各服务读取同一份部署配置。
KBOT_CONFIG_FILE="${KBOT_CONFIG_FILE:-configuration/kbot.toml}"
if [ ! -f "$KBOT_CONFIG_FILE" ]; then
    echo "❌ 未找到部署配置：${KBOT_CONFIG_FILE}"
    echo "   请先复制 configuration/kbot.toml.example 并按环境修改。"
    exit 1
fi
export KBOT_CONFIG_FILE
if ! python scripts/deployment/check_deployment.py; then
    echo "❌ 部署配置校验失败，未启动任何服务。"
    exit 1
fi
CONFIG_ENVIRONMENT=$(python -c 'import sys, tomli; print(tomli.load(open(sys.argv[1], "rb")).get("environment", "development"))' "$KBOT_CONFIG_FILE")
CONFIG_LOG_ROOT=$(python -c 'import sys, tomli; print(tomli.load(open(sys.argv[1], "rb")).get("log_dir", "./logs"))' "$KBOT_CONFIG_FILE")

# UI 测试服务器开关：开发环境默认开启，其他环境强制关闭。
ENVIRONMENT="${ENVIRONMENT:-$CONFIG_ENVIRONMENT}"
ENVIRONMENT="${ENVIRONMENT,,}"
KBOT_UI_ENABLED="${KBOT_UI_ENABLED:-true}"

LOG_ROOT="$CONFIG_LOG_ROOT"
mkdir -p "$LOG_ROOT"

# 统一服务列表：格式 "服务分组:进程名:日志服务:模块入口"
SERVICES=(
    "Model Serving:Embedding:model_serving:model_serving.entrypoints.embedding"
    "Model Serving:LLM:model_serving:model_serving.entrypoints.llm"
    "Model Serving:VLM:model_serving:model_serving.entrypoints.vlm"
    "Model Serving:Visual:model_serving:model_serving.entrypoints.visual"
    "Knowledge Core:API:knowledge_core:knowledge_core.entrypoints.api"
    "Knowledge Core:Index Worker:knowledge_core:knowledge_core.entrypoints.projection"
    "Knowledge Core:Parser:knowledge_core:knowledge_core.entrypoints.parser"
    "Agent Runtime:API:agent_runtime:agent_runtime.entrypoints.api"
    "Agent Runtime:Worker:agent_runtime:agent_runtime.entrypoints.worker"
    "AIOps Agent:API:aiops_agent:aiops_agent.entrypoints.api"
    "AIOps Agent:Worker:aiops_agent:aiops_agent.entrypoints.worker"
    "AIOps Agent:Scheduler:aiops_agent:aiops_agent.entrypoints.scheduler"
    "AIOps Agent:DB Executor:aiops_agent:aiops_agent.entrypoints.db_executor"
    "Main API:API:main_api:main_api.entrypoints.api"
)

# 启动服务函数
start_service() {
    local service_name="$1"
    local log_service="$2"
    local module="$3"
    local process_name="$module"
    local log_dir="${LOG_ROOT}/${log_service}"
    local log_file="${log_dir}/runtime.log"
    mkdir -p "$log_dir"
    touch "$log_file" "${log_dir}/access.log"
    
    echo "🚀 正在启动 ${service_name}..."

    # 外层监督进程把解释器启动错误和退出状态合并进服务运行日志。
    (
        cd "$SERVICE_ROOT" || exit 1
        python -m "$module"
        exit_code=$?
        timestamp="$(date '+%Y-%m-%d %H:%M:%S.%3N')"
        if [ "$exit_code" -eq 0 ]; then
            printf '%s | INFO     | [supervisor:%s] process-supervisor - 进程已退出 | service=%s | exit_code=0\n' \
                "$timestamp" "$process_name" "$service_name" >&2
        elif [ "$exit_code" -gt 128 ]; then
            signal_number=$((exit_code - 128))
            printf '%s | ERROR    | [supervisor:%s] process-supervisor - 进程异常退出 | service=%s | exit_code=%s | signal=%s\n' \
                "$timestamp" "$process_name" "$service_name" "$exit_code" "$signal_number" >&2
        else
            printf '%s | ERROR    | [supervisor:%s] process-supervisor - 进程异常退出 | service=%s | exit_code=%s\n' \
                "$timestamp" "$process_name" "$service_name" "$exit_code" >&2
        fi
        exit "$exit_code"
    ) >/dev/null 2>>"$log_file" &
    local pid=$!

    # 后台 Worker 不监听 HTTP 端口，仅确认进程没有在启动后立即退出。
    if [ "$module" = "knowledge_core.entrypoints.projection" ] \
        || [ "$module" = "agent_runtime.entrypoints.worker" ]; then
        sleep 1
        if ! kill -0 $pid 2>/dev/null; then
            echo "  ❌ ${service_name} Worker 在启动期间退出"
            [ -s "$log_file" ] \
                && tail -n 50 "$log_file" | sed 's/^/    | /'
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
                tail -n 50 "$log_file" | sed 's/^/    | /'
            else
                echo "  ❌ ${service_name} 启动失败（无额外错误日志）"
            fi
            return 1
        fi

        # 额外检查：端口是否已在监听（对于已知端口的服务）
        local port=""
        case "$module" in
            model_serving.entrypoints.embedding)   port="18091" ;;
            model_serving.entrypoints.llm)         port="18092" ;;
            model_serving.entrypoints.vlm)         port="18094" ;;
            model_serving.entrypoints.visual)      port="18093" ;;
            knowledge_core.entrypoints.parser)     port="18095" ;;
            knowledge_core.entrypoints.api)        port="18090" ;;
            agent_runtime.entrypoints.api)         port="18100" ;;
            aiops_agent.entrypoints.api)           port="18110" ;;
            aiops_agent.entrypoints.db_executor)   port="18111" ;;
            aiops_agent.entrypoints.worker)        port="18112" ;;
            aiops_agent.entrypoints.scheduler)     port="18113" ;;
            main_api.entrypoints.api)               port="18099" ;;
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
    if [ "$ENVIRONMENT" != "development" ] && [ "$ENVIRONMENT" != "dev" ]; then
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
    local log_dir="${LOG_ROOT}/developer_tools"
    local runtime_log="${log_dir}/runtime.log"
    local access_log="${log_dir}/access.log"
    mkdir -p "$log_dir"
    touch "$runtime_log" "$access_log"
    if ss -tln 2>/dev/null | grep -q ":${port} "; then
        echo "❌ UI 测试服务器端口 ${port} 已被占用。"
        return 1
    fi

    echo "🚀 正在启动 UI 测试服务器..."
    (
        cd "$SERVICE_ROOT" || exit 1
        exec python3 -m http.server "$port" -d tools/dev_console
    ) >>"$runtime_log" 2>>"$access_log" &
    local pid=$!
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  ❌ UI 测试服务器启动失败"
        [ -s "$access_log" ] && tail -n 30 "$access_log" | sed 's/^/    | /'
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
echo "  日志根目录: $(pwd)/${LOG_ROOT}/"
current_group=""
for service in "${SERVICES[@]}"; do
    IFS=':' read -r group name log_service module <<< "$service"
    if [ "$group" != "$current_group" ]; then
        echo
        echo "━━━━━━━━━━ ${group} ━━━━━━━━━━"
        current_group="$group"
    fi
    start_service "$name" "$log_service" "$module" || exit 1
done

if [ "$ENVIRONMENT" = "development" ] || [ "$ENVIRONMENT" = "dev" ]; then
    echo
    echo "━━━━━━━━━━ Development Tools ━━━━━━━━━━"
fi
start_development_ui || exit 1

echo
echo "🎉 全部 KBot 服务启动完成！"
