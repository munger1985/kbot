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
    if conda_env_exists "kbot4"; then
        KBOT_CONDA_ENV="kbot4"
    elif conda_env_exists "cube"; then
        KBOT_CONDA_ENV="cube"
        echo "⚠ 未找到 kbot4 环境，本地启动自动使用 cube。"
    else
        echo "❌ 未找到 kbot4 或 cube Conda 环境。"
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

# 启动前先以内容指纹校验内部包。版本号相同但 Wheel 内容落后时，
# 在跨进程锁内自动更新；更新失败则不允许服务继续加载混合版本。
KBOT_CONFIG_FILE="${KBOT_CONFIG_FILE:-configuration/kbot.toml}"
if [ ! -f "$KBOT_CONFIG_FILE" ]; then
    echo "❌ 未找到部署配置：${KBOT_CONFIG_FILE}"
    echo "   请先复制 configuration/kbot.toml.example 并按环境修改。"
    exit 1
fi
export KBOT_CONFIG_FILE
CONFIG_ENVIRONMENT=$(python -c 'import sys; exec("try:\n import tomllib as toml\nexcept ImportError:\n import tomli as toml"); print(toml.load(open(sys.argv[1], "rb")).get("environment", "development"))' "$KBOT_CONFIG_FILE")
PACKAGE_INSTALL_MODE="${KBOT_INSTALL_MODE:-$CONFIG_ENVIRONMENT}"
case "${PACKAGE_INSTALL_MODE,,}" in
    production) PACKAGE_INSTALL_MODE="production" ;;
    *) PACKAGE_INSTALL_MODE="development" ;;
esac
if ! python scripts/deployment/ensure_workspace_packages.py \
    --mode "$PACKAGE_INSTALL_MODE"; then
    echo "❌ 内部包与源码不一致且自动更新失败，未启动任何服务。"
    exit 1
fi

# 安装与启动必须使用同一个解释器，避免内部包被装入 base 或其他 Conda 环境。
if ! python -c '
import agent_runtime
import aiops_agent
import data_query
import knowledge_core
import knowledge_retrieval_app
import km_asset_app
import main_api
import model_serving
import platform_clients
import platform_core
' >/dev/null 2>&1; then
    echo "❌ 当前 Conda 环境缺少 KBot 内部包：${KBOT_CONDA_ENV}（$(command -v python)）"
    echo "   请执行：KBOT_CONDA_ENV=${KBOT_CONDA_ENV} bash scripts/deployment/install_workspace.sh"
    exit 1
fi

# native crash 时把 Python 调用栈写入所属服务运行日志。
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
# 统一脚本已把 stderr 接入服务 runtime.log，关闭 Loguru 控制台副本避免重复。
export KBOT_LOG_CONSOLE="false"

# 开发环境由 install_workspace.sh 以 editable package 安装内部包，生产环境安装 Wheel。
# 启动过程不拼接源码搜索路径，确保两种模式使用相同的包边界。
export KBOT_RESOURCE_DIR="${KBOT_RESOURCE_DIR:-${SERVICE_ROOT}/resources}"

# 启动脚本与各服务读取同一份部署配置。
if ! python scripts/deployment/check_deployment.py; then
    echo "❌ 部署配置校验失败，未启动任何服务。"
    exit 1
fi
CONFIG_LOG_ROOT=$(python -c 'import sys, tomli; print(tomli.load(open(sys.argv[1], "rb")).get("log_dir", "./logs"))' "$KBOT_CONFIG_FILE")

LOG_ROOT="$CONFIG_LOG_ROOT"
STARTUP_LOG_FILE="${LOG_ROOT}/main_api/runtime.log"
mkdir -p "$(dirname "$STARTUP_LOG_FILE")"

append_startup_log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S.%3N')"
    printf '%s | %-8s | [startup:preflight] startup - %s\n' \
        "$timestamp" "$level" "$message" >>"$STARTUP_LOG_FILE"
}

# 把包预检结论写入运维日志，便于确认服务实际加载的源码或 Wheel。
AGENT_RUNTIME_ORIGIN=$(python -c '
import agent_runtime
print(agent_runtime.__file__ or "-")
')
SOURCE_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')
append_startup_log "INFO" \
    "内部包预检通过：mode=${PACKAGE_INSTALL_MODE} commit=${SOURCE_COMMIT} python=$(command -v python) agent_runtime=${AGENT_RUNTIME_ORIGIN}"

FOUNDATION_MODE="--check-foundation"
FOUNDATION_ACTION="只读校验"
FOUNDATION_CHECK_OUTPUT=$(
    python scripts/db/apply_oracle_schema.py "$FOUNDATION_MODE" 2>&1
)
FOUNDATION_CHECK_STATUS=$?
case "${CONFIG_ENVIRONMENT,,}:${FOUNDATION_CHECK_STATUS}" in
    dev:3|development:3|debug:3)
        FOUNDATION_MODE="--foundation-only"
        FOUNDATION_ACTION="自动修复并校验"
        FOUNDATION_REPAIR_OUTPUT=$(
            python scripts/db/apply_oracle_schema.py "$FOUNDATION_MODE" 2>&1
        )
        FOUNDATION_CHECK_STATUS=$?
        FOUNDATION_CHECK_OUTPUT="${FOUNDATION_CHECK_OUTPUT}"$'\n'\
"检测到开发库基础数据漂移，已自动执行幂等修复："$'\n'\
"${FOUNDATION_REPAIR_OUTPUT}"
        ;;
esac
if [ -n "$FOUNDATION_CHECK_OUTPUT" ]; then
    printf '%s\n' "$FOUNDATION_CHECK_OUTPUT"
    while IFS= read -r output_line; do
        if [ "$FOUNDATION_CHECK_STATUS" -eq 0 ]; then
            append_startup_log "INFO" \
                "平台基础数据${FOUNDATION_ACTION}：${output_line}"
        else
            append_startup_log "ERROR" \
                "平台基础数据${FOUNDATION_ACTION}：${output_line}"
        fi
    done <<< "$FOUNDATION_CHECK_OUTPUT"
fi

case "$FOUNDATION_CHECK_STATUS" in
    0)
        ;;
    3)
        echo "❌ 平台基础数据校验未通过，未启动任何服务。"
        echo "   请根据上方 PDB、Schema 和具体缺失项确认目标后执行："
        echo "   python scripts/db/apply_oracle_schema.py --foundation-only"
        append_startup_log "ERROR" "平台基础数据不完整，启动已终止"
        exit 1
        ;;
    *)
        echo "❌ 平台基础数据${FOUNDATION_ACTION}失败，未启动任何服务。"
        if [ "$FOUNDATION_MODE" = "--foundation-only" ]; then
            echo "   开发环境已自动尝试幂等修复，请检查上方具体错误、数据库连通性和 Schema 权限。"
        else
            echo "   这不代表系统未初始化；请检查数据库连通性、部署配置和 Conda 环境。"
        fi
        echo "   完整输出已记录：${STARTUP_LOG_FILE}"
        append_startup_log "ERROR" \
            "预检执行异常，退出码=${FOUNDATION_CHECK_STATUS}，启动已终止"
        exit 1
        ;;
esac

# UI 测试服务器开关：开发环境默认开启，其他环境强制关闭。
ENVIRONMENT="${ENVIRONMENT:-$CONFIG_ENVIRONMENT}"
ENVIRONMENT="${ENVIRONMENT,,}"
KBOT_UI_ENABLED="${KBOT_UI_ENABLED:-true}"

mkdir -p "$LOG_ROOT"

# 统一服务列表：格式 "服务分组:进程名:日志服务:模块入口"
SERVICES=(
    "Model Serving:Embedding:model_serving:model_serving.entrypoints.embedding"
    "Model Serving:LLM:model_serving:model_serving.entrypoints.llm"
    "Model Serving:VLM:model_serving:model_serving.entrypoints.vlm"
    "Model Serving:Visual:model_serving:model_serving.entrypoints.visual"
    "Model Serving:OCR:model_serving:model_serving.entrypoints.ocr"
    "Knowledge Core:API:knowledge_core:knowledge_core.entrypoints.api"
    "Knowledge Core:Index Worker:knowledge_core:knowledge_core.entrypoints.projection"
    "Knowledge Core:Parser:knowledge_core:knowledge_core.entrypoints.parser"
    "Knowledge Retrieval App:API:knowledge_retrieval_app:knowledge_retrieval_app.entrypoints.api"
    "KM Asset App:API:km_asset_app:km_asset_app.entrypoints.api"
    "KM Asset App:Worker:km_asset_app:km_asset_app.entrypoints.worker"
    "KM Asset App:Slack Worker:km_asset_app:km_asset_app.entrypoints.slack_worker"
    "Agent Runtime:API:agent_runtime:agent_runtime.entrypoints.api"
    "Agent Runtime:Worker:agent_runtime:agent_runtime.entrypoints.worker"
    "Data Query:API:data_query:data_query.entrypoints.api"
    "Data Query:Worker:data_query:data_query.entrypoints.worker"
    "AIOps Agent:API:aiops_agent:aiops_agent.entrypoints.api"
    "AIOps Agent:Worker:aiops_agent:aiops_agent.entrypoints.worker"
    "AIOps Agent:Scheduler:aiops_agent:aiops_agent.entrypoints.scheduler"
    "AIOps Agent:DB Executor:aiops_agent:aiops_agent.entrypoints.db_executor"
    "Main API:API:main_api:main_api.entrypoints.api"
    "Main API:Notification Worker:main_api:main_api.entrypoints.notification_worker"
)

# 返回模块对应的监听端口；纯后台 Worker 返回空值。
service_port() {
    case "$1" in
        model_serving.entrypoints.embedding) echo "18091" ;;
        model_serving.entrypoints.llm) echo "18092" ;;
        model_serving.entrypoints.vlm) echo "18094" ;;
        model_serving.entrypoints.visual) echo "18093" ;;
        model_serving.entrypoints.ocr) echo "18096" ;;
        knowledge_core.entrypoints.parser) echo "18095" ;;
        knowledge_core.entrypoints.api) echo "18090" ;;
        knowledge_retrieval_app.entrypoints.api) echo "18150" ;;
        km_asset_app.entrypoints.api) echo "18160" ;;
        agent_runtime.entrypoints.api) echo "18100" ;;
        data_query.entrypoints.api) echo "18140" ;;
        data_query.entrypoints.worker) echo "18141" ;;
        aiops_agent.entrypoints.api) echo "18110" ;;
        aiops_agent.entrypoints.db_executor) echo "18111" ;;
        aiops_agent.entrypoints.worker) echo "18112" ;;
        aiops_agent.entrypoints.scheduler) echo "18113" ;;
        main_api.entrypoints.api) echo "18099" ;;
        *) echo "" ;;
    esac
}

is_process_only_service() {
    [ "$1" = "knowledge_core.entrypoints.projection" ] \
        || [ "$1" = "km_asset_app.entrypoints.worker" ] \
        || [ "$1" = "km_asset_app.entrypoints.slack_worker" ] \
        || [ "$1" = "agent_runtime.entrypoints.worker" ] \
        || [ "$1" = "main_api.entrypoints.notification_worker" ]
}

STARTED_PIDS=()
STARTED_NAMES=()
STARTED_LOG_FILES=()
STARTED_MODULES=()
STARTED_PORTS=()
STARTED_STATES=()

# 只负责拉起进程，不在这里串行等待端口。
launch_service() {
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
    STARTED_PIDS+=("$pid")
    STARTED_NAMES+=("$service_name")
    STARTED_LOG_FILES+=("$log_file")
    STARTED_MODULES+=("$module")
    STARTED_PORTS+=("$(service_port "$module")")
    STARTED_STATES+=("PENDING")
    echo "  ↳ ${service_name} 已拉起（PID: $pid）"
}

report_start_failure() {
    local service_name="$1"
    local log_file="$2"
    if [ -s "$log_file" ]; then
        echo "  ⚠  ${service_name} 启动失败，错误日志如下:"
        tail -n 50 "$log_file" | sed 's/^/    | /'
    else
        echo "  ❌ ${service_name} 启动失败（无额外错误日志）"
    fi
}

check_service_ports_available() {
    local listening service group name log_service module port
    local conflict=0
    listening="$(ss -tln 2>/dev/null || true)"
    for service in "${SERVICES[@]}"; do
        IFS=':' read -r group name log_service module <<< "$service"
        port="$(service_port "$module")"
        [ -n "$port" ] || continue
        if grep -q ":${port} " <<< "$listening"; then
            echo "❌ ${group} / ${name} 端口 ${port} 已被占用。"
            conflict=1
        fi
    done
    [ "$conflict" -eq 0 ]
}

# 所有进程拉起后统一轮询，整体等待时间不再按服务数量累加。
wait_for_services() {
    local max_wait="${KBOT_STARTUP_TIMEOUT_SECONDS:-30}"
    if ! [[ "$max_wait" =~ ^[1-9][0-9]*$ ]]; then
        echo "❌ KBOT_STARTUP_TIMEOUT_SECONDS 必须是正整数。"
        return 1
    fi
    local elapsed=0
    local total="${#STARTED_PIDS[@]}"
    local remaining="$total"
    local index pid name module port log_file listening

    echo
    echo "⏳ 正在并行等待 ${total} 个服务就绪（最长 ${max_wait}s）..."
    while [ "$elapsed" -lt "$max_wait" ]; do
        listening="$(ss -tln 2>/dev/null || true)"
        for index in "${!STARTED_PIDS[@]}"; do
            [ "${STARTED_STATES[$index]}" = "PENDING" ] || continue
            pid="${STARTED_PIDS[$index]}"
            name="${STARTED_NAMES[$index]}"
            module="${STARTED_MODULES[$index]}"
            port="${STARTED_PORTS[$index]}"
            log_file="${STARTED_LOG_FILES[$index]}"

            if ! kill -0 "$pid" 2>/dev/null; then
                STARTED_STATES[$index]="FAILED"
                report_start_failure "$name" "$log_file"
                return 1
            fi
            if is_process_only_service "$module" && [ "$elapsed" -ge 1 ]; then
                STARTED_STATES[$index]="READY"
                remaining=$((remaining - 1))
                echo "✅ ${name} 已启动（PID: $pid，无监听端口）"
                continue
            fi
            if [ -n "$port" ] \
                && grep -q ":${port} " <<< "$listening"; then
                STARTED_STATES[$index]="READY"
                remaining=$((remaining - 1))
                echo "✅ ${name} 启动成功（PID: $pid，端口: $port）"
            fi
        done
        [ "$remaining" -eq 0 ] && return 0
        sleep 1
        elapsed=$((elapsed + 1))
    done

    for index in "${!STARTED_PIDS[@]}"; do
        [ "${STARTED_STATES[$index]}" = "PENDING" ] || continue
        echo "  ⚠  ${STARTED_NAMES[$index]} 进程仍在运行（PID: ${STARTED_PIDS[$index]}）但端口 ${STARTED_PORTS[$index]} 尚未就绪（${max_wait}s 超时）"
    done
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
        exec python3 tools/dev_console/server.py --port "$port"
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
check_service_ports_available || exit 1
current_group=""
for service in "${SERVICES[@]}"; do
    IFS=':' read -r group name log_service module <<< "$service"
    if [ "$group" != "$current_group" ]; then
        echo
        echo "━━━━━━━━━━ ${group} ━━━━━━━━━━"
        current_group="$group"
    fi
    launch_service "${group} / ${name}" "$log_service" "$module" || exit 1
done

if [ "$ENVIRONMENT" = "development" ] || [ "$ENVIRONMENT" = "dev" ]; then
    echo
    echo "━━━━━━━━━━ Development Tools ━━━━━━━━━━"
fi
start_development_ui || exit 1
wait_for_services || exit 1

echo
echo "🎉 全部 KBot 服务启动完成！"
