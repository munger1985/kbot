#!/usr/bin/env bash

set -euo pipefail

KBOT_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KBOT_SOURCE_ROOT"

python_bin="${KBOT_PYTHON:-}"
selected_conda_env=""
mode="${KBOT_INSTALL_MODE:-development}"

find_conda() {
    local candidate="${CONDA_EXE:-}"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    candidate="$(command -v conda || true)"
    if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    for candidate in \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

conda_env_exists() {
    local conda_bin="$1"
    local env_name="$2"
    "$conda_bin" env list 2>/dev/null \
        | awk 'NF > 1 && $1 !~ /^#/ {print $1}' \
        | grep -Fxq "$env_name"
}

if [[ -z "$python_bin" ]]; then
    conda_bin="$(find_conda || true)"
    if [[ -n "$conda_bin" ]]; then
        selected_conda_env="${KBOT_CONDA_ENV:-}"
        if [[ -z "$selected_conda_env" ]]; then
            if conda_env_exists "$conda_bin" "kbot4"; then
                selected_conda_env="kbot4"
            elif conda_env_exists "$conda_bin" "cube"; then
                selected_conda_env="cube"
                echo "⚠ 未找到 kbot4 环境，安装目标自动使用 cube。"
            fi
        fi
        if [[ -n "$selected_conda_env" ]]; then
            if ! conda_env_exists "$conda_bin" "$selected_conda_env"; then
                echo "指定的 Conda 环境不存在：$selected_conda_env" >&2
                exit 1
            fi
            python_bin="$(
                "$conda_bin" run -n "$selected_conda_env" \
                    python -c 'import sys; print(sys.executable)'
            )"
        fi
    fi
fi
if [[ -z "$python_bin" ]]; then
    python_bin="$(command -v python || command -v python3 || true)"
fi
[[ -n "$python_bin" ]] || {
    echo "未找到 Python 解释器" >&2
    exit 1
}
if [[ ! -x "$python_bin" ]]; then
    echo "Python 解释器不可执行：$python_bin" >&2
    exit 1
fi
python_bin="$("$python_bin" -c 'import sys; print(sys.executable)')"
if [[ -n "$selected_conda_env" ]]; then
    echo "KBot 安装目标：Conda 环境 ${selected_conda_env}（${python_bin}）"
else
    echo "KBot 安装目标：${python_bin}"
fi
if [[ "$#" -gt 1 || ("$#" -eq 1 && "$1" != "--production") ]]; then
    echo "Usage: $0 [--production]" >&2
    exit 2
fi
if [[ "${1:-}" == "--production" ]]; then
    mode="production"
fi
if [[ "$mode" != "development" && "$mode" != "production" ]]; then
    echo "KBOT_INSTALL_MODE 只允许 development 或 production" >&2
    exit 2
fi

members=(
    "packages/platform_core"
    "packages/platform_clients"
    "services/model_serving"
    "services/knowledge_core"
    "services/knowledge_retrieval_app"
    "services/agent_runtime"
    "services/aiops_agent"
    "services/data_query"
    "services/main_api"
)
distributions=(
    "kbot-platform-core==4.0.0"
    "kbot-platform-clients==4.0.0"
    "kbot-model-serving==4.0.0"
    "kbot-knowledge-core==4.0.0"
    "kbot-knowledge-retrieval-app==4.0.0"
    "kbot-agent-runtime==4.0.0"
    "kbot-aiops-agent==4.0.0"
    "kbot-data-query==4.0.0"
    "kbot-main-api==4.0.0"
)

for member in "${members[@]}"; do
    if [[ ! -f "$member/pyproject.toml" ]]; then
        echo "工作区成员缺少 pyproject.toml：$member" >&2
        exit 1
    fi
done

"$python_bin" -m pip install -r requirements.txt

if [[ "$mode" == "development" ]]; then
    for member in "${members[@]}"; do
        "$python_bin" -m pip install --no-deps -e "$member"
    done
else
    wheel_dir="${KBOT_WHEEL_DIR:-$KBOT_SOURCE_ROOT/var/release/wheels}"
    mkdir -p "$wheel_dir"
    for member in "${members[@]}"; do
        "$python_bin" -m pip wheel --no-deps --wheel-dir "$wheel_dir" "$member"
    done
    "$python_bin" -m pip install --force-reinstall --no-deps \
        --no-index --find-links "$wheel_dir" \
        "${distributions[@]}"
fi

KBOT_WORKSPACE_PACKAGE_MODE="$mode" \
    "$python_bin" tests/acceptance/check_workspace_packages.py

# 本地初始化创建或补齐工作区 .env；有效密钥不覆盖，格式错误的密钥会自动修复。
# 生产环境使用外部 EnvironmentFile/Secret 时可设置 KBOT_SKIP_LOCAL_ENV_INIT=1。
if [[ "${KBOT_SKIP_LOCAL_ENV_INIT:-0}" != "1" ]]; then
    "$python_bin" scripts/deployment/init_local_env.py
fi

if [[ "$mode" == "development" ]]; then
    echo "KBot 开发工作区已安装：第三方依赖 + 全部内部 editable package。"
else
    echo "KBot 生产 wheel 已构建并安装：$wheel_dir"
fi
