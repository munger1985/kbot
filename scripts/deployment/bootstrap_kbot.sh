#!/usr/bin/env bash

set -euo pipefail

KBOT_DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KBOT_DEPLOY_ROOT"

deployment_mode="development"
skip_install=0
schema_dry_run=0
start_after_install=0
config_file="${KBOT_CONFIG_FILE:-configuration/kbot.toml}"
services_config="configuration/oracle_schema_services.ini"

usage() {
    echo "Usage: $0 [--production] [--skip-install] [--schema-dry-run] [--start] [--config PATH] [--services-config PATH]"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --production)
            deployment_mode="production"
            shift
            ;;
        --skip-install)
            skip_install=1
            shift
            ;;
        --schema-dry-run)
            schema_dry_run=1
            shift
            ;;
        --start)
            start_after_install=1
            shift
            ;;
        --config)
            [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
            config_file="$2"
            shift 2
            ;;
        --services-config)
            [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
            services_config="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$deployment_mode" == "production" && "$start_after_install" -eq 1 ]]; then
    echo "生产环境不通过开发启动器拉起进程，请使用 systemd、容器或编排平台。" >&2
    exit 2
fi

if [[ ! -f "$config_file" ]]; then
    if [[ "$config_file" != "configuration/kbot.toml" ]]; then
        echo "外部部署配置不存在：$config_file" >&2
        exit 1
    fi
    cp configuration/kbot.toml.example "$config_file"
    echo "已从模板创建 $config_file；继续前请确保数据库地址和 Secret 已通过环境注入。"
fi
[[ -f "$services_config" ]] || {
    echo "服务初始化配置不存在：$services_config" >&2
    exit 1
}

find_conda() {
    local candidate="${CONDA_EXE:-}"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi
    candidate="$(command -v conda || true)"
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi
    for candidate in \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
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

resolve_python() {
    local selected="${KBOT_PYTHON:-}"
    local conda_bin=""
    local env_name="${KBOT_CONDA_ENV:-}"
    if [[ -n "$selected" ]]; then
        "$selected" -c 'import sys; print(sys.executable)'
        return
    fi
    conda_bin="$(find_conda || true)"
    if [[ -n "$conda_bin" ]]; then
        if [[ -z "$env_name" ]]; then
            if conda_env_exists "$conda_bin" "kbot4"; then
                env_name="kbot4"
            elif conda_env_exists "$conda_bin" "cube"; then
                env_name="cube"
            fi
        fi
        if [[ -n "$env_name" ]]; then
            conda_env_exists "$conda_bin" "$env_name" || {
                echo "指定的 Conda 环境不存在：$env_name" >&2
                exit 1
            }
            "$conda_bin" run -n "$env_name" python -c \
                'import sys; print(sys.executable)'
            return
        fi
    fi
    selected="$(command -v python || command -v python3 || true)"
    [[ -n "$selected" ]] || { echo "未找到 Python 解释器" >&2; exit 1; }
    "$selected" -c 'import sys; print(sys.executable)'
}

python_bin="$(resolve_python)"
export KBOT_PYTHON="$python_bin"
export KBOT_CONFIG_FILE="$config_file"

echo "[1/7] 目标解释器：$python_bin"
if [[ "$skip_install" -eq 0 ]]; then
    echo "[2/7] 安装第三方依赖和 KBot 内部包"
    if [[ "$deployment_mode" == "production" ]]; then
        bash scripts/deployment/install_workspace.sh --production
    else
        bash scripts/deployment/install_workspace.sh
    fi
else
    echo "[2/7] 已按参数跳过依赖安装"
fi

echo "[3/7] 校验配置、进程拓扑和规范 Oracle DDL"
"$python_bin" tests/acceptance/check_configuration_contract.py
"$python_bin" tests/acceptance/check_process_topology.py
"$python_bin" tests/acceptance/check_oracle_schema.py
"$python_bin" scripts/deployment/check_deployment.py

echo "[4/7] 解析建表计划"
"$python_bin" scripts/db/apply_oracle_schema.py \
    --config "$services_config" \
    --dry-run

if [[ "$schema_dry_run" -eq 1 ]]; then
    echo "Schema 预检完成；未连接数据库，也未执行 DDL 或初始化数据。"
    exit 0
fi

echo "[5/7] 初始化空白 Oracle Schema 与首次登录基础数据"
"$python_bin" scripts/db/apply_oracle_schema.py \
    --config "$services_config"

echo "[6/7] 校验默认 Domain、ADMIN、权限目录和成员授权"
"$python_bin" scripts/db/apply_oracle_schema.py --check-foundation

echo "[7/7] 完成部署初始化"
echo "已创建规范结构、Prompt Catalog、默认 Domain、ADMIN 及完整系统管理员授权。"
echo "模型、Collection、Agent、会话和 AIOps 等业务数据保持为空。"

if [[ "$start_after_install" -eq 1 ]]; then
    bash start_kbot.sh
else
    echo "开发环境可执行 bash start_kbot.sh；生产环境请使用正式进程编排启动 20 个进程。"
fi
