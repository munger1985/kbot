#!/usr/bin/env bash

set -euo pipefail

KBOT_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KBOT_SOURCE_ROOT"

python -m pip install -r requirements.txt
python -m pip install --no-deps \
    -e packages/platform_core \
    -e packages/platform_clients \
    -e services/main_api \
    -e services/agent_runtime \
    -e services/knowledge_core \
    -e services/aiops_agent \
    -e services/model_serving

echo "KBot 工作区依赖与可编辑服务包安装完成。"
