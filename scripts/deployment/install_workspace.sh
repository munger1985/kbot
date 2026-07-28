#!/usr/bin/env bash

set -euo pipefail

KBOT_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KBOT_SOURCE_ROOT"

python -m pip install -r requirements.txt

echo "KBot 第三方依赖安装完成；服务将由启动脚本直接从源码目录加载。"
