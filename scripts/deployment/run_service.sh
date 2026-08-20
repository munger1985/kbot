#!/usr/bin/env bash

set -euo pipefail

KBOT_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KBOT_SOURCE_ROOT"

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <python.module>" >&2
    exit 2
fi

python_bin="${KBOT_PYTHON:-$(command -v python || command -v python3)}"
mode="${KBOT_INSTALL_MODE:-production}"
"$python_bin" scripts/deployment/ensure_workspace_packages.py --mode "$mode"
exec "$python_bin" -m "$1"
