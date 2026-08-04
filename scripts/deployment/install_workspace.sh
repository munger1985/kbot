#!/usr/bin/env bash

set -euo pipefail

KBOT_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KBOT_SOURCE_ROOT"

python -m pip install -r requirements.txt

# 本地初始化只创建或补齐工作区 .env；已有密钥绝不覆盖。
# 生产环境使用外部 EnvironmentFile/Secret 时可设置 KBOT_SKIP_LOCAL_ENV_INIT=1。
if [[ "${KBOT_SKIP_LOCAL_ENV_INIT:-0}" != "1" ]]; then
    python - "$KBOT_SOURCE_ROOT" <<'PY'
from __future__ import annotations

import base64
import os
import re
import secrets
import stat
import sys
from pathlib import Path


root = Path(sys.argv[1])
env_path = root / ".env"
example_path = root / ".env.example"
if not env_path.exists():
    env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")

text = env_path.read_text(encoding="utf-8")


def empty(value: str) -> bool:
    return value.strip().strip('"').strip("'") == ""


def set_if_empty(name: str, value: str) -> None:
    global text
    pattern = re.compile(rf"(?m)^{re.escape(name)}=(.*)$")
    match = pattern.search(text)
    if match is not None:
        if not empty(match.group(1)):
            return
        text = text[:match.start()] + f'{name}="{value}"' + text[match.end():]
        return
    if not text.endswith("\n"):
        text += "\n"
    text += f'{name}="{value}"\n'


set_if_empty(
    "KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY",
    base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("="),
)
set_if_empty("KBOT_AIOPS_CREDENTIAL_KEY_VERSION", "2026-08")
env_path.write_text(text, encoding="utf-8")
os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)
PY
fi

echo "KBot 第三方依赖安装完成；服务将由启动脚本直接从源码目录加载。"
