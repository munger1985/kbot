"""创建本地 .env，并补齐或修复服务专用加密密钥。"""

from __future__ import annotations

import base64
import re
import secrets
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _decode_key(value: str) -> bytes | None:
    normalized = value.strip().strip('"').strip("'")
    if not normalized:
        return None
    try:
        return base64.urlsafe_b64decode(normalized + "=" * (-len(normalized) % 4))
    except Exception:
        return b""


def initialize_local_env(*, root: Path = ROOT) -> tuple[str, ...]:
    """保留有效 Secret；为空或格式错误时生成新的 32 字节密钥。"""
    env_path = root / ".env"
    example_path = root / ".env.example"
    if not env_path.exists():
        env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
    text = env_path.read_text(encoding="utf-8")
    changed: list[str] = []

    def set_value(name: str, value: str, *, require_32_bytes: bool = False) -> None:
        nonlocal text
        pattern = re.compile(rf"(?m)^{re.escape(name)}=(.*)$")
        match = pattern.search(text)
        if match is not None:
            current = match.group(1)
            decoded = _decode_key(current) if require_32_bytes else None
            valid = len(decoded) == 32 if require_32_bytes else bool(
                current.strip().strip('"').strip("'")
            )
            if valid:
                return
            text = text[:match.start()] + f'{name}="{value}"' + text[match.end():]
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += f'{name}="{value}"\n'
        changed.append(name)

    def random_key() -> str:
        return base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode("ascii").rstrip("=")

    set_value(
        "KBOT_MANAGED_CREDENTIAL_KEY",
        random_key(),
        require_32_bytes=True,
    )
    set_value("KBOT_MANAGED_CREDENTIAL_KEY_VERSION", "2026-08")
    env_path.write_text(text, encoding="utf-8")
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return tuple(changed)


def main() -> int:
    changed = initialize_local_env()
    if changed:
        print("本地 .env 已补齐或修复：" + "、".join(changed))
    else:
        print("本地 .env Secret 已有效，无需修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
