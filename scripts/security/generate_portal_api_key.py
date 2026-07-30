"""生成 Portal API Key 和可写入配置的摘要。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import re
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PLATFORM_CORE_SRC = ROOT / "packages" / "platform_core" / "src"
if str(PLATFORM_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_CORE_SRC))

_PORTAL_KEY_TABLE = "[[portal_api_keys]]"
_KEY_ID_LINE = re.compile(r'^\s*key_id\s*=\s*["\'](?P<value>[^"\']+)["\']\s*(?:#.*)?$')
_CLIENT_ID_LINE = re.compile(r'^\s*client_id\s*=\s*["\'][^"\']*["\']\s*(?:#.*)?$')
_KEY_DIGEST_LINE = re.compile(r'^\s*key_digest\s*=\s*["\'][^"\']*["\']\s*(?:#.*)?$')


def _config_path(value: str | None) -> Path:
    """解析部署配置路径，优先使用显式参数。"""
    return Path(
        value
        or os.getenv("KBOT_CONFIG_FILE")
        or ROOT / "configuration" / "kbot.toml"
    )


def _replace_or_add(
    lines: list[str],
    pattern: re.Pattern[str],
    field: str,
    value: str,
) -> list[str]:
    """替换表项；缺失时在当前数组表末尾追加。"""
    replacement = f"{field} = {json.dumps(value, ensure_ascii=False)}\n"
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = replacement
            return lines
    lines.append(replacement)
    return lines


def upsert_portal_api_key(
    *,
    config_path: Path,
    key_id: str,
    client_id: str,
    key_digest: str,
) -> None:
    """原子地写入或替换指定 Portal Key 的非敏感注册信息。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到部署配置文件：{config_path}")
    if not client_id or len(client_id) > 128:
        raise ValueError("client_id 长度必须在 1 到 128 个字符之间")

    original = config_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    table_starts = [
        index for index, line in enumerate(lines)
        if line.strip() == _PORTAL_KEY_TABLE
    ]
    target_start = None
    target_end = None
    for start in table_starts:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].lstrip().startswith("[[")
            ),
            len(lines),
        )
        if any(
            (match := _KEY_ID_LINE.match(line)) and match.group("value") == key_id
            for line in lines[start + 1:end]
        ):
            target_start, target_end = start, end
            break

    if target_start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend(
            [
                f"{_PORTAL_KEY_TABLE}\n",
                f"key_id = {json.dumps(key_id, ensure_ascii=False)}\n",
                f"client_id = {json.dumps(client_id, ensure_ascii=False)}\n",
                f"key_digest = {json.dumps(key_digest, ensure_ascii=False)}\n",
            ]
        )
    else:
        table_lines = lines[target_start + 1:target_end]
        _replace_or_add(table_lines, _CLIENT_ID_LINE, "client_id", client_id)
        _replace_or_add(
            table_lines,
            _KEY_DIGEST_LINE,
            "key_digest",
            key_digest,
        )
        lines[target_start + 1:target_end] = table_lines

    mode = stat.S_IMODE(config_path.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=config_path.parent,
        prefix=f".{config_path.name}.",
        delete=False,
    ) as temporary:
        temporary.writelines(lines)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(mode)
    try:
        os.replace(temporary_path, config_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成只显示一次的 KBot Portal API Key",
    )
    parser.add_argument("--key-id", required=True, help="配置中使用的 Key ID")
    parser.add_argument("--client-id", default="portal", help="调用方 Client ID")
    parser.add_argument("--config", help="待更新的 kbot.toml 路径")
    args = parser.parse_args()

    from platform_core.config.settings import get_security_config  # noqa: E402
    from platform_core.security import generate_portal_api_key  # noqa: E402

    config = get_security_config()
    pepper = os.getenv(config.api_key_pepper_env)
    if not pepper:
        parser.error(f"必须先设置环境变量 {config.api_key_pepper_env}")

    raw_key, digest = generate_portal_api_key(
        key_id=args.key_id,
        pepper=pepper,
    )
    config_path = _config_path(args.config)
    upsert_portal_api_key(
        config_path=config_path,
        key_id=args.key_id,
        client_id=args.client_id,
        key_digest=digest,
    )
    print(f"已更新 {config_path} 中的 [[portal_api_keys]] 注册信息。")
    print("请立即将以下明文保存到门户 Secret，KBot 不保存该值：")
    print(raw_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
