"""生成 Portal API Key 和可写入配置的摘要。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from platform_core.config.settings import get_security_config
from platform_core.security import generate_portal_api_key


def _toml_string(value: str) -> str:
    """使用 TOML 兼容的 JSON 字符串转义。"""
    return json.dumps(value, ensure_ascii=False)


def upsert_portal_api_key(
    *, config_path: Path, key_id: str, client_id: str, key_digest: str,
) -> None:
    """按 key_id 幂等新增或替换 Portal API Key 摘要配置块。"""
    content = config_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines)
        if line.strip() == "[[portal_api_keys]]"
    ]
    replacement = (
        "[[portal_api_keys]]\n"
        f"key_id = {_toml_string(key_id)}\n"
        f"client_id = {_toml_string(client_id)}\n"
        f"key_digest = {_toml_string(key_digest)}\n"
    )
    target: tuple[int, int] | None = None
    for start in starts:
        end = start + 1
        while end < len(lines) and not lines[end].lstrip().startswith("["):
            end += 1
        block = "".join(lines[start:end])
        if f"key_id = {_toml_string(key_id)}" in block:
            target = (start, end)
            break
    if target is None:
        separator = "" if not content or content.endswith("\n\n") else "\n"
        updated = f"{content}{separator}{replacement}"
    else:
        start, end = target
        updated = "".join([*lines[:start], replacement, *lines[end:]])
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成只显示一次的 KBot4 Portal API Key",
    )
    parser.add_argument("--key-id", required=True, help="配置中使用的 Key ID")
    parser.add_argument("--client-id", default="portal", help="Portal 客户端标识")
    parser.add_argument(
        "--config-file", type=Path,
        default=Path("configuration/kbot.toml"),
        help="需要写入摘要的 kbot.toml",
    )
    args = parser.parse_args()

    config = get_security_config()
    pepper = os.getenv(config.api_key_pepper_env)
    if not pepper:
        parser.error(f"必须先设置环境变量 {config.api_key_pepper_env}")

    raw_key, digest = generate_portal_api_key(
        key_id=args.key_id,
        pepper=pepper,
    )
    upsert_portal_api_key(
        config_path=args.config_file,
        key_id=args.key_id,
        client_id=args.client_id,
        key_digest=digest,
    )
    print("请立即将以下明文保存到门户 Secret，KBot4 不保存该值：")
    print(raw_key)
    print(f"\n摘要已写入：{args.config_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
