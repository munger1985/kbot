"""生成 Portal API Key 和可写入配置的摘要。"""

from __future__ import annotations

import argparse
import os

from platform_core.config.settings import get_security_config
from platform_core.security import generate_portal_api_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成只显示一次的 KBot Portal API Key",
    )
    parser.add_argument("--key-id", required=True, help="配置中使用的 Key ID")
    args = parser.parse_args()

    config = get_security_config()
    pepper = os.getenv(config.api_key_pepper_env)
    if not pepper:
        parser.error(f"必须先设置环境变量 {config.api_key_pepper_env}")

    raw_key, digest = generate_portal_api_key(
        key_id=args.key_id,
        pepper=pepper,
    )
    print("请立即将以下明文保存到门户 Secret，KBot 不保存该值：")
    print(raw_key)
    print("\n写入 [[security.portal_api_keys]] 的 key_digest：")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
