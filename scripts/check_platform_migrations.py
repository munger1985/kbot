"""部署前静态检查 Platform Migration。"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = ROOT / "migrations" / "platform"
FORBIDDEN_RUNTIME_OBJECTS = (
    "KBOT_MD_DOMAIN",
    "KBOT_MD_USER",
)


def main() -> int:
    files = sorted(MIGRATION_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    numbers = [
        int(re.match(r"(\d+)_", path.name).group(1))
        for path in files
    ]
    expected = list(range(1, len(files) + 1))
    errors: list[str] = []
    if numbers != expected:
        errors.append(f"Migration 序号为 {numbers}，预期为 {expected}")
    for path in files:
        content = path.read_text(encoding="utf-8").upper()
        for object_name in FORBIDDEN_RUNTIME_OBJECTS:
            if object_name in content:
                errors.append(
                    f"{path.name} 禁止依赖旧对象 {object_name}"
                )
    if errors:
        print("Platform Migration 检查失败：")
        print("\n".join(errors))
        return 1
    print(f"Platform Migration 检查通过（{len(files)} 个脚本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
