"""Validate the KC migration bundle before deployment.

This is deliberately a static guard; applying SQL remains a deployment
operation. It catches ordering gaps and accidental references to the V1
KB/File/TxtChunk tables before an Oracle migration is attempted.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = ROOT / "migrations" / "kc"
V1_TOKENS = ("KBOT_MD_KB", "KBOT_MD_FILE", "KBOT_MD_TXT_CHUNK", "TXTCHUNK")


def main() -> int:
    files = sorted(MIGRATION_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    numbers = [int(re.match(r"(\d+)_", path.name).group(1)) for path in files]
    expected = list(range(1, len(files) + 1))
    errors: list[str] = []
    if numbers != expected:
        errors.append(f"migration sequence is {numbers}, expected {expected}")
    for path in files:
        text = path.read_text(encoding="utf-8").upper()
        for token in V1_TOKENS:
            if token in text:
                errors.append(f"{path.name} references V1 object {token}")
    if errors:
        print("KC migration check failed:")
        print("\n".join(errors))
        return 1
    print(f"KC migration check passed ({len(files)} scripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
