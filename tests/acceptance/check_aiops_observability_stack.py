#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STACK = ROOT / "services/aiops_agent/deployment/observability"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="kbot-aiops-stack-") as temporary:
        state = Path(temporary) / "state"
        password = Path(temporary) / "oracle-password"
        password.write_text("acceptance-only-password", encoding="utf-8")
        subprocess.run(
            [
                str(ROOT / "scripts/aiops-stack"),
                "configure",
                "--preset",
                "oracle-lite",
                "--state-dir",
                str(state),
                "--target-key",
                "acceptance-oracle",
                "--oracle-host",
                "127.0.0.1",
                "--oracle-service",
                "FREEPDB1",
                "--oracle-user",
                "kbot_monitor",
                "--oracle-password-file",
                str(password),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(ROOT / "scripts/aiops-stack"),
                "validate",
                "--state-dir",
                str(state),
            ],
            check=True,
        )
        if os.stat(state / "secrets/oracle_password").st_mode & 0o037:
            raise RuntimeError("Oracle Secret权限过宽")
    if "@sha256:" not in (STACK / "images.env").read_text(encoding="utf-8"):
        raise RuntimeError("镜像清单没有固定Digest")
    print("AIOps观测栈检查通过：Compose、Profile、Secret和Oracle采集配置有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
