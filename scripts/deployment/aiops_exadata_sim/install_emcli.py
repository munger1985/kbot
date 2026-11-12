"""从经运维核对摘要的原厂 kit 安装只读客户端；不下载、不携带 OEM 凭据。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


DESTINATION = Path("/opt/aiops/emcli")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{64}", arguments.sha256):
        raise SystemExit("需由管理员提供原厂 kit 的 SHA256 摘要。")
    if DESTINATION.exists() or DESTINATION.is_symlink():
        raise SystemExit("客户端目录已存在，保留现场，不覆盖。")
    with arguments.kit.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != arguments.sha256:
        raise SystemExit("原厂 kit 摘要不符。")
    DESTINATION.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if DESTINATION.parent.is_symlink():
        raise SystemExit("客户端父目录不能是符号链接。")
    with tempfile.TemporaryDirectory(prefix="emcli-install-", dir=DESTINATION.parent) as staging:
        client = Path(staging) / "client"
        subprocess.run(["java", "-jar", str(arguments.kit.resolve()), "-install_dir=" + str(client)],
                       check=True, timeout=120, stdout=subprocess.DEVNULL)
        subprocess.run([str(client / "emcli"), "version"], check=True, timeout=20, stdout=subprocess.DEVNULL)
        client.rename(DESTINATION)
    print(json.dumps({"installed": str(DESTINATION), "kit_sha256": digest, "credentials_configured": False}))


if __name__ == "__main__":
    main()
