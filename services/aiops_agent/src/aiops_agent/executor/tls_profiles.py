"""受控数据库 TLS Profile 解析，只允许读取配置根目录内的固定证书文件。"""

from __future__ import annotations

import re
import ssl
from pathlib import Path


_PROFILE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_PEM_BYTES = 1024 * 1024


class TLSProfileError(ValueError):
    pass


class TLSProfileResolver:
    """将逻辑 Profile ID 解析为严格校验的 SSLContext。"""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def resolve(self, profile_ref: str) -> ssl.SSLContext:
        if _PROFILE_REF.fullmatch(profile_ref) is None:
            raise TLSProfileError("TLS Profile引用格式无效")
        candidate = self._root / profile_ref
        if candidate.is_symlink():
            raise TLSProfileError("TLS Profile目录不能是符号链接")
        profile = candidate.resolve()
        if not profile.is_relative_to(self._root) or not profile.is_dir():
            raise TLSProfileError("TLS Profile不存在或越界")
        ca_file = self._regular_pem(profile / "ca.pem", required=True)
        cert_file = self._regular_pem(
            profile / "client-cert.pem", required=False
        )
        key_file = self._regular_pem(
            profile / "client-key.pem", required=False
        )
        if (cert_file is None) != (key_file is None):
            raise TLSProfileError("TLS客户端证书和私钥必须同时配置")
        try:
            context = ssl.create_default_context(cafile=str(ca_file))
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            if cert_file is not None and key_file is not None:
                context.load_cert_chain(
                    certfile=str(cert_file), keyfile=str(key_file)
                )
            return context
        except (OSError, ssl.SSLError) as exc:
            raise TLSProfileError("TLS Profile证书内容无效") from exc

    def _regular_pem(self, path: Path, *, required: bool) -> Path | None:
        if not path.exists():
            if required:
                raise TLSProfileError(f"TLS Profile缺少{path.name}")
            return None
        resolved = path.resolve()
        if (
            not resolved.is_relative_to(self._root)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size <= 0
            or path.stat().st_size > _MAX_PEM_BYTES
        ):
            raise TLSProfileError(f"TLS Profile文件无效：{path.name}")
        return resolved
