"""从数据库部署密码文件生成模拟器运行配置。"""

from __future__ import annotations

import configparser
import os
import stat
import tempfile
from pathlib import Path


PASSWORD_SECTIONS = {
    "database.oracle": "oracle",
    "database.postgresql": "postgresql",
    "database.mysql": "mysql",
}


def _read_password(path: Path, database_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{database_name}业务账号密码文件不存在")
    file_mode = stat.S_IMODE(resolved.stat().st_mode)
    if file_mode & 0o077:
        raise ValueError(f"{database_name}业务账号密码文件权限必须是0600或更严格")
    value = resolved.read_text(encoding="utf-8").rstrip("\r\n")
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"{database_name}业务账号密码文件内容无效")
    return value


def generate_runtime_config(
    template_path: Path,
    output_path: Path,
    password_files: dict[str, Path],
) -> Path:
    """读取三库业务密码文件并原子生成0600运行配置。"""

    template = template_path.expanduser().resolve()
    if not template.is_file():
        raise ValueError("模拟器配置模板不存在")
    if set(password_files) != set(PASSWORD_SECTIONS.values()):
        raise ValueError("必须同时提供Oracle、PostgreSQL和MySQL业务账号密码文件")

    parser = configparser.ConfigParser(interpolation=None)
    with template.open("r", encoding="utf-8") as stream:
        parser.read_file(stream)
    for section, database_name in PASSWORD_SECTIONS.items():
        if not parser.has_section(section):
            raise ValueError(f"配置模板缺少[{section}]")
        parser.set(
            section,
            "password",
            _read_password(password_files[database_name], database_name),
        )

    output = output_path.expanduser().resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            parser.write(stream)
        temporary.replace(output)
        os.chmod(output, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return output
