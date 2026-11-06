"""容器安装器专用：初始化空库，或只读接管同范围的已有部署。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_accounts(path: Path, accounts: dict[str, str]) -> None:
    """交接文件只写一次；不在 stdout、日志或命令参数中暴露密码。"""
    if path.exists() or path.is_symlink():
        raise RuntimeError("首次账号交接文件已存在，拒绝覆盖或重置密码")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        try:
            json.dump({"accounts": accounts, "must_change_password": True}, target, ensure_ascii=False, indent=2)
            target.flush()
            os.fsync(target.fileno())
            temporary.chmod(0o600)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


async def initialize(apps: list[str], *, check_only: bool, accounts_path: Path) -> None:
    # 延迟导入，让 --help 能在连接数据库前检查安装器能力。
    import bcrypt
    from sqlalchemy import select

    from main_api.entities.access_control import PlatformUserCredentialEntity
    from platform_core.config import get_settings
    from platform_core.database.oracle import create_database_runtime
    from scripts.db import apply_oracle_schema as schema
    from scripts.db.initialize_aiops import initialize_aiops
    from scripts.db.initialize_product_app import APP_BOOTSTRAPS, initialize_app
    from scripts.db.seed_aiops_inspection_templates import seed_default_inspection_templates

    config_path = ROOT / "configuration/oracle_schema_services.ini"
    catalog = tomllib.loads((ROOT / "installation/catalog.toml").read_text(encoding="utf-8"))
    expected = set(catalog["base_pack"]["service_configs"]) | {"platform_core"}
    for app_id in apps:
        expected.update(catalog["app_packs"][app_id]["service_configs"])
    if set(schema.load_service_selection(config_path).ordered) != expected:
        raise RuntimeError("App 选择与规范 Schema 初始化范围不一致")
    if check_only:
        # 接管是显式部署操作，复用规范 DDL 的对象校验，不放入应用启动或 readiness。
        labels = [statement.label for statement in schema.load_schema_statements(tuple(sorted(expected)))]
        tables = {label.rsplit(" ", 1)[1] for label in labels if label.startswith("CREATE TABLE ")}
        views = {label.rsplit(" ", 1)[1] for label in labels if label.startswith(("CREATE VIEW ", "CREATE OR REPLACE VIEW "))}
        runtime = create_database_runtime(get_settings())
        try:
            async with runtime.engine.connect() as connection:
                aiops_manifest = json.loads(
                    (ROOT / "database/oracle/aiops_agent/schema_manifest.json").read_text(encoding="utf-8")
                ) if "aiops" in apps else None
                await schema._validate_schema(
                    connection, expected_tables=tables, expected_views=views,
                    aiops_manifest=aiops_manifest,
                )
        finally:
            await runtime.close()
    if not check_only:
        if accounts_path.exists() or accounts_path.is_symlink():
            raise RuntimeError("已有首次账号交接文件，拒绝重复初始化")
        await schema.apply_schema(dry_run=False, config_path=config_path)
    await schema.maintain_platform_foundation(repair=False)
    for app_id in apps:
        if app_id == "aiops":
            await initialize_aiops(check_only=check_only)
        else:
            await initialize_app(APP_BOOTSTRAPS[app_id], check_only=check_only)
    if check_only:
        print("已有平台及所选 App 的基础账号与授权校验通过；未执行 DDL 或修改密码")
        return
    if "aiops" in apps:
        await seed_default_inspection_templates()
    accounts = {"ADMIN": secrets.token_urlsafe(24)}
    for app_id in apps:
        user = "aiopsadmin" if app_id == "aiops" else APP_BOOTSTRAPS[app_id].admin_user
        accounts[user] = secrets.token_urlsafe(24)
    # 先保存可恢复的私有交接文件，再一次性提交随机凭据；失败由外层安装记录阻止重试。
    write_accounts(accounts_path, accounts)
    runtime = create_database_runtime(get_settings())
    try:
        async with runtime.session_factory() as session:
            async with session.begin():
                credentials = (await session.execute(
                    select(PlatformUserCredentialEntity)
                    .where(PlatformUserCredentialEntity.user_id.in_(accounts))
                    .with_for_update()
                )).scalars().all()
                if {credential.user_id for credential in credentials} != set(accounts):
                    raise RuntimeError("首次管理员登录凭据不完整")
                for credential in credentials:
                    credential.password_hash = bcrypt.hashpw(
                        accounts[credential.user_id].encode("utf-8"), bcrypt.gensalt(rounds=12)
                    ).decode("ascii")
                    credential.must_change_password = "Y"
                    credential.password_updated_at = datetime.now(timezone.utc)
        print("首次安装完成；随机账号密码已写入私有交接文件，首次登录必须修改")
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", nargs="+", choices=("knowledge_retrieval", "media_studio", "aiops"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--accounts-file", type=Path, default=Path("/var/lib/kbot/.kbot/initial-accounts.json"))
    args = parser.parse_args()
    try:
        asyncio.run(initialize(args.apps, check_only=args.check_only, accounts_path=args.accounts_file))
    except Exception as error:
        # 数据库异常可能包含连接串或绑定值，不输出原始异常。
        origin = getattr(error, "orig", error)
        detail = origin.args[0] if origin.args else None
        code = str(getattr(detail, "full_code", ""))
        code = code if re.fullmatch(r"(?:ORA-\d{5}|DPY-\d{4}|DPI-\d{4})", code) else ""
        print(f"INSTALLATION_ERROR={type(error).__name__} {code}；安装初始化失败，请检查受控数据库状态。禁止直接重放空库脚本。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
