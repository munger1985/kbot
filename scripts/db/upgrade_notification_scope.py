"""部署专用范围切换：仅接受空的旧外发结构，不删除业务或通知数据。"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
CONFIG_KINDS = ("connection", "binding", "policy", "audit")
RESOURCE_KINDS = ("job", "attempt")


def scope_steps(prefix, oracle):
    """固定对象清单；非空旧结构必须另行审阅数据转换，不自动合并。"""
    config = [prefix + (kind.upper() if oracle else kind) for kind in CONFIG_KINDS]
    resource = [prefix + (kind.upper() if oracle else kind) for kind in RESOURCE_KINDS]
    if oracle:
        return [*(f"ALTER TABLE {name} DROP COLUMN DOMAIN_ID" for name in config),
                *(f"ALTER TABLE {name} MODIFY (DOMAIN_ID NULL)" for name in resource),
                "ALTER TABLE KBOT_MANAGED_CREDENTIAL MODIFY (DOMAIN_ID NULL)",
                "CREATE INDEX IX_NOTIFY_CONN_SCOPE ON KBOT_NOTIFY_CONNECTION (PROVIDER, ENABLED)"]
    return [*(f"ALTER TABLE {name} DROP COLUMN domain_id" for name in config),
            *(f"ALTER TABLE {name} ALTER COLUMN domain_id DROP NOT NULL" for name in resource),
            f"CREATE INDEX ix_notify_conn_scope ON {prefix}connection (tenant_id, provider, enabled)"]


async def apply_scope(conn, *, prefix, oracle, apply):
    names = [prefix + (kind.upper() if oracle else kind) for kind in (*CONFIG_KINDS, *RESOURCE_KINDS, "entry")]
    # 部署者必须先停止通知生产者与 Worker；PG 再加事务级表锁。
    if not oracle:
        await conn.execute(text("SET LOCAL lock_timeout = '5s'"))
        await conn.execute(text("SET LOCAL statement_timeout = '60s'"))
        await conn.execute(text("LOCK TABLE " + ", ".join(names) + " IN ACCESS EXCLUSIVE MODE"))
    for name in names:
        if (await conn.execute(text(f"SELECT count(*) FROM {name}"))).scalar_one():
            raise ValueError("旧外发结构存在记录，停止范围切换；未清除或合并任何数据")
    if oracle and (await conn.execute(text(
        "SELECT count(*) FROM KBOT_MANAGED_CREDENTIAL WHERE NAMESPACE='notifications' AND STATUS='ACTIVE'"
    ))).scalar_one():
        raise ValueError("存在旧通知凭据，需先审阅密文范围转换；未改写任何数据")
    # 拒绝重复执行或未知部分升级；先核对全部旧配置列，尚未执行任何 DDL。
    for name in [prefix + (kind.upper() if oracle else kind) for kind in CONFIG_KINDS]:
        if oracle:
            count = (await conn.execute(text(
                "SELECT count(*) FROM user_tab_columns WHERE table_name=:name AND column_name='DOMAIN_ID'"
            ), {"name": name})).scalar_one()
        else:
            count = (await conn.execute(text(
                "SELECT count(*) FROM information_schema.columns WHERE table_schema=current_schema() "
                "AND table_name=:name AND column_name='domain_id'"
            ), {"name": name})).scalar_one()
        if count != 1:
            raise ValueError("不是完整的旧 Domain 范围结构，停止；请验收当前规范，不要重复切换")
    steps = scope_steps(prefix, oracle)
    if not apply:
        return steps
    # 删除的仅为待替换普通索引，不删除表或数据。
    if oracle:
        await conn.execute(text("ALTER SESSION SET DDL_LOCK_TIMEOUT = 5"))
    await conn.execute(text("DROP INDEX " + ("IX_NOTIFY_CONN_SCOPE" if oracle else "ix_notify_conn_scope")))
    for statement in steps:
        await conn.execute(text(statement))
    await conn.commit()
    return steps

async def main(args):
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "packages/platform_core/src")]
    from dotenv import load_dotenv
    from platform_core.config import get_settings
    from platform_core.database.oracle import create_database_runtime
    load_dotenv(ROOT / ".env", override=False)
    runtime = create_database_runtime(get_settings())
    try:
        async with runtime.engine.connect() as conn:
            steps = await apply_scope(conn, prefix="KBOT_NOTIFY_", oracle=True, apply=args.apply)
            print("范围切换已执行" if args.apply else "只读预检通过，尚未执行 DDL")
            for statement in steps:
                print(statement)
    finally:
        await runtime.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通知全局范围切换；必须先停止生产者与 Worker，仅接受空旧结构")
    parser.add_argument("--apply", action="store_true")
    try:
        asyncio.run(main(parser.parse_args()))
    except Exception as exc:
        print(str(exc) if isinstance(exc, ValueError) else "部署预检／DDL 执行失败，请只读核对元数据；未输出可能含凭据的异常正文")
        raise SystemExit(1) from None
