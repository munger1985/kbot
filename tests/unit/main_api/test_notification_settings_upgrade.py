"""新增公共设置只执行一条建表；任何既有通知结构漂移先于 DDL 阻断。"""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Column, MetaData, Table, Uuid

ROOT = Path(__file__).resolve().parents[3]
ORACLE = (ROOT / "database/oracle/main_api").exists()
path = ROOT / ("scripts/db" if ORACLE else "scripts/deployment") / "upgrade_notification_settings.py"
spec = importlib.util.spec_from_file_location("settings_upgrade", path)
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


@pytest.mark.asyncio
@pytest.mark.parametrize("exists,apply", [(False, False), (False, True), (True, True)])
async def test_upgrade_only_creates_missing_settings_and_never_rewrites_records(exists, apply):
    table = Table("notify_settings", MetaData(), Column("id", Uuid(), primary_key=True))
    conn = AsyncMock()
    conn.scalar.return_value = int(exists)
    baseline, current = AsyncMock(), AsyncMock()
    changed = await upgrade.upgrade(conn, table=table, verify_existing=baseline,
        verify_current=current, oracle=ORACLE, apply=apply)
    baseline.assert_awaited_once_with(conn)
    assert changed == (apply and not exists)
    statements = [str(call.args[0]).strip() for call in conn.execute.await_args_list]
    assert sum(sql.startswith("CREATE TABLE") for sql in statements) == int(changed)
    assert not any(sql.startswith(("UPDATE ", "DELETE ", "DROP ", "TRUNCATE ", "INSERT ")) for sql in statements)
    assert current.await_count == int(exists or apply)
    conn.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_scope_drift_stops_before_any_schema_write():
    conn = AsyncMock()
    before = AsyncMock(side_effect=ValueError("通知范围结构不一致"))
    with pytest.raises(ValueError):
        await upgrade.upgrade(conn, table=None, verify_existing=before,
            verify_current=AsyncMock(), oracle=ORACLE, apply=True)
    conn.scalar.assert_not_awaited()
    assert not any(str(call.args[0]).startswith(("CREATE", "ALTER", "DROP")) for call in conn.execute.await_args_list)
