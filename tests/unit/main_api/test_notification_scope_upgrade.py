"""范围切换只在部署执行；有记录时必须在全部 DDL 前停止。"""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
ORACLE = (ROOT / "scripts/db/upgrade_notification_scope.py").exists()
FILE = ROOT / ("scripts/db/upgrade_notification_scope.py" if ORACLE else "scripts/deployment/upgrade_notification_scope.py")
PREFIX = "KBOT_NOTIFY_" if ORACLE else "platform_notification_external_"
spec = importlib.util.spec_from_file_location("notification_scope_upgrade", FILE)
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def result(count):
    value = Mock()
    value.scalar_one.return_value = count
    return value


@pytest.mark.asyncio
async def test_nonempty_tables_stop_before_any_ddl():
    conn = Mock(execute=AsyncMock(return_value=result(1)), commit=AsyncMock())
    with pytest.raises(ValueError, match="存在记录"):
        await upgrade.apply_scope(conn, prefix=PREFIX, oracle=ORACLE, apply=True)
    statements = [str(call.args[0]) for call in conn.execute.await_args_list]
    assert not any(sql.startswith(("ALTER", "CREATE", "DROP", "UPDATE", "DELETE", "TRUNCATE")) for sql in statements)
    conn.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_only_preflight_does_not_apply_plan():
    async def execute(statement, *args):
        return result(1 if "column" in str(statement).lower() else 0)
    conn = Mock(execute=AsyncMock(side_effect=execute), commit=AsyncMock())
    plan = await upgrade.apply_scope(conn, prefix=PREFIX, oracle=ORACLE, apply=False)
    assert plan
    assert not any(str(call.args[0]).startswith(("ALTER", "CREATE", "DROP")) for call in conn.execute.await_args_list)
    conn.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_or_partial_upgrade_stops_before_ddl():
    conn = Mock(execute=AsyncMock(return_value=result(0)), commit=AsyncMock())
    with pytest.raises(ValueError, match="不是完整"):
        await upgrade.apply_scope(conn, prefix=PREFIX, oracle=ORACLE, apply=True)
    assert not any(str(call.args[0]).startswith(("ALTER", "CREATE", "DROP")) for call in conn.execute.await_args_list)


def test_scope_plan_never_deletes_rows_or_business_tables():
    steps = upgrade.scope_steps(PREFIX, ORACLE)
    assert not any(sql.startswith(("DELETE", "TRUNCATE", "UPDATE", "DROP TABLE")) for sql in steps)
    assert not any("UNIQUE" in sql or "CHECK" in sql for sql in steps)
    if ORACLE:
        assert "ALTER TABLE KBOT_NOTIFY_JOB MODIFY (DOMAIN_ID NULL)" in steps
        assert "ALTER TABLE KBOT_MANAGED_CREDENTIAL MODIFY (DOMAIN_ID NULL)" in steps
    else:
        assert "ALTER TABLE platform_notification_delivery_job ALTER COLUMN domain_id DROP NOT NULL" in steps
        assert all("tenant_id DROP" not in sql for sql in steps)
