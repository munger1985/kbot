"""Oracle 增量部署合同必须覆盖规范对象且不能清理业务表。"""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("notification_upgrade", ROOT / "scripts/db/upgrade_notification_delivery.py")
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def test_upgrade_contract_matches_canonical_without_destructive_sql():
    sql = (ROOT / "database/oracle/operations/upgrade_notification_delivery_8.sql").read_text()
    canonical = (ROOT / "database/oracle/main_api/003_notification_delivery.sql").read_text()
    expected = upgrade.contract(sql)
    assert expected == upgrade.contract(canonical)
    assert len(expected["tables"]) == 8
    assert len(expected["indexes"]) == 7
    uncommented = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    assert not any(word in uncommented.upper() for word in ("DELETE ", "TRUNCATE ", "DROP ", "UPDATE "))
    assert expected["tables"]["KBOT_NOTIFY_JOB"]["columns"]["CONTENT"] == ["JSON", "N"]
