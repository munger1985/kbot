"""显式 Oracle 定向对齐：默认只读、固定列、幂等及历史数据保护。"""

from copy import deepcopy
import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.deployment import align_aiops_resource_collection as deployment


def state():
    return {"target_fingerprint": "fixed-target", "columns": [
        ["ACTOR_ID", "VARCHAR2", 256, "N"], ["AGENT_ID", "RAW", 16, "N"],
        ["AGENT_VERSION_ID", "RAW", 16, "N"]], "constraints": [
            ["NN_AGENT", "C", None, '"AGENT_ID" IS NOT NULL'],
            ["NN_VERSION", "C", None, '"AGENT_VERSION_ID" IS NOT NULL'],
            ["FK_AGENT", "R", None, None], ["PK_RUN", "P", None, None],
            ["NN_ACTOR", "C", None, '"ACTOR_ID" IS NOT NULL']],
        "indexes": [["PK_RUN", "UNIQUE", "VALID"]], "index_columns": [["PK_RUN", "OPS_RUN_ID", 1, "ASC"]],
        "run_count": 11245, "run_ids_hash": "historical-ids"}


def aligned(before):
    after = deepcopy(before)
    for column in after["columns"]:
        if column[0] in deployment.AGENT_COLUMNS:
            column[3] = "Y"
    after["constraints"] = [row for row in after["constraints"] if not deployment.nullable_constraint(row)]
    return after


def test_plan_is_fixed_and_idempotent():
    before = state()
    assert deployment.build_plan(before) == [
        "ALTER TABLE KBOT_OPS_RUN MODIFY (AGENT_ID NULL, AGENT_VERSION_ID NULL)"]
    before["columns"][1][3] = "Y"
    assert deployment.build_plan(before) == ["ALTER TABLE KBOT_OPS_RUN MODIFY (AGENT_VERSION_ID NULL)"]
    before["columns"][2][3] = "Y"
    assert deployment.build_plan(before) == []


@pytest.mark.parametrize("column", [None, ["AGENT_ID", "VARCHAR2", 16, "N"],
    ["AGENT_ID", "RAW", 32, "N"], ["AGENT_ID", "RAW", 16, "?"]])
def test_rejects_missing_or_drifted_columns(column):
    before = state()
    before["columns"] = [row for row in before["columns"] if row[0] != "AGENT_ID"]
    if column:
        before["columns"].append(column)
    with pytest.raises(ValueError):
        deployment.build_plan(before)


def test_readonly_does_not_execute_ddl_or_write_reports(tmp_path):
    connection = MagicMock()
    with patch.object(deployment, "snapshot", return_value=state()):
        summary = deployment.align(connection, report_dir=tmp_path)
    assert not summary["apply"]
    connection.cursor.assert_not_called()
    assert not list(tmp_path.iterdir())


def plan_hash(before):
    return deployment.digest({"before": before, "statements": deployment.build_plan(before)})


def test_wrong_plan_never_executes_ddl(tmp_path):
    connection = MagicMock()
    with patch.object(deployment, "snapshot", return_value=state()), pytest.raises(ValueError):
        deployment.align(connection, apply=True, expected_plan_hash="stale", report_dir=tmp_path)
    connection.cursor.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("public", [True, False])
def test_rejects_public_or_symlink_report_directory(tmp_path, public):
    directory = tmp_path / "report"
    if public:
        directory.mkdir(mode=0o755)
    else:
        directory.symlink_to(tmp_path, target_is_directory=True)
    connection = MagicMock()
    before = state()
    with patch.object(deployment, "snapshot", return_value=before), pytest.raises(ValueError):
        deployment.align(connection, apply=True, expected_plan_hash=plan_hash(before), report_dir=directory)
    connection.cursor.assert_not_called()


def test_apply_changes_only_two_columns_and_audits_privately(tmp_path):
    tmp_path.chmod(0o700)
    before = state()
    connection = MagicMock()
    with patch.object(deployment, "snapshot", side_effect=[before, aligned(before)]):
        deployment.align(connection, apply=True, expected_plan_hash=plan_hash(before), report_dir=tmp_path)
    statements = [call.args[0] for call in connection.cursor.return_value.__enter__.return_value.execute.call_args_list]
    assert statements == ["ALTER SESSION SET DDL_LOCK_TIMEOUT=10", *deployment.build_plan(before)]
    assert json.loads((tmp_path / "completed.json").read_text())["verified"]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in tmp_path.iterdir())
    assert json.loads((tmp_path / "before.json").read_text()) == before


@pytest.mark.parametrize("field", ["target_fingerprint", "run_count", "run_ids_hash",
    "indexes", "index_columns", "constraints", "columns"])
def test_verify_detects_any_protected_change(field):
    before = state()
    after = aligned(before)
    after[field] = [] if isinstance(after[field], list) else "changed"
    with pytest.raises(ValueError):
        deployment.verify(before, after)


def test_failure_records_implicit_commit_without_claiming_rollback(tmp_path):
    tmp_path.chmod(0o700)
    before = state()
    after = aligned(before)
    after["run_ids_hash"] = "changed"
    with patch.object(deployment, "snapshot", side_effect=[before, after]), pytest.raises(ValueError):
        deployment.align(MagicMock(), apply=True, expected_plan_hash=plan_hash(before), report_dir=tmp_path)
    report = json.loads((tmp_path / "partial.json").read_text())
    assert report == {"statements": deployment.build_plan(before), "verified": False}
    assert not (tmp_path / "completed.json").exists()


def test_constraint_filter_does_not_remove_other_checks():
    assert not deployment.nullable_constraint(["C1", "C", '"AGENT_ID" IS NOT NULL OR "ACTOR_ID" IS NOT NULL'])
    assert not deployment.nullable_constraint(["FK1", "R", '"AGENT_ID" IS NOT NULL'])
    deployment.verify(state(), aligned(state()))


def test_already_aligned_apply_executes_no_table_ddl(tmp_path):
    tmp_path.chmod(0o700)
    before = aligned(state())
    connection = MagicMock()
    with patch.object(deployment, "snapshot", return_value=before):
        deployment.align(connection, apply=True, expected_plan_hash=plan_hash(before), report_dir=tmp_path)
    statements = [call.args[0] for call in connection.cursor.return_value.__enter__.return_value.execute.call_args_list]
    assert statements == ["ALTER SESSION SET DDL_LOCK_TIMEOUT=10"]
    assert json.loads((tmp_path / "completed.json").read_text())["statements"] == []


def test_ddl_failure_preserves_before_and_records_no_completed_ddl(tmp_path):
    tmp_path.chmod(0o700)
    before = state()
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value.execute.side_effect = [None, RuntimeError()]
    with patch.object(deployment, "snapshot", return_value=before), pytest.raises(RuntimeError):
        deployment.align(connection, apply=True, expected_plan_hash=plan_hash(before), report_dir=tmp_path)
    assert json.loads((tmp_path / "partial.json").read_text()) == {"statements": [], "verified": False}
    assert (tmp_path / "before.json").exists()


def test_snapshot_is_readonly_and_does_not_expose_schema_identity():
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.__iter__.side_effect = [iter([["db", "pdb", "owner", "owner"]]),
        iter(state()["columns"]), iter(state()["constraints"]), iter(state()["indexes"]),
        iter(state()["index_columns"]), iter([["01"], ["02"]])]
    result = deployment.snapshot(connection)
    assert result["run_count"] == 2
    assert "owner" not in json.dumps(result)
    assert all(call.args[0].startswith("SELECT ") for call in cursor.execute.call_args_list)


def test_snapshot_rejects_cross_schema_session():
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value.__iter__.return_value = iter([
        ["db", "pdb", "other", "owner"]])
    with pytest.raises(ValueError):
        deployment.snapshot(connection)
