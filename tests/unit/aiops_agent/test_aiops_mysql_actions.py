"""MySQL受控Action目录、编译和严格渲染测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiops_agent.actions import (
    ActionCompilerRegistry,
    ActionRegistry,
    ActionRenderer,
)


def _template(action_id: str):
    return next(
        item for item in ActionRegistry.load().templates
        if item.definition.db_type == "MYSQL"
        and item.definition.action_template_id == action_id
    )


def _assessment(tool_id: str, columns: tuple[str, ...], row: tuple[object, ...]):
    return SimpleNamespace(evidence=(SimpleNamespace(
        trust_level="SOURCE_VERIFIED",
        tool_id=tool_id,
        columns=tuple({"name": name} for name in columns),
        rows=(row,),
        evidence_ref="evidence-1",
    ),))


@pytest.mark.parametrize(
    ("action_id", "parameters", "command"),
    (
        ("db.mysql.query.terminate", {"session_id": 42}, "KILL QUERY 42"),
        (
            "db.mysql.table.analyze",
            {"table_ref": {
                "schema": "app", "object_type": "TABLE",
                "object_name": "orders",
            }},
            "ANALYZE TABLE `app`.`orders`",
        ),
        (
            "db.mysql.event.enable",
            {"event_ref": {
                "schema": "app", "object_type": "EVENT",
                "object_name": "nightly_rollup",
            }},
            "ALTER EVENT `app`.`nightly_rollup` ENABLE",
        ),
        (
            "db.mysql.event.disable",
            {"event_ref": {
                "schema": "app", "object_type": "EVENT",
                "object_name": "nightly_rollup",
            }},
            "ALTER EVENT `app`.`nightly_rollup` DISABLE",
        ),
        (
            "db.mysql.variable.set_persist",
            {"parameter_name": "max_connections", "parameter_value": 500},
            "SET PERSIST max_connections = 500",
        ),
        (
            "db.mysql.replication.start",
            {"channel_name": "source_1"},
            "START REPLICA FOR CHANNEL 'source_1'",
        ),
        (
            "db.mysql.replication.stop",
            {"channel_name": "source_1"},
            "STOP REPLICA FOR CHANNEL 'source_1'",
        ),
    ),
)
def test_mysql_actions_render_only_catalog_commands(
    action_id: str, parameters: dict[str, object], command: str
) -> None:
    rendered = ActionRenderer().render(_template(action_id), parameters)
    assert rendered.command_text == command
    assert rendered.execution_mode == "EXECUTABLE_AFTER_APPROVAL"


def test_mysql_action_renderer_rejects_identifier_and_value_escape() -> None:
    with pytest.raises(ValueError):
        ActionRenderer().render(
            _template("db.mysql.table.analyze"),
            {"table_ref": {
                "schema": "app`; DROP TABLE users", "object_type": "TABLE",
                "object_name": "orders",
            }},
        )
    with pytest.raises(ValueError):
        ActionRenderer().render(
            _template("db.mysql.variable.set_persist"),
            {"parameter_name": "max_connections", "parameter_value": 1},
        )


def test_mysql_action_compilers_use_only_source_verified_facts() -> None:
    compiler = ActionCompilerRegistry()
    query = compiler.compile_turn(
        compiler_id="mysql-query-terminate.v1",
        assessment=_assessment(
            "db.session.current_sql",
            ("session_id", "digest"),
            (42, "A" * 64),
        ),
        db_type="MYSQL",
    )
    analyze = compiler.compile_turn(
        compiler_id="mysql-table-analyze.v1",
        assessment=_assessment(
            "db.mysql.statistics.health",
            (
                "table_schema", "table_name", "update_time",
                "indexes_without_cardinality",
            ),
            ("app", "orders", None, 2),
        ),
        db_type="MYSQL",
    )

    assert query is not None and query.parameters == {"session_id": 42}
    assert analyze is not None
    assert analyze.parameters["table_ref"]["object_name"] == "orders"
