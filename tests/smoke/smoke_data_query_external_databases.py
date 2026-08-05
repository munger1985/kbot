"""使用真实 PostgreSQL 和 MySQL 验证连接测试与受控只读查询。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from types import SimpleNamespace

import aiomysql
import asyncpg

from data_query.adapters.query_executor import DataSourceExecutorResolver
from data_query.connectors import compile_dialect_query
from data_query.connectors.connection_tester import test_data_source_connection
from data_query.connectors.postgresql import (
    PostgreSQLExecutionLimits,
    PostgreSQLReadOnlyExecutor,
    compile_postgresql_query,
)
from data_query.contracts import (
    DataQueryPlanV1,
    DataSourceConnectionTest,
    DataSourceCredentials,
    DataSourceEndpoint,
    DatasetDefinition,
    MeasureDefinition,
    PlanMeasure,
    SemanticModelDefinition,
)
from agent_runtime.runtime import ExecutionContext
from agent_runtime.specialists.data_query import (
    DataQuerySkill,
    MCPDataQueryExecutor,
    SemanticDataQueryExecutor,
)
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


TABLE_NAME = "kbot_dq_connector_smoke"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"真实数据库 Smoke 缺少环境变量：{name}")
    return value


def _model(schema: str) -> SemanticModelDefinition:
    return SemanticModelDefinition(
        datasets=(DatasetDefinition(
            name="smoke_rows",
            display_name="连接器测试数据",
            physical_schema=schema,
            physical_object=TABLE_NAME,
        ),),
        measures=(MeasureDefinition(
            name="row_count",
            display_name="记录数",
            dataset="smoke_rows",
            aggregation="COUNT",
            value_type="INTEGER",
        ),),
    )


def _plan() -> DataQueryPlanV1:
    return DataQueryPlanV1(
        semantic_model_id=uuid7(),
        semantic_model_version=1,
        dataset="smoke_rows",
        measures=(PlanMeasure(name="row_count", aggregation="COUNT"),),
        limit=1,
    )


async def _postgresql() -> str:
    host = os.getenv("KBOT_DQ_SMOKE_POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("KBOT_DQ_SMOKE_POSTGRES_PORT", "5432"))
    database = _required("KBOT_DQ_SMOKE_POSTGRES_DATABASE")
    username = _required("KBOT_DQ_SMOKE_POSTGRES_USERNAME")
    password = _required("KBOT_DQ_SMOKE_POSTGRES_PASSWORD")
    schema = os.getenv("KBOT_DQ_SMOKE_POSTGRES_SCHEMA", "public")
    endpoint = DataSourceEndpoint(
        host=host,
        port=port,
        database=database,
        allowed_schemas=(schema,),
        tls_enabled=False,
    )
    credentials = DataSourceCredentials(username=username, password=password)
    connection = await asyncpg.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    try:
        await connection.execute(
            f'DROP TABLE IF EXISTS "{schema}"."{TABLE_NAME}"'
        )
        await connection.execute(
            f'CREATE TABLE "{schema}"."{TABLE_NAME}" '
            "(row_id INTEGER PRIMARY KEY, label TEXT NOT NULL)"
        )
        await connection.executemany(
            f'INSERT INTO "{schema}"."{TABLE_NAME}" (row_id, label) '
            "VALUES ($1, $2)",
            ((1, "甲"), (2, "乙")),
        )
        checked = await test_data_source_connection(command=DataSourceConnectionTest(
            source_type="POSTGRESQL",
            endpoint=endpoint,
            credentials=credentials,
        ))
        assert checked.ok
        compiled = compile_postgresql_query(
            plan=_plan(), model=_model(schema), policy_max_limit=1,
        )
        @asynccontextmanager
        async def factory():
            item = await asyncpg.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
            )
            try:
                yield item
            finally:
                await item.close()

        query_result = await PostgreSQLReadOnlyExecutor(
            connection_factory=factory,
            limits=PostgreSQLExecutionLimits(
                statement_timeout_seconds=10,
                lock_timeout_seconds=5,
                max_rows=1,
                max_result_bytes=16_384,
                search_path=(schema,),
            ),
        ).execute(compiled)
        assert query_result.observed_row_count == 1
        assert int(query_result.rows[0]["row_count"]) == 2
        await _agent_semantic_contract(
            plan=_plan(),
            model=_model(schema),
            rows=query_result.rows,
            columns=query_result.columns,
        )
        return checked.database_version or "unknown"
    finally:
        await connection.execute(
            f'DROP TABLE IF EXISTS "{schema}"."{TABLE_NAME}"'
        )
        await connection.close()


async def _mysql() -> str:
    host = os.getenv("KBOT_DQ_SMOKE_MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("KBOT_DQ_SMOKE_MYSQL_PORT", "13306"))
    database = os.getenv("KBOT_DQ_SMOKE_MYSQL_DATABASE", "kbot_dq_smoke")
    username = os.getenv("KBOT_DQ_SMOKE_MYSQL_USERNAME", "root")
    password = _required("KBOT_DQ_SMOKE_MYSQL_PASSWORD")
    endpoint = DataSourceEndpoint(
        host=host,
        port=port,
        database=database,
        allowed_schemas=(database,),
        tls_enabled=False,
    )
    credentials = DataSourceCredentials(username=username, password=password)
    connection = await aiomysql.connect(
        host=host,
        port=port,
        db=database,
        user=username,
        password=password,
        autocommit=True,
    )
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`")
            await cursor.execute(
                f"CREATE TABLE `{TABLE_NAME}` "
                "(row_id INTEGER PRIMARY KEY, label VARCHAR(32) NOT NULL)"
            )
            await cursor.executemany(
                f"INSERT INTO `{TABLE_NAME}` (row_id, label) VALUES (%s, %s)",
                ((1, "甲"), (2, "乙")),
            )
        checked = await test_data_source_connection(command=DataSourceConnectionTest(
            source_type="MYSQL",
            endpoint=endpoint,
            credentials=credentials,
        ))
        assert checked.ok
        compiled = compile_dialect_query(
            dialect="MYSQL",
            plan=_plan(),
            model=_model(database),
            policy_max_limit=1,
        )
        query_result = await DataSourceExecutorResolver._execute_mysql(
            endpoint,
            username,
            password,
            {
                "statement_timeout_seconds": 10,
                "max_rows": 1,
                "max_result_bytes": 16_384,
            },
            compiled,
        )
        assert query_result.observed_row_count == 1
        assert int(query_result.rows[0]["row_count"]) == 2
        return checked.database_version or "unknown"
    finally:
        async with connection.cursor() as cursor:
            await cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`")
        connection.close()


async def _agent_semantic_contract(*, plan, model, rows, columns) -> None:
    """用真实查询结果验证 Agent Runtime 的 SEMANTIC 统一产物。"""
    run_id = uuid7()

    class _DataQueryClient:
        async def get_planning_context(self, **kwargs):
            del kwargs
            return {
                "models": [{
                    "semantic_model_id": str(plan.semantic_model_id),
                    "semantic_model_version": 1,
                    "display_name": "连接器测试模型",
                    "datasets": [
                        {"name": item.name, "display_name": item.display_name}
                        for item in model.datasets
                    ],
                    "dimensions": [],
                    "measures": [
                        {"name": item.name, "aggregation": item.aggregation}
                        for item in model.measures
                    ],
                    "max_rows": 1,
                }]
            }

        async def create_run(self, **kwargs):
            del kwargs
            return {"data_query_run_id": str(run_id)}

        async def get_run(self, **kwargs):
            del kwargs
            return {"status": "COMPLETED"}

        async def get_result(self, **kwargs):
            del kwargs
            return {
                "columns": [{"name": name} for name in columns],
                "preview_rows": list(rows),
                "row_count": len(rows),
                "observed_row_count": len(rows),
                "truncated": False,
                "provenance": {
                    "semantic_model_id": str(plan.semantic_model_id),
                },
            }

    class _ModelClient:
        async def get_llm_json(self, **kwargs):
            del kwargs
            return plan.model_dump(mode="json")

    class _PromptResolver:
        async def resolve(self, key):
            return SimpleNamespace(content=key)

    auth = AuthContext(
        principal_kind=PrincipalKind.SERVICE,
        client_id="data-query-smoke",
        calling_service="kbot-agent-runtime-worker",
        request_id="data-query-smoke",
        trace_id="data-query-smoke",
        domain_id="20",
        asserted_user_id="data-query-smoke",
    )
    context = ExecutionContext(
        domain_id=20,
        agent_id=uuid7(),
        run_id=uuid7(),
        task_id=uuid7(),
        task_key="data_query",
        actor_id="data-query-smoke",
        request_id="data-query-smoke",
        trace_id="data-query-smoke",
        original_input="共有多少条连接器测试数据？",
        policy_snapshot={"auth_context": auth.model_dump(mode="json")},
        config_snapshot={
            "agent": {
                "data_query_mode": "SEMANTIC",
                "models": {
                    "data_planner_llm": {
                        "served_model_name": "deterministic-smoke-planner"
                    }
                },
            }
        },
    )
    result = await DataQuerySkill(
        mcp_executor=MCPDataQueryExecutor(client=None),
        semantic_executor=SemanticDataQueryExecutor(
            client=_DataQueryClient(),
            model_client=_ModelClient(),
            prompt_resolver=_PromptResolver(),
        ),
    ).execute(context)
    assert result.artifact.schema_version == "QUERY_RESULT.v1"
    assert result.artifact.payload["provider"] == "SEMANTIC"
    assert int(result.artifact.payload["rows"][0]["row_count"]) == 2


async def main() -> None:
    postgresql_version = await _postgresql()
    mysql_version = await _mysql()
    print(
        "Data Query 外部数据库 Smoke 通过："
        f"PostgreSQL={postgresql_version}，MySQL={mysql_version}，"
        "连接测试、参数化只读查询与 Agent SEMANTIC 统一产物均成功"
    )


if __name__ == "__main__":
    asyncio.run(main())
