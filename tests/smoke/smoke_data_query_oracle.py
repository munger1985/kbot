"""使用本地 Oracle 验证 Data Query Schema、连通性与受控只读查询。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

from sqlalchemy import delete, select, text, update


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_query.adapters.query_executor import DataSourceExecutorResolver  # noqa: E402
from data_query.entities import (  # noqa: E402
    DataQueryExecutionEntity,
    DataQueryRunEntity,
)
from data_query.persistence import create_data_query_uow_factory  # noqa: E402
from data_query.workers.query_runs import DataQueryWorkerService  # noqa: E402
from data_query.connectors.connection_tester import (  # noqa: E402
    test_data_source_connection,
)
from data_query.connectors.dialect_compiler import compile_dialect_query  # noqa: E402
from data_query.contracts import (  # noqa: E402
    DataQueryPlanV1,
    DataSourceConnectionTest,
    DataSourceCredentials,
    DataSourceEndpoint,
    DatasetDefinition,
    MeasureDefinition,
    PlanMeasure,
    SemanticModelDefinition,
)
from platform_core.config import get_settings  # noqa: E402
from platform_core.database.oracle import create_database_runtime  # noqa: E402
from platform_core.identity import uuid7  # noqa: E402
from platform_core.persistence.orm import BaseEntity  # noqa: E402
from tests.acceptance.check_oracle_schema import (  # noqa: E402
    SERVICE_TABLES,
    SERVICE_VIEWS,
)


async def smoke() -> None:
    """验证规范对象存在，并通过 Query Plan 完成一条真实只读查询。"""
    settings = get_settings()
    oracle = settings.database.oracle
    endpoint = DataSourceEndpoint(
        host=oracle.host,
        port=oracle.port,
        database=oracle.service_name,
        allowed_schemas=(oracle.username.upper(),),
        tls_enabled=False,
    )
    credentials = DataSourceCredentials(
        username=oracle.username,
        password=oracle.require_password(),
    )
    connection_result = await test_data_source_connection(
        command=DataSourceConnectionTest(
            source_type="ORACLE",
            endpoint=endpoint,
            credentials=credentials,
        )
    )
    assert connection_result.ok

    runtime = create_database_runtime(settings)
    try:
        async with runtime.engine.connect() as connection:
            tables = set(
                (await connection.execute(text(
                    "SELECT table_name FROM user_tables "
                    "WHERE table_name LIKE 'KBOT_DQ_%'"
                ))).scalars()
            )
            views = set(
                (await connection.execute(text(
                    "SELECT view_name FROM user_views "
                    "WHERE view_name LIKE 'KBOT_V_DQ_%'"
                ))).scalars()
            )
            database_columns = {
                table_name: set((await connection.execute(text(
                    "SELECT column_name FROM user_tab_columns "
                    "WHERE table_name=:table_name"
                ), {"table_name": table_name})).scalars())
                for table_name in SERVICE_TABLES["data_query"]
            }
        assert tables == SERVICE_TABLES["data_query"], (
            f"Data Query 表不一致：{sorted(tables)}"
        )
        assert views == SERVICE_VIEWS["data_query"], (
            f"Data Query 视图不一致：{sorted(views)}"
        )
        for table_name in SERVICE_TABLES["data_query"]:
            mapped_columns = {
                column.name.upper()
                for column in BaseEntity.metadata.tables[table_name].columns
            }
            assert mapped_columns == database_columns[table_name], (
                f"{table_name} ORM 列不一致："
                f"ORM={sorted(mapped_columns)}，"
                f"Oracle={sorted(database_columns[table_name])}"
            )
    finally:
        await runtime.close()

    semantic_model_id = uuid7()
    model = SemanticModelDefinition(
        datasets=(DatasetDefinition(
            name="domains",
            display_name="平台域",
            physical_schema=oracle.username.upper(),
            physical_object="KBOT_PLATFORM_DOMAIN",
        ),),
        measures=(MeasureDefinition(
            name="domain_count",
            display_name="平台域数量",
            dataset="domains",
            aggregation="COUNT",
            value_type="INTEGER",
        ),),
    )
    plan = DataQueryPlanV1(
        semantic_model_id=semantic_model_id,
        semantic_model_version=1,
        dataset="domains",
        measures=(PlanMeasure(
            name="domain_count",
            aggregation="COUNT",
        ),),
        limit=1,
    )
    compiled = compile_dialect_query(
        dialect="ORACLE",
        plan=plan,
        model=model,
        policy_max_limit=1,
    )
    result = await DataSourceExecutorResolver._execute_oracle(
        endpoint,
        credentials.username,
        credentials.password,
        {
            "statement_timeout_seconds": 10,
            "max_rows": 1,
            "max_result_bytes": 16_384,
        },
        compiled,
    )
    assert result.observed_row_count == 1
    assert int(next(iter(result.rows[0].values()))) >= 0

    # 在真实 Oracle 上验证过期租约可被接管，旧 Worker 不能覆盖新租约。
    runtime = create_database_runtime(settings)
    run_id = uuid7()
    execution_id = uuid7()
    try:
        async with runtime.session_factory() as session:
            domain_id = int((await session.execute(text(
                "SELECT MIN(domain_id) FROM KBOT_PLATFORM_DOMAIN"
            ))).scalar_one())
            session.add(DataQueryRunEntity(
                data_query_run_id=run_id,
                domain_id=domain_id,
                actor_id="data-query-smoke",
                trace_id=str(uuid7()),
                idempotency_key=str(uuid7()),
                request_fingerprint="a" * 64,
                original_question="租约接管验证",
                standalone_query="租约接管验证",
                status="QUEUED",
                plan_snapshot_json={},
                policy_snapshot_json={},
                semantic_model_snapshot_json={},
            ))
            await session.flush()
            session.add(DataQueryExecutionEntity(
                data_query_execution_id=execution_id,
                domain_id=domain_id,
                data_query_run_id=run_id,
                attempt_no=1,
                status="QUEUED",
                connector_type="ORACLE",
                connector_version="oracle-v1",
                query_plan_hash="b" * 64,
                compiled_query_hash="c" * 64,
                preflight_json={"readonly": True},
            ))
            await session.commit()
        uow_factory = create_data_query_uow_factory(runtime.session_factory)
        first_worker = DataQueryWorkerService(
            uow_factory=uow_factory,
            executor_resolver=object(),
            worker_id="smoke-worker-1",
            lease_seconds=30,
            result_availability_hours=1,
        )
        second_worker = DataQueryWorkerService(
            uow_factory=uow_factory,
            executor_resolver=object(),
            worker_id="smoke-worker-2",
            lease_seconds=30,
            result_availability_hours=1,
        )
        first_claim = await first_worker._claim_one()
        assert first_claim is not None
        async with runtime.session_factory() as session:
            await session.execute(
                update(DataQueryExecutionEntity)
                .where(DataQueryExecutionEntity.data_query_execution_id == execution_id)
                .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
            )
            await session.commit()
        second_claim = await second_worker._claim_one()
        assert second_claim is not None
        assert second_claim.lease_token != first_claim.lease_token
        await first_worker._complete_failure(
            claimed=first_claim,
            error_code="EXECUTION_FAILED",
        )
        async with runtime.session_factory() as session:
            execution = (await session.execute(
                select(DataQueryExecutionEntity).where(
                    DataQueryExecutionEntity.data_query_execution_id == execution_id
                )
            )).scalar_one()
            assert execution.status == "EXECUTING"
            assert execution.lease_owner == "smoke-worker-2"
    finally:
        async with runtime.session_factory() as session:
            await session.execute(delete(DataQueryExecutionEntity).where(
                DataQueryExecutionEntity.data_query_execution_id == execution_id
            ))
            await session.execute(delete(DataQueryRunEntity).where(
                DataQueryRunEntity.data_query_run_id == run_id
            ))
            await session.commit()
        await runtime.close()
    print(
        "Data Query Oracle Smoke 通过："
        f"{len(tables)} 张表及 ORM 映射，{len(views)} 个视图，"
        "连接测试成功，受控只读查询返回 1 行，租约接管通过"
    )


if __name__ == "__main__":
    asyncio.run(smoke())
