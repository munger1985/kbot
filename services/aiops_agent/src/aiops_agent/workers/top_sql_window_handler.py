"""读取冻结范围内的历史工作负载快照。"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from aiops_agent.application.top_sql_window import TOP_SQL_WINDOW_TOOL, validate_top_sql_window
from aiops_agent.workers.handlers import TaskExecutionContext


class TopSqlWindowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["TOP_SQL_WINDOW_RESULT.v1"] = "TOP_SQL_WINDOW_RESULT.v1"
    target_id: UUID
    result: dict[str, Any]


class TopSqlWindowHandler:
    def __init__(self, *, workload_service) -> None:
        self._service = workload_service

    async def execute(self, context: TaskExecutionContext) -> TopSqlWindowResult:
        execution = context.plan_snapshot["investigation_execution"]
        invocation = execution["direct_invocations"][context.task_key]
        tool = invocation["tool"]
        if tool["tool_id"] != TOP_SQL_WINDOW_TOOL:
            raise ValueError("冻结的 Top SQL 工具身份不匹配")
        parameters = validate_top_sql_window(dict(tool["parameters"]))
        database = execution["database"]
        result = await self._service.read_top_sql_window(
            domain_id=int(database["domain_id"]),
            target_id=context.target_id,
            period_start=datetime.fromisoformat(parameters["period_start"]),
            period_end=datetime.fromisoformat(parameters["period_end"]),
            limit=parameters["limit"],
        )
        return TopSqlWindowResult(target_id=context.target_id, result=result)
