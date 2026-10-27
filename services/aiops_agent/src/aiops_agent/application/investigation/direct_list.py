"""固定列表入口可选证据的确定性规划。"""

from platform_core.contracts.aiops import InvestigationAction, MeasurementSemantics

from aiops_agent.application.top_sql_window import TOP_SQL_REFERENCE_KIND, TOP_SQL_WINDOW_TOOL


_CUMULATIVE_TOOLS = {
    "POSTGRESQL": "db.postgresql.statement.top",
    "MYSQL": "db.mysql.statement.digest.top",
    "ORACLE": "db.sql.top_current",
}


def add_top_sql_reference(investigation, *, db_type: str, available_tools: tuple[dict, ...]):
    """仅在可用时冻结累计参考，运行时仅在窗口不可用时读取。"""
    tool_id = _CUMULATIVE_TOOLS.get(db_type)
    if not tool_id or tool_id not in {str(tool.get("tool_id")) for tool in available_tools}:
        return investigation, None
    window = next((action for action in investigation.plan.actions if action.tool_id == TOP_SQL_WINDOW_TOOL), None)
    if window is None:
        return investigation, None
    next_id = max((int(action.action_id[1:]) for action in investigation.plan.actions), default=0) + 1
    action = InvestigationAction(
        action_id=f"a{next_id}",
        question="窗口排名不可用时读取累计 Top SQL 作为参考",
        tool_id=tool_id,
        input={"limit": int(window.input["limit"])},
        expected_evidence_kind=TOP_SQL_REFERENCE_KIND,
        measurement_semantics=MeasurementSemantics.CUMULATIVE_SINCE_LOAD,
        depends_on=(window.action_id,),
        optional=True,
    )
    return investigation.model_copy(update={
        "plan": investigation.plan.model_copy(update={"actions": (*investigation.plan.actions, action)}),
    }), tool_id
