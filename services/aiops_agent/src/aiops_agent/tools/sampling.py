"""按需采样的准入和整轮预算，不扩大动态查询的对象或函数权限。"""

from platform_core.contracts.aiops.investigation import DiagnosticSampling, DiagnosticObservationSchedule


def validate_sampling_plan(actions):
    sampled = [action for action in actions if getattr(action, "sampling", None) is not None and not getattr(action, "deferred", False)]
    if len(sampled) > 1:
        raise ValueError("每轮最多一组按需数据库采样")
    scheduled = [action for action in actions if getattr(action, "observation_schedule", None) is not None and not getattr(action, "deferred", False)]
    if len(sampled) + len(scheduled) > 1:
        raise ValueError("每轮短时采样与持久化未来观察共用一组额度")
    for action in (*sampled, *scheduled):
        if not action.tool_id.startswith("db."):
            raise ValueError("追加采样只适用于低成本目录工具或受控动态数据库查询")
        if str(action.measurement_semantics) not in {"CURRENT_ACTIVITY", "CUMULATIVE_SINCE_LOAD"}:
            raise ValueError("追加采样必须保留当前活动或累计统计口径，不能标为历史窗口增量")


def freeze_sampling(action, definition, parameters):
    value = getattr(action, "sampling", None)
    scheduled = getattr(action, "observation_schedule", None)
    if value is None and scheduled is None:
        return None
    if definition.cost_level != "LOW":
        raise ValueError("该工具成本不允许追加采样")
    if int(parameters.get("limit", 1)) > 200:
        raise ValueError("按需采样每次最多请求 200 行")
    return DiagnosticSampling.model_validate(value).model_dump(mode="json") if value is not None else None


def freeze_dynamic_sampling(action, validated):
    """动态采样仅收紧已通过 AST 策略的查询，不能批准高风险计划采集。"""
    value = getattr(action, "sampling", None)
    scheduled = getattr(action, "observation_schedule", None)
    if value is None and scheduled is None:
        return None
    validate_sampling_plan((action,))
    if getattr(validated, "explain", False) or getattr(validated, "execution_decision", "AUTO_EXECUTE") != "AUTO_EXECUTE":
        raise ValueError("追加采样不允许 EXPLAIN 语句或需要审批的查询")
    if validated.max_rows > 200:
        raise ValueError("按需动态采样每次最多请求 200 行")
    return DiagnosticSampling.model_validate(value).model_dump(mode="json") if value is not None else None


def freeze_observation_schedule(action):
    """只冻结明确目的的未来区间，真正调度须等待本次基线成功。"""
    value = getattr(action, "observation_schedule", None)
    if value is None:
        return None
    validate_sampling_plan((action,))
    return DiagnosticObservationSchedule.model_validate(value).model_dump(mode="json")
