"""按需采样的目录准入和整轮预算，不扩大任意动态查询能力。"""

from platform_core.contracts.aiops.investigation import DiagnosticSampling


def validate_sampling_plan(actions):
    sampled = [action for action in actions if getattr(action, "sampling", None) is not None and not getattr(action, "deferred", False)]
    if len(sampled) > 1:
        raise ValueError("每轮最多一组按需数据库采样")
    for action in sampled:
        if not action.tool_id.startswith("db.") or action.tool_id.endswith("readonly_query"):
            raise ValueError("追加采样只适用于目录中允许的低成本原子数据库工具")
        if str(action.measurement_semantics) not in {"CURRENT_ACTIVITY", "CUMULATIVE_SINCE_LOAD"}:
            raise ValueError("追加采样必须保留当前活动或累计统计口径，不能标为历史窗口增量")


def freeze_sampling(action, definition, parameters):
    value = getattr(action, "sampling", None)
    if value is None:
        return None
    sampling = DiagnosticSampling.model_validate(value.model_dump(mode="python"))
    if definition.cost_level != "LOW":
        raise ValueError("该工具成本不允许追加采样")
    if int(parameters.get("limit", 1)) > 200:
        raise ValueError("按需采样每次最多请求 200 行")
    return sampling.model_dump(mode="json")
