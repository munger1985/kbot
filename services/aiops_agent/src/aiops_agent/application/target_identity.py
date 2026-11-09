"""主动诊断 Run 入口共同冻结的 Target 身份，不包含连接信息或凭据。"""


def target_identity_snapshot(target) -> dict[str, str | int]:
    """身份在创建 Run 时确定，重试和重规划不得按当前配置刷新。"""
    snapshot = {
        "target_id": str(target.target_id),
        "row_version": int(target.row_version),
        "db_type": str(target.db_type),
    }
    name = getattr(target, "display_name", None)
    if name:
        snapshot["display_name"] = str(name)
    return snapshot


def target_answer_presentation(context) -> dict[str, str]:
    """回答只使用本轮冻结的目标名；内部标识保留用于关联，不作为正文对象名称。"""
    target = dict(context.plan_snapshot.get("target") or {})
    target_id = str(context.target_id)
    name = target.get("display_name") if str(target.get("target_id") or "") == target_id else None
    return {
        "target_id": target_id,
        "display_name": str(name) if name else "所选数据库",
        "policy": (
            "正文、标题及对象与时段使用 display_name，不输出 target UUID、内部资源或绑定标识，"
            "也不在名称后括号补充这些内部标识。名称未提供时使用所选数据库，不猜测名称。"
            "这是逻辑运维目标名，不替代证据中实际观测的数据库名、实例名、主机名或容器名；"
            "SQL ID、query_id、digest 等用户需要定位的业务诊断标识仍保留原值。"
        ),
    }
