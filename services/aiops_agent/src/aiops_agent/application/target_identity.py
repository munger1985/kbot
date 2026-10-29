"""主动诊断 Run 入口共同冻结的 Target 身份，不包含连接信息或凭据。"""


def target_identity_snapshot(target) -> dict[str, str | int]:
    """身份在创建 Run 时确定，重试和重规划不得按当前配置刷新。"""
    return {
        "target_id": str(target.target_id),
        "row_version": int(target.row_version),
        "db_type": str(target.db_type),
    }
