"""AIOps 用例层可稳定处理的持久化错误。"""


class AIOpsApplicationError(RuntimeError):
    """可安全映射到 HTTP 的 AIOps 用例错误。"""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def resource_not_found(resource_name: str) -> AIOpsApplicationError:
    """隐藏跨 Domain 与资源不存在的差异。"""
    return AIOpsApplicationError(
        code="OPS_RESOURCE_NOT_FOUND",
        message=f"{resource_name}不存在",
        status_code=404,
    )


def row_version_changed() -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code="OPS_ROW_VERSION_CHANGED",
        message="资源已被其他请求更新，请刷新后重试",
        status_code=412,
    )


def state_conflict(message: str) -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code="OPS_STATE_CONFLICT",
        message=message,
        status_code=409,
    )


def validation_failed(message: str) -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code="OPS_VALIDATION_FAILED",
        message=message,
        status_code=422,
    )


def dependency_unavailable(message: str) -> AIOpsApplicationError:
    return AIOpsApplicationError(
        code="OPS_DEPENDENCY_UNAVAILABLE",
        message=message,
        status_code=503,
        retryable=True,
    )


class AIOpsPersistenceError(RuntimeError):
    """持久化操作无法按契约完成。"""


class UnitOfWorkStateError(AIOpsPersistenceError):
    """UoW 生命周期或提交顺序不合法。"""


class StaleLeaseError(AIOpsPersistenceError):
    """Worker 使用了已失效、已接管或不匹配的租约。"""


class StateConflictError(AIOpsPersistenceError):
    """当前状态不允许请求的迁移。"""


class RowVersionChangedError(AIOpsPersistenceError):
    """聚合已被其他事务更新。"""
