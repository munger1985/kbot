"""AIOps 用例层可稳定处理的持久化错误。"""

from __future__ import annotations

import re

from sqlalchemy.exc import DBAPIError, IntegrityError


_NON_RETRYABLE_ORACLE_CONTRACT_CODES = frozenset(
    {
        1,
        904,
        918,
        936,
        942,
        1400,
        1401,
        1407,
        1438,
        12899,
        2289,
        2290,
        2291,
        2292,
        4043,
        4063,
        4098,
    }
)


def _oracle_error_code(exc: BaseException) -> int | None:
    candidates: list[object] = [exc]
    if isinstance(exc, DBAPIError) and exc.orig is not None:
        candidates.insert(0, exc.orig)
    for candidate in candidates:
        code = getattr(candidate, "code", None)
        if isinstance(code, int):
            return code
        args = getattr(candidate, "args", ())
        if args:
            detail = args[0]
            detail_code = getattr(detail, "code", None)
            if isinstance(detail_code, int):
                return detail_code
        match = re.search(r"ORA-(\d{5})", str(candidate))
        if match is not None:
            return int(match.group(1))
    return None


def is_schema_or_integrity_error(exc: BaseException) -> bool:
    """识别不应通过重复业务调用恢复的 Schema 与完整性错误。"""
    pending: list[BaseException] = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, IntegrityError):
            return True
        if _oracle_error_code(current) in _NON_RETRYABLE_ORACLE_CONTRACT_CODES:
            return True
        for related in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
            getattr(current, "orig", None),
        ):
            if isinstance(related, BaseException):
                pending.append(related)
    return False


class AIOpsSchemaNotReadyError(RuntimeError):
    """Schema 契约未就绪，禁止进入任何付费模型调用。"""

    code = "AIOPS_SCHEMA_NOT_READY"
    retryable = False

    def __init__(self, checks: dict[str, str] | None) -> None:
        checks = checks or {}
        failed = sorted(
            name for name, status in checks.items() if status != "ok"
        )
        super().__init__(
            "AIOps Schema 未就绪：" + ",".join(failed or ["unknown"])
        )
        self.failed_checks = tuple(failed)


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
