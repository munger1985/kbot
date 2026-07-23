"""AIOps 用例层可稳定处理的持久化错误。"""


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
