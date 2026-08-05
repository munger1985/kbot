"""Knowledge Core 模型引用反查用例。"""

from collections.abc import Callable
from uuid import UUID

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgeCoreModelReferenceService:
    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

    async def list(self, *, model_id: UUID) -> list[dict]:
        async with self._uow_factory() as uow:
            if uow.model_references is None:
                raise RuntimeError("模型引用 Repository 未初始化")
            return await uow.model_references.list_by_model(model_id=model_id)
