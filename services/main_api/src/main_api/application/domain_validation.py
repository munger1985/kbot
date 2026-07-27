"""Portal 请求的 Domain 存在性与启用状态校验。"""

from collections.abc import Callable
from typing import Any


class DomainValidationService:
    def __init__(self, *, app_id: int, uow_factory: Callable[[], Any]):
        self._app_id = app_id
        self._uow_factory = uow_factory

    async def is_active(self, domain_id: str) -> bool:
        try:
            parsed_domain_id = int(domain_id)
        except (TypeError, ValueError):
            return False
        if parsed_domain_id <= 0 or str(parsed_domain_id) != domain_id:
            return False
        async with self._uow_factory() as uow:
            return await uow.domains.exists_active(
                app_id=self._app_id,
                domain_id=parsed_domain_id,
            )
