"""KC 系统案例资源用例；只允许可信平台后台调用。"""

from platform_core.contracts.aiops.case_collection import CASE_MODEL_ROLES
from platform_core.contracts.aiops.case_resource import CaseResourceResult
from knowledge_core.domain.case_resource import CaseResourceConflict


class SystemCaseResourceService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def ensure(self, request):
        async with self._uow_factory() as uow:
            collection = await uow.case_resources.ensure(request)
            models = collection.models_json or {}
            missing = sorted(role for role in CASE_MODEL_ROLES if not models.get(role))
            policy = str((collection.parse_policy_json or {}).get("parse_strategy", "AUTO")).upper()
            result = CaseResourceResult(request=request, collection_id=collection.collection_id,
                missing_model_roles=missing, configuration_status=(
                    "TEXT_POLICY_REQUIRED" if policy not in {"AUTO", "TEXT"}
                    else "MODELS_REQUIRED" if missing else "CONFIGURED"))
            await uow.commit()
            return result
