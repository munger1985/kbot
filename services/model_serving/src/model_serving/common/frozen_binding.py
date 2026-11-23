"""模型服务从实际驱动配置计算不可变调用身份；不输出连接信息或凭据。"""

from hashlib import sha256
import json
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from platform_core.contracts import INTERNAL_API_V1
from platform_core.exceptions import DatabaseException, DataNotFoundException
from platform_core.security import get_auth_context


class FrozenBinding(BaseModel):
    model_id: UUID
    revision: str
    inference_ref: str
    representation: dict | None = None


class FrozenModelConflict(ValueError):
    """当前启用配置已不是审批所冻结的版本。"""


def frozen_revision(data: dict) -> str:
    # 目录版本包含在摘要中；不把明文凭据放入摘要。
    spec = {key: value for key, value in data.items() if key != "api_key"}
    return sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def create_frozen_binding_router(pool_getter, *, representation_kind=None) -> APIRouter:
    router = APIRouter(prefix=INTERNAL_API_V1, tags=["Model Inference"])

    @router.get("/models/{model_id}/inference-binding", response_model=FrozenBinding, response_model_exclude_none=True)
    async def resolve_binding(model_id: UUID, request: Request):
        context = get_auth_context(request)
        if "model.inference" not in context.scopes:
            raise HTTPException(403, detail={"code": "MODEL_INFERENCE_SCOPE_REQUIRED"})
        try:
            return await pool_getter(request).inference_binding(model_id, representation_kind=representation_kind)
        except (DataNotFoundException, LookupError, ValueError):
            raise HTTPException(404, detail={"code": "MODEL_BINDING_UNAVAILABLE"}) from None
        except (DatabaseException, RuntimeError):
            raise HTTPException(503, detail={"code": "MODEL_BINDING_LOOKUP_FAILED"}) from None

    return router
