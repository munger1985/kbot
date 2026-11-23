"""知识任务的受认证模型目录绑定与带版本推理传输。"""

from uuid import UUID

from platform_core.security import build_internal_auth_headers
from .knowledge_computation import KnowledgeComputationClient


class KnowledgeModelBindingClient(KnowledgeComputationClient):
    def _auth_headers(self, auth_context):
        context = auth_context.model_copy(update={"scopes": ("model.inference",), "calling_service": self.caller_service})
        return build_internal_auth_headers(audience=self.audience, caller_service=self.caller_service, context=context)

    async def binding(self, model_id, *, auth_context):
        return await self.json("GET", f"/internal/v1/models/{UUID(str(model_id))}/inference-binding",
                               auth_context=auth_context, max_bytes=4096)

    async def infer(self, *, payload, auth_context):
        return await self.json("POST", "/internal/v1/inference", auth_context=auth_context, payload=payload)


def build_binding_clients(*, caller, ocr, vlm):
    return {
        role: KnowledgeModelBindingClient(caller_service=caller, audience=endpoint.audience,
            base_url=endpoint.base_url, timeout_seconds=endpoint.timeout_seconds)
        for role, endpoint in (("ocr", ocr), ("vlm", vlm))
    }
