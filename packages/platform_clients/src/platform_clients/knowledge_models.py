"""模型表示 HTTP 端口；App 负责模型授权和真实图像材料化。"""

from .knowledge_computation import KnowledgeComputationClient
from platform_core.security import build_internal_auth_headers


class NativeRepresentationClient(KnowledgeComputationClient):
    def __init__(self, *, max_response_bytes=67108864, **kwargs):
        if type(max_response_bytes) is not int or not 0 < max_response_bytes <= 134217728:
            raise ValueError("模型表示响应预算无效")
        super().__init__(**kwargs)
        self.max_response_bytes = max_response_bytes

    def _auth_headers(self, auth_context):
        """模型服务使用内部凭证及每次请求新签发的受众绑定 AuthContext。"""
        return build_internal_auth_headers(audience=self.audience, caller_service=self.caller_service,
                                           context=auth_context)

    async def represent(self, payload, *, auth_context):
        return await self.json("POST", "/internal/v1/represent", auth_context=auth_context, payload=payload,
                               max_bytes=self.max_response_bytes)
