"""模型表示 HTTP 端口；App 负责模型授权和真实图像材料化。"""

from .knowledge_computation import KnowledgeComputationClient


class NativeRepresentationClient(KnowledgeComputationClient):
    def __init__(self, *, max_response_bytes=67108864, **kwargs):
        if type(max_response_bytes) is not int or not 0 < max_response_bytes <= 134217728:
            raise ValueError("模型表示响应预算无效")
        super().__init__(**kwargs)
        self.max_response_bytes = max_response_bytes

    async def represent(self, payload, *, auth_context):
        return await self.json("POST", "/internal/v1/represent", auth_context=auth_context, payload=payload,
                               max_bytes=self.max_response_bytes)
