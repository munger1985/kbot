"""图像模型 Provider 适配器。"""

from typing import Any

from model_serving.llm.responses.oci_adapter import OciGrokResponsesAdapter
from platform_core.contracts import ImageGenerationRequest, ImageGenerationResult
from platform_core.dictionary import ImageGenerationProvider


class ImageModelAdapter:
    """把独立图像 Provider 合同映射到已验证的 OCI 调用实现。"""

    def __init__(self, *, timeout_seconds: int = 600) -> None:
        self._oci = OciGrokResponsesAdapter(timeout_seconds=timeout_seconds)

    async def generate_image(
        self,
        request: ImageGenerationRequest,
        material: dict[str, Any],
    ) -> ImageGenerationResult:
        provider = str(material.get("provider") or "").strip().lower()
        if provider != ImageGenerationProvider.OCI.value:
            raise ValueError(f"图像生成进程不支持 Provider：{provider}")
        mapped = dict(material)
        mapped["provider"] = "oci"
        return await self._oci.generate_image(request, mapped)
