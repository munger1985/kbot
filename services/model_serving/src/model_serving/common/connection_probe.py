"""对已启用的模型执行一次受控推理，仅验证连通性与响应。"""

import base64
from functools import lru_cache
from io import BytesIO
from typing import Any
from uuid import UUID

from PIL import Image, ImageDraw, ImageFont

from platform_core.contracts import ImageGenerationRequest
from platform_core.dictionary import ModelCategory


MODEL_PROBE_TIMEOUT_SECONDS = 90
_PROBE_TEXT = "KBot model connection test"
_TEXT_PROMPT = "Reply with OK."


@lru_cache(maxsize=1)
def probe_image_base64() -> str:
    """生成测试用 PNG，不读取用户图片或发布测试产物。"""
    image = Image.new("RGB", (240, 160), "white")
    ImageDraw.Draw(image).text(
        (34, 52), "KBOT", fill="black", font=ImageFont.load_default(size=40),
    )
    with BytesIO() as stream:
        image.save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")


def _llm_has_output(response: Any) -> bool:
    payload = getattr(getattr(response, "data", None), "chat_response", response)
    if str(getattr(payload, "text", "") or "").strip():
        return True
    choices = getattr(payload, "choices", None) or ()
    for choice in choices:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            from model_serving.llm.model.oci_client import extract_oci_text_content

            content = extract_oci_text_content(content)
        if str(content or getattr(message, "reasoning_content", "") or "").strip():
            return True
    return False


async def probe_model(
    *, category: int, service: Any, model: dict[str, Any], model_id: UUID,
    trace_id: str,
) -> None:
    """复用类别对应的推理服务；测试输入和输出都不经管理接口传递。"""
    identity = str(model_id)
    if category == ModelCategory.LLM:
        result = await service.chat(
            identity, messages=[{"role": "user", "content": _TEXT_PROMPT}],
            stream=False, max_tokens=128, timeout=MODEL_PROBE_TIMEOUT_SECONDS,
        )
        usable = _llm_has_output(result)
    elif category == ModelCategory.TXT_EMBEDDING:
        result = await service.embed_texts(identity, texts=[_PROBE_TEXT], batch_size=1)
        usable = len(result.data) == 1 and bool(result.data[0].embedding)
    elif category == ModelCategory.IMG_EMBEDDING:
        directions = model["model_params"].get("retrieval_directions") or ()
        direction = next((value for value in directions if value.startswith("IMAGE_TO_")), None)
        if direction is not None:
            result = await service.represent(
                identity, input_kind="IMAGE", direction=direction, image_base64=probe_image_base64(),
            )
        elif directions:
            result = await service.represent(
                identity, input_kind="TEXT", direction=directions[0], text=_PROBE_TEXT,
            )
        else:
            raise ValueError("视觉模型没有登记检索方向")
        usable = bool(result["vectors"]) and any(result["mask"])
    elif category == ModelCategory.VLM:
        result = await service.inference(
            identity, messages=[{"role": "user", "content": [
                {"type": "text", "text": "Read the text in the image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{probe_image_base64()}",
                }},
            ]}], stream=False, max_tokens=128, timeout=MODEL_PROBE_TIMEOUT_SECONDS,
        )
        usable = bool(result["choices"][0]["message"]["content"].strip())
    elif category == ModelCategory.OCR:
        result = await service.infer(model_id=model_id, image_base64=probe_image_base64())
        usable = isinstance(result["text"], str)
    elif category == ModelCategory.IMAGE_GENERATION:
        result = await service.generate_image(ImageGenerationRequest(
            model_id=model_id, prompt="A plain blue square on a white background.",
            count=1, trace_id=trace_id,
        ))
        if result.error_code == "PROVIDER_TIMEOUT":
            raise TimeoutError
        usable = result.status == "COMPLETED" and bool(result.artifacts)
    else:
        raise ValueError("当前模型类别不支持测试")
    if not usable:
        raise ValueError("模型未返回有效的推理结果")
