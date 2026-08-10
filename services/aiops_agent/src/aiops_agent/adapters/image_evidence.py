"""对话图片证据的 OCR/VLM 模型客户端。"""

import aiohttp

from platform_clients.model import AIModelConfigClient
from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class ImageEvidenceModelClient:
    def __init__(self, *, caller_service: str, ocr_config, vlm_config):
        self._caller = caller_service
        self._ocr = ocr_config
        self._vlm = vlm_config
        self._catalogs = {
            "OCR": AIModelConfigClient(
                base_url=ocr_config.base_url,
                timeout=ocr_config.timeout_seconds,
                caller_service=caller_service,
                audience=ocr_config.audience,
            ),
            "VLM": AIModelConfigClient(
                base_url=vlm_config.base_url,
                timeout=vlm_config.timeout_seconds,
                caller_service=caller_service,
                audience=vlm_config.audience,
            ),
        }

    async def process(
        self,
        *,
        mode: str,
        model_id,
        mime_type: str,
        content_base64: str,
    ) -> dict:
        config = self._ocr if mode == "OCR" else self._vlm
        definition = await self._catalogs[mode].get_model(model_id)
        headers = {
            "Content-Type": "application/json",
            **build_internal_auth_headers(
                audience=config.audience,
                caller_service=self._caller,
            ),
        }
        if mode == "OCR":
            payload = {
                "model_id": str(model_id),
                "image_base64": content_base64,
                "mime_type": mime_type,
            }
        else:
            payload = {
                "served_model_name": definition["served_model_name"],
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "提取并解释运维截图中可见的事实、数值、错误和表格，不猜测不可见内容。",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{content_base64}"
                                },
                            },
                        ],
                    }
                ],
            }
        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{config.base_url.rstrip('/')}{INTERNAL_API_V1}/inference",
                headers=headers,
                json=payload,
            ) as response:
                body = await response.json(content_type=None)
                if response.status != 200:
                    raise RuntimeError(
                        f"{mode} 推理失败，HTTP {response.status}: {body}"
                    )
        if mode == "VLM":
            choices = body.get("choices") or []
            text = (
                str((choices[0].get("message") or {}).get("content") or "")
                if choices
                else ""
            )
            return {
                "text": text,
                "response": body,
                "model_revision": str(model_id),
                "provider": definition.get("provider"),
            }
        return body


__all__ = ["ImageEvidenceModelClient"]
