"""对话图片证据的 OCR/VLM 模型客户端。"""

import aiohttp
from loguru import logger

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
        prompt_content: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict:
        config = self._ocr if mode == "OCR" else self._vlm
        try:
            definition = await self._catalogs[mode].get_model(model_id)
        except Exception as exc:
            logger.warning(
                "图片模型定义读取失败 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | error_type={}",
                run_id,
                agent_id,
                trace_id,
                mode,
                model_id,
                type(exc).__name__,
            )
            raise
        logger.info(
            "图片模型定义已读取 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | served_model_name={} | provider={}",
            run_id,
            agent_id,
            trace_id,
            mode,
            model_id,
            definition.get("served_model_name"),
            definition.get("provider"),
        )
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
            prompt = str(prompt_content or "").strip()
            if not prompt:
                raise ValueError("VLM 图片解析必须提供已登记 Prompt")
            payload = {
                "served_model_name": definition["served_model_name"],
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
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
        logger.info(
            "开始图片模型推理 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | served_model_name={}",
            run_id,
            agent_id,
            trace_id,
            mode,
            model_id,
            definition.get("served_model_name"),
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{config.base_url.rstrip('/')}{INTERNAL_API_V1}/inference",
                    headers=headers,
                    json=payload,
                ) as response:
                    body = await response.json(content_type=None)
                    if response.status != 200:
                        logger.warning(
                            "图片模型推理响应异常 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | http_status={}",
                            run_id,
                            agent_id,
                            trace_id,
                            mode,
                            model_id,
                            response.status,
                        )
                        raise RuntimeError(
                            f"{mode} 推理失败，HTTP {response.status}"
                        )
        except Exception as exc:
            logger.warning(
                "图片模型推理请求失败 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={} | error_type={}",
                run_id,
                agent_id,
                trace_id,
                mode,
                model_id,
                type(exc).__name__,
            )
            raise
        logger.info(
            "图片模型推理完成 | run_id={} | agent_id={} | trace_id={} | mode={} | model_id={}",
            run_id,
            agent_id,
            trace_id,
            mode,
            model_id,
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
