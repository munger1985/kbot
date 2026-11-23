"""对本轮会话附件做有界 OCR；识别结果不是知识库证据。"""

import asyncio
import base64
from dataclasses import replace
import re

from aiohttp import ClientError
from loguru import logger

from knowledge_algorithms.contracts.errors import AlgorithmError, require
from platform_clients.knowledge_computation import KnowledgeComputationClientError


async def read_question_images(images, *, client, model_id, auth_context, run_id):
    results = list(images)
    failures = (AlgorithmError, ClientError, KnowledgeComputationClientError, TimeoutError)

    def warn(exc, index=None):
        code = getattr(exc, "code", "TIMEOUT" if isinstance(exc, TimeoutError) else "OCR_REQUEST_FAILED")
        logger.warning("提问图片 OCR 未完成 | run_id={} | stage=question_ocr | error_code={} | upstream_status={} | image_index={}",
                       run_id, getattr(code, "value", code), getattr(exc, "status_code", None), index)

    try:
        # 共享一个批次期限；OCR 无法完成时仍保留原图和已完成的识别结果。
        async with asyncio.timeout(20):
            binding = await client.binding(model_id, auth_context=auth_context)
            binding_revision = binding.get("revision") if isinstance(binding, dict) else None
            require(isinstance(binding_revision, str) and re.fullmatch(r"[a-f0-9]{64}", binding_revision)
                    and str(binding.get("model_id")) == str(model_id), "OCR 模型绑定响应无效", stage="question_ocr")
            for index, image in enumerate(images):
                try:
                    reply = await client.infer(auth_context=auth_context, payload={
                        "model_id": model_id, "expected_revision": binding_revision, "mime_type": image.mime_type,
                        "image_base64": base64.b64encode(image.data).decode("ascii"),
                    })
                    # 推理端按 expected_revision 校验配置；响应的 model_revision 是 OCR 引擎版本，不是配置指纹。
                    engine_revision = reply.get("model_revision") if isinstance(reply, dict) else None
                    require(isinstance(reply, dict) and str(reply.get("model_id")) == str(model_id)
                            and isinstance(engine_revision, str) and engine_revision.strip()
                            and isinstance(reply.get("text"), str), "OCR 识别响应身份或字段无效", stage="question_ocr")
                    text = reply["text"].replace("\x00", "").strip().encode("utf-8", errors="replace")
                    results[index] = replace(image, ocr_text=text[:8192].decode("utf-8", errors="ignore"),
                                             ocr_truncated=len(text) > 8192)
                except failures as exc:
                    warn(exc, index + 1)
                    if getattr(exc, "code", None) == "MODEL_REVISION_CHANGED":
                        break
    except failures as exc:
        warn(exc)
    logger.info("提问图片 OCR 处理结束 | run_id={} | stage=question_ocr | image_count={} | readable_count={}",
                run_id, len(images), sum(bool(image.ocr_text) for image in results))
    return tuple(results)
