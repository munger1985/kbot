"""Runtime 唯一证据模型请求适配器；发送正文/图像字节，不发送预览 URL。"""

import base64
import json
import re
import time

import aiohttp
from loguru import logger
from knowledge_algorithms.contracts.errors import require
from platform_clients.knowledge_computation import KnowledgeComputationClientError
from agent_runtime.application.knowledge_delivery import ModelEvidenceReply, EvidenceDeliveryError


class KnowledgeEvidenceModelGateway:
    def __init__(self, *, client, auth_context, model_snapshot_ref, model_payload, system_prompt,
                 supports_images, clock=time.time):
        require(model_snapshot_ref and system_prompt and isinstance(model_payload, dict),
                "Composer 缺少冻结模型和 Prompt 配置")
        require(set(model_payload) <= {"model_id", "max_tokens", "temperature"}
                and "model_id" in model_payload
                and type(model_payload.get("max_tokens")) is int and model_payload["max_tokens"] > 0,
                "输出上限必须来自 Model Serving 冻结配置")
        self.client, self.auth_context = client, auth_context
        self.snapshot, self.model_payload, self.prompt = model_snapshot_ref, dict(model_payload), system_prompt
        self.supports_images, self.clock = supports_images, clock

    def prepare(self, *, question, evidence_parts, model_snapshot_ref, question_images=()):
        require(model_snapshot_ref == self.snapshot, "Composer 模型快照发生改变")
        content = [{"type": "text", "text": question}]
        prompt = self.prompt
        if question_images:
            prompt += ("\n本轮另有用户上传的提问图片，它们不是知识库引用。直接观察这些图片并结合用户问题回答；"
                       "来自用户图片的观察用自然语言说明来源，不生成引用标签。"
                       "只使用相关的知识库证据补充，凡使用库内证据须保留其 [D1] 等实际标签；"
                       "没有相关知识库证据时只回答图中可观察的内容，并明确不能从图中确定的事项。")
            for index, image in enumerate(question_images, 1):
                content.append({"type": "text", "text": f"用户上传的提问图片 {index}（不是知识库证据）"})
                content.append(self._image_part(image.data, image.mime_type))
                if image.ocr_text:
                    qualifier = "已截断" if image.ocr_truncated else "可能有误"
                    content.append({"type": "text", "text":
                        f"图片 {index} 的 OCR 识别文字（{qualifier}，只作辅助，以原图为准）：\n{image.ocr_text}"})
            if evidence_parts:
                content.append({"type": "text", "text": "以下是知识库检索证据，和用户上传的图片分别处理。"})
        for part in evidence_parts:
            require(type(part["data"]) is bytes, "Composer 必须交付已经材料化的字节")
            content.append({"type": "text", "text": "[" + part["label"] + "]"})
            if part["kind"] == "TEXT":
                content.append({"type": "text", "text": part["data"].decode("utf-8")})
            else:
                content.append(self._image_part(part["data"], part["mime_type"]))
        # VLM 要求所有角色的内容均为数组；纯文本 LLM 继续接收字符串。
        system_content = [{"type": "text", "text": prompt}] if self.supports_images else prompt
        user_content = content if self.supports_images else "\n\n".join(part["text"] for part in content)
        payload = {**self.model_payload, "stream": False, "messages": [
            {"role": "system", "content": system_content}, {"role": "user", "content": user_content}]}
        return payload

    def _image_part(self, data, mime_type):
        require(self.supports_images and type(data) is bytes and data
                and mime_type in {"image/png", "image/jpeg", "image/webp"}, "当前冻结模型不支持该图像输入")
        encoded = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": "data:" + mime_type + ";base64," + encoded}}

    async def generate(self, *, question, evidence_parts, model_snapshot_ref, question_images=(), on_preview=None):
        try:
            payload = self.prepare(question=question, evidence_parts=evidence_parts,
                                   model_snapshot_ref=model_snapshot_ref, question_images=question_images)
        except (ValueError, KeyError, TypeError) as exc:
            raise EvidenceDeliveryError("FAILED") from exc
        try:
            endpoint = "/internal/v1/inference" if self.supports_images else "/internal/v1/chat/completions"
            if on_preview is not None:
                body = await self._stream_answer(endpoint, {**payload, "stream": True}, on_preview)
                return self._reply(body, int(self.clock()))
            response = await self.client.json("POST", endpoint,
                                              auth_context=self.auth_context, payload=payload)
        except KnowledgeComputationClientError as exc:
            logger.warning("知识回答模型请求失败 | error_code={} | stage={} | upstream_status={} | question_image_count={}",
                           exc.code, exc.stage or "model_delivery", exc.status_code, len(question_images))
            raise EvidenceDeliveryError("FAILED" if 400 <= exc.status_code < 500 else "UNKNOWN_DELIVERY") from exc
        # 收到成功响应只能证明请求已到服务；不宣称模型理解/使用了全部证据。
        sent_at = int(self.clock())
        try:
            body = response["choices"][0]["message"]["content"]
            require(type(body) is str, "回答正文格式无效")
        except (KeyError, IndexError, TypeError, ValueError):
            return ModelEvidenceReply("", (), sent_at)
        return self._reply(body, sent_at)

    @staticmethod
    def _reply(body, sent_at):
        used = tuple(dict.fromkeys(re.findall(r"\[([A-Z]+[0-9]+)\]", body)))
        return ModelEvidenceReply(body, used, sent_at)

    async def _stream_answer(self, endpoint, payload, on_preview):
        """只转发正文增量；必须收到 Model Serving 的完整结束帧。"""
        parts, size, frame = [], 0, None
        try:
            async with self.client.response("POST", endpoint, auth_context=self.auth_context,
                                            payload=payload) as response:
                if response.content_type != "text/event-stream":
                    raise ValueError("非流式模型响应")
                # aiohttp 按行缓冲字节（含跨网络块的 UTF-8 字符），并限制单行长度。
                async for raw in response.content:
                    size += len(raw)
                    if size > 8 * 1024 * 1024:
                        raise ValueError("模型流超过大小限制")
                    line = raw.rstrip(b"\r\n")
                    if line.startswith(b":"):
                        continue
                    if line:
                        if frame is not None or not line.startswith(b"data:"):
                            raise ValueError("模型流帧格式无效")
                        frame = line[5:].strip()
                        continue
                    if frame is None:
                        continue
                    if frame == b"[DONE]":
                        body = "".join(parts)
                        if not body.strip():
                            raise ValueError("模型流没有回答正文")
                        return body
                    event = json.loads(frame)
                    frame = None
                    if not isinstance(event, dict) or "error" in event:
                        raise ValueError("模型流返回错误")
                    choices = event["choices"]
                    if choices == []:  # 用量统计不含正文。
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") not in {None, "stop", "length"}:
                        raise ValueError("模型回答未正常结束")
                    text = choice["delta"].get("content")
                    if text is not None and not isinstance(text, str):
                        raise ValueError("模型正文增量格式无效")
                    if text:
                        parts.append(text)
                        await on_preview(text)
                raise ValueError("模型流缺少结束帧")
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            logger.warning("知识回答模型流未完整交付 | error_type={}", type(exc).__name__)
            raise EvidenceDeliveryError("UNKNOWN_DELIVERY") from exc
