"""Runtime 唯一证据模型请求适配器；发送正文/图像字节，不发送预览 URL。"""

import base64
import re
import time

from knowledge_algorithms.contracts.errors import require
from platform_clients.knowledge_computation import KnowledgeComputationClientError
from agent_runtime.application.knowledge_delivery import ModelEvidenceReply, EvidenceDeliveryError


class KnowledgeEvidenceModelGateway:
    def __init__(self, *, client, auth_context, model_snapshot_ref, model_payload, system_prompt,
                 supports_images, clock=time.time):
        require(model_snapshot_ref and system_prompt and isinstance(model_payload, dict),
                "Composer 缺少冻结模型和 Prompt 配置")
        require(set(model_payload) <= {"model_id", "served_model_name", "max_tokens", "temperature"}
                and (("model_id" in model_payload) != ("served_model_name" in model_payload))
                and type(model_payload.get("max_tokens")) is int and model_payload["max_tokens"] > 0,
                "输出上限必须来自 Model Serving 冻结配置")
        self.client, self.auth_context = client, auth_context
        self.snapshot, self.model_payload, self.prompt = model_snapshot_ref, dict(model_payload), system_prompt
        self.supports_images, self.clock = supports_images, clock

    def prepare(self, *, question, evidence_parts, model_snapshot_ref):
        require(model_snapshot_ref == self.snapshot, "Composer 模型快照发生改变")
        content = [{"type": "text", "text": question}]
        for part in evidence_parts:
            require(type(part["data"]) is bytes, "Composer 必须交付已经材料化的字节")
            content.append({"type": "text", "text": "[" + part["label"] + "]"})
            if part["kind"] == "TEXT":
                content.append({"type": "text", "text": part["data"].decode("utf-8")})
            else:
                require(self.supports_images and part["mime_type"] in {"image/png", "image/jpeg", "image/webp"},
                        "当前冻结模型不支持图像输入")
                encoded = base64.b64encode(part["data"]).decode("ascii")
                content.append({"type": "image_url", "image_url":
                    {"url": "data:" + part["mime_type"] + ";base64," + encoded}})
        payload = {**self.model_payload, "stream": False, "messages": [
            {"role": "system", "content": self.prompt}, {"role": "user", "content": content}]}
        return payload

    async def generate(self, *, question, evidence_parts, model_snapshot_ref):
        try:
            payload = self.prepare(question=question, evidence_parts=evidence_parts,
                                   model_snapshot_ref=model_snapshot_ref)
        except (ValueError, KeyError, TypeError) as exc:
            raise EvidenceDeliveryError("FAILED") from exc
        try:
            endpoint = "/internal/v1/inference" if self.supports_images else "/internal/v1/chat/completions"
            response = await self.client.json("POST", endpoint,
                                              auth_context=self.auth_context, payload=payload)
        except KnowledgeComputationClientError as exc:
            raise EvidenceDeliveryError("FAILED" if 400 <= exc.status_code < 500 else "UNKNOWN_DELIVERY") from exc
        # 收到成功响应只能证明请求已到服务；不宣称模型理解/使用了全部证据。
        sent_at = int(self.clock())
        try:
            body = response["choices"][0]["message"]["content"]
            require(type(body) is str, "回答正文格式无效")
        except (KeyError, IndexError, TypeError, ValueError):
            return ModelEvidenceReply("", (), sent_at)
        used = tuple(dict.fromkeys(re.findall(r"\[([A-Z]+[0-9]+)\]", body)))
        return ModelEvidenceReply(body, used, sent_at)
