"""用户可见的 AIOps 连续诊断对话。"""

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.entities import EvidenceRequestEntity, ImageEvidenceProcessingEntity, OpsArtifactEntity, OpsConversationEntity, OpsConversationMessageEntity, OpsConversationRunEntity
from platform_core.identity import uuid7


class AIOpsConversationService:
    def __init__(self, *, uow_factory, image_model_client=None):
        self._uow_factory = uow_factory
        self._image_model_client = image_model_client

    async def create_or_append(self, *, domain_id: int, agent_id: UUID, actor_id: str, message: str, conversation_id: UUID | None = None) -> dict[str, Any]:
        message = message.strip()
        if not message: raise validation_failed("诊断消息不能为空")
        async with self._uow_factory() as uow:
            if conversation_id is None:
                agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
                if agent is None or agent.status != "ACTIVE" or agent.current_version_id is None:
                    raise resource_not_found("Active AIOps Agent")
                conversation = OpsConversationEntity(domain_id=domain_id, agent_id=agent_id, agent_version_id=agent.current_version_id, status="ACTIVE", source_type="CHAT", created_by=actor_id)
                await uow.conversations.add_conversation(conversation)
            else:
                conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id, lock=True)
                if conversation is None or conversation.created_by != actor_id: raise resource_not_found("Conversation")
                if conversation.agent_id != agent_id: raise state_conflict("Conversation Agent 不可变")
                conversation.status, conversation.updated_at = "ACTIVE", datetime.now(timezone.utc)
            sequence = await uow.conversations.next_message_sequence(conversation_id=conversation.conversation_id)
            user_message = OpsConversationMessageEntity(conversation_id=conversation.conversation_id, sequence_no=sequence, role="USER", message_type="USER_MESSAGE", payload_json={"text": message}, created_by=actor_id)
            progress = OpsConversationMessageEntity(conversation_id=conversation.conversation_id, sequence_no=sequence + 1, role="AGENT", message_type="AGENT_PROGRESS", payload_json={"summary": "正在判断取证范围与可用权限"})
            await uow.conversations.add_message(user_message); await uow.conversations.add_message(progress)
            await uow.commit()
            return self._view(conversation, (user_message, progress), ())

    async def list(self, *, domain_id: int, actor_id: str, agent_id: UUID | None = None, limit: int = 50):
        async with self._uow_factory() as uow:
            rows = await uow.conversations.list_conversations(domain_id=domain_id, created_by=actor_id, agent_id=agent_id, limit=limit)
            titles = await uow.conversations.first_user_messages(conversation_ids=[row.conversation_id for row in rows])
            return [self._summary(row, titles.get(row.conversation_id)) for row in rows]

    async def get(self, *, domain_id: int, conversation_id: UUID, actor_id: str):
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id)
            if row is None or row.created_by != actor_id: raise resource_not_found("Conversation")
            return self._view(row, await uow.conversations.list_messages(conversation_id=conversation_id), await uow.conversations.list_runs(conversation_id=conversation_id), await uow.conversations.list_action_steps(conversation_id=conversation_id))

    async def target_for(self, *, domain_id: int, agent_id: UUID) -> UUID:
        async with self._uow_factory() as uow:
            agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
            version = None if agent is None or agent.current_version_id is None else await uow.agents.version(agent_id=agent_id, agent_version_id=agent.current_version_id)
            if agent is None or agent.status != "ACTIVE" or version is None or version.target_id is None:
                raise validation_failed("AIOps Agent 必须启用并绑定诊断目标")
            return version.target_id

    async def attach_run(self, *, domain_id: int, conversation_id: UUID, ops_run_id: UUID, purpose: str) -> None:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id, lock=True)
            if conversation is None: raise resource_not_found("Conversation")
            await uow.conversations.add_run(OpsConversationRunEntity(conversation_id=conversation_id, ops_run_id=ops_run_id, purpose=purpose, sequence_no=await uow.conversations.next_run_sequence(conversation_id=conversation_id)))
            await uow.commit()

    async def request_evidence(self, *, domain_id: int, conversation_id: UUID, actor_id: str, purpose: str, suggested_sql: str | None = None):
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id, lock=True)
            if conversation is None: raise resource_not_found("Conversation")
            row = EvidenceRequestEntity(conversation_id=conversation_id, purpose=purpose, suggested_sql=suggested_sql, sql_hash=hashlib.sha256(suggested_sql.encode()).hexdigest() if suggested_sql else None, requested_by=actor_id)
            await uow.conversations.add_evidence_request(row); conversation.status = "WAITING_EVIDENCE"
            await uow.conversations.add_message(OpsConversationMessageEntity(conversation_id=conversation_id, sequence_no=await uow.conversations.next_message_sequence(conversation_id=conversation_id), role="AGENT", message_type="EVIDENCE_REQUEST", payload_json={"request_id": str(row.request_id), "purpose": purpose, "suggested_sql": suggested_sql}))
            await uow.commit(); return {"request_id": str(row.request_id), "status": row.status}

    async def submit_evidence_text(self, *, domain_id: int, conversation_id: UUID, request_id: UUID, actor_id: str, text: str, skipped: bool = False):
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id, lock=True)
            evidence = await uow.conversations.get_evidence_request(conversation_id=conversation_id, request_id=request_id, lock=True)
            if conversation is None or conversation.created_by != actor_id or evidence is None: raise resource_not_found("EvidenceRequest")
            if evidence.status != "OPEN": raise state_conflict("EvidenceRequest 不再接受输入")
            evidence.status, conversation.status = ("SKIPPED" if skipped else "RECEIVED"), "ACTIVE"
            message = OpsConversationMessageEntity(conversation_id=conversation_id, sequence_no=await uow.conversations.next_message_sequence(conversation_id=conversation_id), role="USER", message_type="EVIDENCE_SKIPPED" if skipped else "EVIDENCE_TEXT", payload_json={"request_id": str(request_id), "text": text}, created_by=actor_id)
            await uow.conversations.add_message(message); await uow.commit()
            return self._view(conversation, (message,), ())

    async def upload_evidence_file(self, *, domain_id: int, conversation_id: UUID, request_id: UUID, actor_id: str, filename: str, mime_type: str, content_base64: str, text: str | None = None):
        try: content = base64.b64decode(content_base64, validate=True)
        except ValueError as exc: raise validation_failed("证据文件不是有效 Base64 内容") from exc
        if not content or len(content) > 10 * 1024 * 1024: raise validation_failed("证据文件必须介于 1 字节和 10 MiB 之间")
        if not (
            mime_type.startswith("image/")
            or mime_type
            in {"text/plain", "text/csv", "text/html", "application/csv"}
        ):
            raise validation_failed("仅支持图片、文本、CSV 或 HTML 证据")
        processing_id = None
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id)
            runs = await uow.conversations.list_runs(conversation_id=conversation_id)
            evidence = await uow.conversations.get_evidence_request(conversation_id=conversation_id, request_id=request_id, lock=True)
            if conversation is None or conversation.created_by != actor_id or not runs or evidence is None: raise resource_not_found("Conversation Evidence")
            if evidence.status != "OPEN": raise state_conflict("EvidenceRequest 不再接受输入")
            digest = hashlib.sha256(content).hexdigest()
            artifact = await uow.runs.add_artifact(OpsArtifactEntity(ops_run_id=runs[-1].ops_run_id, artifact_key=f"conversation-evidence:{conversation_id}:{digest}", artifact_type="USER_EVIDENCE_FILE", schema_version="USER_EVIDENCE_FILE.v1", payload_json={"filename": filename, "mime_type": mime_type, "content_base64": content_base64}, content_hash=digest, byte_size=len(content), provenance_json={"producer": "aiops.conversation", "submitted_by": actor_id}, trust_level="USER_PROVIDED", security_level=2))
            selected_mode = None
            if mime_type.startswith("image/"):
                version = await uow.agents.version(
                    agent_id=conversation.agent_id,
                    agent_version_id=conversation.agent_version_id,
                )
                if version is None:
                    raise resource_not_found("AgentVersion")
                selected_mode = self._image_processing_mode(
                    strategy=str(
                        (version.config_json or {}).get(
                            "image_processing_strategy"
                        )
                        or "AUTO"
                    ),
                    mime_type=mime_type,
                )
                capability = dict(
                    (version.image_capabilities_json or {}).get(
                        selected_mode.lower()
                    )
                    or {}
                )
                model_id = capability.get("default_model_id")
                if not model_id:
                    raise validation_failed(
                        f"当前 Agent 未配置 {selected_mode}，无法处理诊断图片"
                    )
                processing_id = uuid7()
                await uow.conversations.add_image_processing(
                    ImageEvidenceProcessingEntity(
                        processing_id=processing_id,
                        conversation_id=conversation_id,
                        evidence_request_id=request_id,
                        source_artifact_id=artifact.artifact_id,
                        processing_mode=selected_mode,
                        model_id=UUID(str(model_id)),
                        model_revision=str(model_id),
                        input_hash=artifact.content_hash,
                        status="PENDING",
                        created_by=actor_id,
                    )
                )
            evidence.status, conversation.status = "RECEIVED", "ACTIVE"
            message = OpsConversationMessageEntity(conversation_id=conversation_id, sequence_no=await uow.conversations.next_message_sequence(conversation_id=conversation_id), role="USER", message_type="EVIDENCE_FILE", payload_json={"request_id": str(request_id), "filename": filename, "mime_type": mime_type, "processing_mode": selected_mode, "text": (text or "").strip() or None}, artifact_id=artifact.artifact_id, created_by=actor_id)
            await uow.conversations.add_message(message); await uow.commit()
            result = {**self._view(conversation, (message,), ()), "artifact_id": str(artifact.artifact_id), "image_processing_id": str(processing_id) if processing_id else None}
        if processing_id is not None:
            await self.process_image_evidence(processing_id=processing_id)
        return result

    async def process_image_evidence(self, *, processing_id: UUID) -> None:
        if self._image_model_client is None:
            raise state_conflict("图片处理模型服务未配置")
        async with self._uow_factory() as uow:
            processing = await uow.conversations.get_image_processing(
                processing_id=processing_id
            )
            source = (
                await uow.runs.get_artifact(
                    artifact_id=processing.source_artifact_id
                )
                if processing
                else None
            )
            if processing is None or source is None:
                raise resource_not_found("ImageEvidenceProcessing")
            source_payload = dict(source.payload_json or {})
            content_base64 = str(source_payload.get("content_base64") or "")
            mime_type = str(source_payload.get("mime_type") or "image/png")
        try:
            output = await self._image_model_client.process(
                mode=processing.processing_mode,
                model_id=processing.model_id,
                mime_type=mime_type,
                content_base64=content_base64,
            )
        except Exception as exc:
            async with self._uow_factory() as uow:
                row = await uow.conversations.get_image_processing(
                    processing_id=processing_id, lock=True
                )
                if row is not None:
                    row.status = "FAILED"
                    row.error_code = "IMAGE_MODEL_INFERENCE_FAILED"
                    row.error_message = str(exc)[:2000]
                    await uow.commit()
            raise state_conflict(
                "图片 OCR/VLM 处理失败，请更换模型或重新上传"
            ) from exc
        async with self._uow_factory() as uow:
            row = await uow.conversations.get_image_processing(
                processing_id=processing_id, lock=True
            )
            if row is None or row.status != "PENDING":
                raise state_conflict("图片处理状态已变化")
            payload = {
                "processing_id": str(processing_id),
                "mode": row.processing_mode,
                "model_id": str(row.model_id),
                **output,
            }
            encoded = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            output_artifact = await uow.runs.add_artifact(
                OpsArtifactEntity(
                    ops_run_id=source.ops_run_id,
                    artifact_key=f"image-evidence-output:{processing_id}",
                    artifact_type="IMAGE_EVIDENCE_EXTRACTION",
                    schema_version="IMAGE_EVIDENCE_EXTRACTION.v1",
                    payload_json=payload,
                    content_hash=hashlib.sha256(encoded).hexdigest(),
                    byte_size=len(encoded),
                    provenance_json={
                        "producer": "model-serving",
                        "mode": row.processing_mode,
                        "source_artifact_id": str(source.artifact_id),
                    },
                    trust_level="MODEL_DERIVED",
                    security_level=source.security_level,
                )
            )
            row.status = "SUCCEEDED"
            row.output_artifact_id = output_artifact.artifact_id
            await uow.conversations.add_message(
                OpsConversationMessageEntity(
                    conversation_id=row.conversation_id,
                    sequence_no=await uow.conversations.next_message_sequence(
                        conversation_id=row.conversation_id
                    ),
                    role="AGENT",
                    message_type="IMAGE_EVIDENCE_PROCESSED",
                    payload_json={
                        "processing_id": str(processing_id),
                        "mode": row.processing_mode,
                        "model_id": str(row.model_id),
                        "text": str(output.get("text") or ""),
                    },
                    artifact_id=output_artifact.artifact_id,
                )
            )
            await uow.commit()

    @staticmethod
    def _image_processing_mode(*, strategy: str, mime_type: str) -> str:
        normalized = strategy.strip().upper()
        if normalized == "OCR_FIRST":
            return "OCR"
        if normalized == "VLM_FIRST":
            return "VLM"
        if mime_type in {"image/tiff", "image/bmp"}:
            return "OCR"
        return "VLM"

    async def resolve_action_step(self, *, domain_id: int, conversation_id: UUID, action_step_id: UUID, expected_row_version: int, expected_sql_hash: str) -> UUID:
        async with self._uow_factory() as uow:
            conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id)
            step = await uow.conversations.get_action_step(conversation_id=conversation_id, action_step_id=action_step_id)
            if conversation is None or step is None or step.proposal_id is None: raise resource_not_found("ActionStep")
            if int(step.row_version) != expected_row_version or step.sql_hash != expected_sql_hash: raise state_conflict("动作步骤版本或 SQL Hash 已变化")
            return step.proposal_id

    @staticmethod
    def _summary(row, first=None):
        title = str((first.payload_json or {}).get("text") or "诊断对话") if first else "诊断对话"
        return {"conversation_id": str(row.conversation_id), "agent_id": str(row.agent_id), "status": row.status, "source_type": row.source_type, "title": f"{title[:48].rstrip()}…" if len(title) > 48 else title, "created_at": row.created_at.isoformat() if row.created_at else None, "updated_at": row.updated_at.isoformat() if row.updated_at else None}

    @staticmethod
    def _view(row, messages, runs, action_steps=()):
        return {"conversation_id": str(row.conversation_id), "agent_id": str(row.agent_id), "agent_version_id": str(row.agent_version_id), "status": row.status, "row_version": int(row.row_version), "messages": [{"message_id": str(item.message_id), "sequence_no": int(item.sequence_no), "role": item.role, "message_type": item.message_type, "payload": dict(item.payload_json), "artifact_id": str(item.artifact_id) if item.artifact_id else None, "created_at": item.created_at.isoformat() if item.created_at else None} for item in messages], "runs": [{"ops_run_id": str(item.ops_run_id), "purpose": item.purpose, "sequence_no": int(item.sequence_no)} for item in runs], "action_steps": [{"action_step_id": str(item.action_step_id), "proposal_id": str(item.proposal_id) if item.proposal_id else None, "ordinal": int(item.ordinal), "sql_hash": item.sql_hash, "status": item.status, "row_version": int(item.row_version)} for item in action_steps]}
