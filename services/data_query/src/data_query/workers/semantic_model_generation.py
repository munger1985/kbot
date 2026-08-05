"""持久化语义模型生成作业 Worker。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta

from loguru import logger

from data_query.application.schema_metadata import enrich_semantic_candidate, generate_semantic_candidate
from data_query.application.notifications import publish_data_query_notification
from data_query.contracts import SemanticModelCandidateRequest
from data_query.entities import SemanticModelEntity, SemanticModelVersionEntity
from platform_core.identity import uuid7


class SemanticModelGenerationWorker:
    def __init__(
        self,
        *,
        uow_factory,
        model_config_client,
        model_client,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        self._uow_factory = uow_factory
        self._model_config_client = model_config_client
        self._model_client = model_client
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds

    async def process_one(self) -> bool:
        async with self._uow_factory() as uow:
            assert uow.semantic_model_generation_jobs
            now = datetime.now(UTC)
            lease_token = uuid7()
            job = await uow.semantic_model_generation_jobs.claim_next(
                worker_id=self._worker_id,
                lease_token=lease_token,
                now=now,
                lease_until=now + timedelta(seconds=self._lease_seconds),
            )
            if job is None:
                await uow.commit()
                return False
            command = SemanticModelCandidateRequest.model_validate(job.request_json)
            job_id, domain_id, snapshot_id, actor_id = (
                job.generation_job_id, job.domain_id,
                job.schema_snapshot_id, job.requested_by,
            )
            await uow.commit()

        try:
            candidate = await self._with_heartbeat(
                job_id,
                lease_token,
                self._generate_candidate(
                    domain_id=domain_id,
                    snapshot_id=snapshot_id,
                    command=command,
                ),
            )
            await self._complete(
                job_id=job_id, domain_id=domain_id, actor_id=actor_id,
                command=command, candidate=candidate, lease_token=lease_token,
            )
        except Exception as exc:
            try:
                await self._fail(
                    job_id=job_id, domain_id=domain_id, actor_id=actor_id,
                    display_name=command.display_name, error=exc,
                    lease_token=lease_token,
                )
            except Exception:
                # 保留 RUNNING 状态供 stale recovery 再次领取，不能让整个 Worker 循环退出。
                logger.exception("语义模型生成失败状态持久化失败 | job_id={}", job_id)
        return True

    async def _generate_candidate(self, *, domain_id, snapshot_id, command):
        candidate = await generate_semantic_candidate(
            uow_factory=self._uow_factory,
            domain_id=domain_id,
            snapshot_id=snapshot_id,
            command=command,
        )
        if command.ai_model_id is not None and command.allow_ai_metadata:
            candidate = await enrich_semantic_candidate(
                candidate=candidate,
                command=command,
                uow_factory=self._uow_factory,
                model_config_client=self._model_config_client,
                model_client=self._model_client,
            )
        return candidate

    async def _complete(self, *, job_id, domain_id, actor_id, command, candidate, lease_token) -> None:
        async with self._uow_factory() as uow:
            assert uow.semantic_model_generation_jobs and uow.semantic_models
            assert uow.semantic_model_versions
            job = await uow.semantic_model_generation_jobs.get_by_id(
                generation_job_id=job_id, lock=True,
            )
            if (
                job is None
                or job.status != "RUNNING"
                or job.lease_owner != self._worker_id
                or job.lease_token != lease_token
            ):
                await uow.commit()
                return
            model = SemanticModelEntity(
                semantic_model_id=job_id, domain_id=domain_id,
                display_name=command.display_name, description=command.description,
                created_by=actor_id, updated_by=actor_id,
            )
            await uow.semantic_models.add(model)
            definition = candidate.definition.model_dump(mode="json")
            version = SemanticModelVersionEntity(
                semantic_model_id=model.semantic_model_id, version_no=1,
                data_source_id=candidate.data_source_id,
                schema_snapshot_id=candidate.schema_snapshot_id, status="DRAFT",
                definition_json=definition,
                definition_hash=hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            )
            await uow.semantic_model_versions.add(version)
            job.status = "SUCCEEDED"
            job.lease_owner = None
            job.lease_token = None
            job.lease_until = None
            job.semantic_model_id = model.semantic_model_id
            job.semantic_model_version_id = version.semantic_model_version_id
            job.completed_at = datetime.now(UTC)
            await publish_data_query_notification(
                uow=uow,
                event_type="data_query.semantic_model.generation_completed",
                event_key=f"{job_id}:generation-completed",
                domain_id=int(domain_id), actor_id=actor_id,
                resource_type="semantic_model",
                resource_id=str(model.semantic_model_id),
                resource_name=command.display_name,
                correlation_id=str(job_id), operation_id=str(job_id),
                summary="语义模型草稿已生成。",
                safe_data={"status": "SUCCEEDED"},
            )
            await uow.commit()

    async def _fail(self, *, job_id, domain_id, actor_id, display_name, error: Exception, lease_token) -> None:
        message = str(error).strip()
        code = (
            message if message and message.replace("_", "").isalnum() and message.upper() == message
            else type(error).__name__
        )[:128]
        logger.exception("语义模型生成作业失败 | job_id={} | error_code={}", job_id, code)
        async with self._uow_factory() as uow:
            assert uow.semantic_model_generation_jobs
            job = await uow.semantic_model_generation_jobs.get_by_id(
                generation_job_id=job_id, lock=True,
            )
            if (
                job is None
                or job.status != "RUNNING"
                or job.lease_owner != self._worker_id
                or job.lease_token != lease_token
            ):
                await uow.commit()
                return
            job.status, job.error_code, job.completed_at = "FAILED", code, datetime.now(UTC)
            job.lease_owner = None
            job.lease_token = None
            job.lease_until = None
            await publish_data_query_notification(
                uow=uow,
                event_type="data_query.semantic_model.generation_failed",
                event_key=f"{job_id}:generation-failed",
                domain_id=int(domain_id), actor_id=actor_id,
                resource_type="semantic_model_generation",
                resource_id=str(job_id), resource_name=display_name,
                correlation_id=str(job_id), operation_id=str(job_id),
                summary="语义模型生成失败。",
                safe_data={"status": "FAILED", "error_code": code},
            )
            await uow.commit()

    async def _with_heartbeat(self, job_id, lease_token, operation):
        """生成期间续租，失去租约时禁止旧 Worker 写入终态。"""
        task = asyncio.create_task(operation)
        interval = max(1.0, min(self._lease_seconds / 3, 30.0))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=interval)
                if done:
                    return task.result()
                now = datetime.now(UTC)
                async with self._uow_factory() as uow:
                    assert uow.semantic_model_generation_jobs
                    owned = await uow.semantic_model_generation_jobs.heartbeat(
                        generation_job_id=job_id,
                        worker_id=self._worker_id,
                        lease_token=lease_token,
                        now=now,
                        lease_until=now + timedelta(seconds=self._lease_seconds),
                    )
                    await uow.commit()
                if not owned:
                    task.cancel()
                    raise RuntimeError("SEMANTIC_MODEL_GENERATION_LEASE_LOST")
        finally:
            if not task.done():
                task.cancel()
