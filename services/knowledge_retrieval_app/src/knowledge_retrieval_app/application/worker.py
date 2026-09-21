"""知识检索应用 X Search Worker。"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from loguru import logger

from knowledge_retrieval_app.application.runtime import (
    RESEARCH_KIND,
    TERMINAL_STATUSES,
    already_attempted,
    append_event,
    mark_upstream_attempted,
)
from knowledge_retrieval_app.entities import KnowledgeRetrievalXSourceEntity
from platform_clients.generative import GenerativeResponsesClientError
from platform_core.contracts import ResearchRequest
from platform_core.identity import uuid7


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: UUID
    domain_id: int
    kind: str
    status: str
    actor_id: str
    model_id: UUID
    request_json: dict[str, Any]
    result_json: dict[str, Any] | None
    provider_request_id: str | None
    lease_token: UUID
    trace_id: str


class KnowledgeRetrievalResearchWorker:
    def __init__(
        self,
        *,
        uow_factory,
        generative_client,
        poll_seconds: float = 5,
        lease_seconds: int = 360,
        worker_id: str | None = None,
    ):
        self._uow_factory = uow_factory
        self._generative = generative_client
        self._poll_seconds = poll_seconds
        self._lease_seconds = lease_seconds
        self._worker_id = worker_id or f"{socket.gethostname()}:{uuid7()}"

    async def run_forever(self) -> None:
        logger.info("知识检索 X Search Worker 开始运行：{}", self._worker_id)
        while True:
            handled = await self.process_once()
            if not handled:
                await asyncio.sleep(self._poll_seconds)

    async def process_once(self) -> bool:
        snapshot = await self._claim()
        if snapshot is None:
            return False
        try:
            await self._process(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("X Search Run 处理失败：run_id={}", snapshot.run_id)
            await self._fail(snapshot, code="PROVIDER_UNAVAILABLE", message="处理 Run 时发生未预期错误")
        return True

    async def _claim(self) -> RunSnapshot | None:
        lease_until = datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            row = await uow.runs.claim(worker_id=self._worker_id, lease_until=lease_until)
            snapshot = self._snapshot(row) if row is not None else None
            await uow.commit()
            return snapshot

    async def _process(self, snapshot: RunSnapshot) -> None:
        async with self._uow_factory() as uow:
            row = await self._locked(uow, snapshot)
            if row is None or row.status in TERMINAL_STATUSES:
                return
            if already_attempted(row):
                row.error_code = "PROVIDER_UNAVAILABLE"
                row.error_message = "检测到未完成的上游尝试，已阻止二次调用"
                await append_event(uow, row, stage="FAILED", message=row.error_message)
                await uow.commit()
                return
            mark_upstream_attempted(row)
            await append_event(uow, row, stage="SEARCHING", message="正在调用模型服务")
            await uow.commit()
        try:
            result = await self._generative.research(self._research_request(snapshot))
            await self._complete(snapshot, result)
        except GenerativeResponsesClientError as exc:
            await self._fail(snapshot, code=exc.code, message=exc.message)

    async def _complete(self, snapshot: RunSnapshot, result) -> None:
        async with self._uow_factory() as uow:
            row = await self._locked(uow, snapshot)
            if row is None:
                return
            row.provider_request_id = result.provider_request_id
            row.usage_json = result.usage.model_dump(mode="json") if result.usage else None
            if result.status != "COMPLETED":
                row.error_code = result.error_code or "PROVIDER_UNAVAILABLE"
                row.error_message = result.error_code or "X Search 未完成"
                await append_event(uow, row, stage="FAILED", message=row.error_message)
                await uow.commit()
                return
            await append_event(uow, row, stage="ORGANIZING_SOURCES", message="正在整理来源")
            assert uow.x_sources is not None
            now = datetime.now(timezone.utc)
            for index, citation in enumerate(result.citations, start=1):
                await uow.x_sources.add(KnowledgeRetrievalXSourceEntity(
                    source_id=uuid7(), run_id=row.run_id, domain_id=int(row.domain_id),
                    display_order=index, citation_label=f"[X{index}]",
                    provider_citation_id=citation.provider_citation_id,
                    canonical_url=citation.canonical_url, title=citation.title,
                    excerpt=citation.excerpt, author_handle=citation.author_handle,
                    published_at=citation.published_at, retrieved_at=now,
                ))
            await append_event(uow, row, stage="COMPOSING", message="正在生成结论")
            row.result_json = {"answer": result.answer, "citation_count": len(result.citations)}
            await append_event(uow, row, stage="COMPLETED", message="X Search 已完成")
            await uow.commit()

    async def _fail(self, snapshot: RunSnapshot, *, code: str, message: str) -> None:
        async with self._uow_factory() as uow:
            row = await self._locked(uow, snapshot)
            if row is None or row.status in TERMINAL_STATUSES:
                return
            row.error_code = code
            row.error_message = message
            await append_event(uow, row, stage="FAILED", message=message, payload={"error_code": code})
            await uow.commit()

    async def _locked(self, uow, snapshot: RunSnapshot):
        assert uow.runs is not None
        row = await uow.runs.get(domain_id=snapshot.domain_id, run_id=snapshot.run_id, lock=True)
        if row is None or row.lease_token != snapshot.lease_token:
            logger.warning("放弃写入：租约已失效 run_id={}", snapshot.run_id)
            return None
        return row

    @staticmethod
    def _snapshot(row) -> RunSnapshot:
        if row.lease_token is None:
            raise RuntimeError("认领 Run 后缺少 lease_token")
        return RunSnapshot(
            run_id=row.run_id, domain_id=int(row.domain_id), kind=row.kind,
            status=row.status, actor_id=row.actor_id, model_id=row.model_id,
            request_json=dict(row.request_json or {}),
            result_json=dict(row.result_json) if row.result_json else None,
            provider_request_id=row.provider_request_id, lease_token=row.lease_token,
            trace_id=row.trace_id,
        )

    @staticmethod
    def _research_request(snapshot: RunSnapshot) -> ResearchRequest:
        payload = dict(snapshot.request_json)
        payload.pop("agent_id", None)
        payload["model_id"] = str(snapshot.model_id)
        payload["trace_id"] = payload.get("trace_id") or snapshot.trace_id
        return ResearchRequest.model_validate(payload)
