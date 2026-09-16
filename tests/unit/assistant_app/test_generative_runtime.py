"""智能工作台 X Search / 文生图创建闸门、幂等与 Worker 二次调用拦截。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from assistant_app.application import (
    AssistantApplicationError,
    CreateImageGenerationCommand,
    CreateResearchCommand,
    ImageGenerationService,
    ResearchRunService,
)
from assistant_app.application.runtime import already_attempted, mark_upstream_attempted, public_result
from assistant_app.application.worker import AssistantRunWorker
from assistant_app.entities import (
    AssistantMediaAssetEntity,
    AssistantModelBindingEntity,
    AssistantPromptRevisionEntity,
    AssistantRunEntity,
    AssistantRunEventEntity,
    AssistantXSourceEntity,
)
from platform_core.contracts import GeneratedImageArtifact, ImageGenerationResult, ResearchCitation, ResearchResult
from platform_core.identity import uuid7

import unittest


DOMAIN_ID = 41
ACTOR_ID = "user-41"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _catalog_row(model_id: UUID, *, role: str, active: bool = True, verified: bool = True) -> dict[str, Any]:
    return {
        "model_id": str(model_id),
        "display_name": "测试生成模型",
        "served_model_name": "test-generative-model",
        "provider": "test",
        "status": "ACTIVE" if active else "DISABLED",
        "supports_x_search": bool(verified and role == "X_SEARCH"),
        "supports_image_generation": bool(verified and role == "IMAGE_GENERATION"),
        "supports_responses_streaming": True,
        "capability_verified_at": "2026-09-01T00:00:00+00:00" if verified else None,
        "category": 1,
        "secret": "must-not-leak",
        "config_file": "must-not-leak",
        "api_endpoint": "https://example.invalid",
        "model_params": {"api_key": "must-not-leak"},
    }


class _Catalog:
    def __init__(self, rows: dict[str, dict[str, Any]] | None = None) -> None:
        self.rows = rows or {}

    async def get_model(self, model_id: UUID) -> dict[str, Any]:
        row = self.rows.get(str(model_id))
        if row is None:
            raise LookupError(model_id)
        return row


class _MemoryRepo:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def _prepare(self, row) -> None:
        if getattr(row, "row_version", None) in (None, 0):
            row.row_version = 1

    async def add(self, row) -> None:
        self._prepare(row)
        self.rows.append(row)


class _BindingRepo(_MemoryRepo):
    async def get_by_role(self, *, domain_id: int, role: str, lock: bool = False):
        del lock
        for row in self.rows:
            if int(row.domain_id) == domain_id and row.role == role:
                return row
        return None

    async def list(self, *, domain_id: int):
        return [row for row in self.rows if int(row.domain_id) == domain_id]


class _RunRepo(_MemoryRepo):
    async def add(self, row: AssistantRunEntity) -> None:
        self._prepare(row)
        for existing in self.rows:
            if (
                int(existing.domain_id) == int(row.domain_id)
                and existing.actor_id == row.actor_id
                and existing.idempotency_key == row.idempotency_key
            ):
                raise IntegrityError("UK_ASST_RUN_IDEMP", None, Exception("duplicate"))
        self.rows.append(row)

    async def get(self, *, domain_id: int, run_id: UUID, lock: bool = False):
        del lock
        for row in self.rows:
            if int(row.domain_id) == domain_id and row.run_id == run_id:
                return row
        return None

    async def get_by_idempotency(self, *, domain_id: int, actor_id: str, idempotency_key: str):
        for row in self.rows:
            if int(row.domain_id) == domain_id and row.actor_id == actor_id and row.idempotency_key == idempotency_key:
                return row
        return None

    async def list(self, *, domain_id: int, kind: str | None = None, status: str | None = None, actor_id: str | None = None, limit: int = 50):
        rows = [row for row in self.rows if int(row.domain_id) == domain_id]
        if kind:
            rows = [row for row in rows if row.kind == kind]
        if status:
            rows = [row for row in rows if row.status == status]
        if actor_id:
            rows = [row for row in rows if row.actor_id == actor_id]
        return rows[:limit]

    async def delete(self, row: AssistantRunEntity) -> None:
        self.rows = [item for item in self.rows if item.run_id != row.run_id]

    async def claim(self, *, worker_id: str, lease_until: datetime):
        now = _now()
        for row in self.rows:
            expired = row.lease_until is None or row.lease_until < now
            in_progress = row.status in {"SEARCHING", "ORGANIZING_SOURCES", "COMPOSING", "GENERATING"}
            if (row.status == "ACCEPTED" and expired) or (in_progress and row.lease_until is not None and row.lease_until < now):
                row.lease_owner = worker_id
                row.lease_token = uuid7()
                row.lease_until = lease_until
                row.row_version = int(row.row_version) + 1
                if row.started_at is None:
                    row.started_at = now
                return row
        return None


class _EventRepo(_MemoryRepo):
    async def next_sequence(self, *, run_id: UUID) -> int:
        return len([row for row in self.rows if row.run_id == run_id]) + 1

    async def list(self, *, domain_id: int, run_id: UUID):
        return [row for row in self.rows if int(row.domain_id) == domain_id and row.run_id == run_id]

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        self.rows = [
            row for row in self.rows
            if not (int(row.domain_id) == domain_id and row.run_id == run_id)
        ]


class _SourceRepo(_MemoryRepo):
    async def list(self, *, domain_id: int, run_id: UUID):
        return [row for row in self.rows if int(row.domain_id) == domain_id and row.run_id == run_id]

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        self.rows = [
            row for row in self.rows
            if not (int(row.domain_id) == domain_id and row.run_id == run_id)
        ]


class _RevisionRepo(_MemoryRepo):
    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        self.rows = [
            row for row in self.rows
            if not (int(row.domain_id) == domain_id and row.run_id == run_id)
        ]


class _MediaRepo(_MemoryRepo):
    async def get(self, *, domain_id: int, asset_id: UUID):
        for row in self.rows:
            if int(row.domain_id) == domain_id and row.asset_id == asset_id:
                return row
        return None

    async def list(self, *, domain_id: int, run_id: UUID | None = None, status: str | None = None, limit: int = 50):
        rows = [row for row in self.rows if int(row.domain_id) == domain_id]
        if run_id:
            rows = [row for row in rows if row.run_id == run_id]
        if status:
            rows = [row for row in rows if row.status == status]
        return rows[:limit]

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        self.rows = [
            row for row in self.rows
            if not (int(row.domain_id) == domain_id and row.run_id == run_id)
        ]


class _State:
    def __init__(self) -> None:
        self.bindings = _BindingRepo()
        self.runs = _RunRepo()
        self.run_events = _EventRepo()
        self.x_sources = _SourceRepo()
        self.prompt_revisions = _RevisionRepo()
        self.media_assets = _MediaRepo()


class _UnitOfWork:
    def __init__(self, state: _State) -> None:
        self.bindings = state.bindings
        self.runs = state.runs
        self.run_events = state.run_events
        self.x_sources = state.x_sources
        self.prompt_revisions = state.prompt_revisions
        self.media_assets = state.media_assets
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self) -> None:
        self.commits += 1


class _ObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def get(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class _Generative:
    def __init__(self) -> None:
        self.research_calls = 0
        self.image_calls = 0
        self.last_research = None
        self.last_image = None

    async def research(self, request):
        self.research_calls += 1
        self.last_research = request
        return ResearchResult(
            status="COMPLETED",
            answer="外部实时线索摘要",
            citations=(
                ResearchCitation(
                    provider_citation_id="x-1",
                    canonical_url="https://x.com/example/status/1",
                    title="示例来源",
                    excerpt="摘要",
                    author_handle="example",
                ),
            ),
            provider_request_id="provider-research-1",
        )

    async def generate_image(self, request):
        self.image_calls += 1
        self.last_image = request
        return ImageGenerationResult(
            status="COMPLETED",
            artifacts=(
                GeneratedImageArtifact(
                    provider_artifact_id="img-1",
                    mime_type="image/png",
                    content=b"\x89PNG-test-bytes",
                    width=64,
                    height=64,
                ),
            ),
            provider_request_id="provider-image-1",
        )


def _research_command(**kwargs) -> CreateResearchCommand:
    values = dict(
        domain_id=DOMAIN_ID,
        actor_id=ACTOR_ID,
        request_id="req-1",
        trace_id="trace-1",
        idempotency_key="idem-research-1",
        input="过去七天的官方动态",
    )
    values.update(kwargs)
    return CreateResearchCommand(**values)


def _image_command(**kwargs) -> CreateImageGenerationCommand:
    values = dict(
        domain_id=DOMAIN_ID,
        actor_id=ACTOR_ID,
        request_id="req-2",
        trace_id="trace-2",
        idempotency_key="idem-image-1",
        prompt="一张纸张风工作台插画",
        aspect_ratio="1:1",
        count=1,
    )
    values.update(kwargs)
    return CreateImageGenerationCommand(**values)


class AssistantGenerativeRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.state = _State()
        self.catalog = _Catalog()
        self.store = _ObjectStore()
        self.generative = _Generative()
        self.uow_factory = lambda: _UnitOfWork(self.state)
        self.research = ResearchRunService(uow_factory=self.uow_factory, catalog_client=self.catalog)
        self.images = ImageGenerationService(
            uow_factory=self.uow_factory, catalog_client=self.catalog, object_store=self.store,
        )
        self.worker = AssistantRunWorker(
            uow_factory=self.uow_factory,
            generative_client=self.generative,
            object_store=self.store,
            lease_seconds=360,
        )

    async def _bind(self, *, role: str, verified: bool = True, active: bool = True) -> UUID:
        model_id = uuid7()
        row = _catalog_row(model_id, role=role, active=active, verified=verified)
        self.catalog.rows[str(model_id)] = row
        await self.state.bindings.add(AssistantModelBindingEntity(
            binding_id=uuid7(),
            domain_id=DOMAIN_ID,
            role=role,
            model_id=model_id,
            model_snapshot_json={
                key: row[key]
                for key in (
                    "model_id", "display_name", "served_model_name", "provider", "status",
                    "supports_x_search", "supports_image_generation", "supports_responses_streaming",
                    "capability_verified_at", "category",
                )
            },
            created_by=ACTOR_ID,
            updated_by=ACTOR_ID,
            created_at=_now(),
            updated_at=_now(),
        ))
        return model_id

    async def test_research_create_rejects_missing_binding(self) -> None:
        with self.assertRaises(AssistantApplicationError) as raised:
            await self.research.create(_research_command())
        self.assertEqual("MODEL_BINDING_MISSING", raised.exception.code)
        self.assertEqual(422, raised.exception.status_code)
        self.assertEqual([], self.state.runs.rows)

    async def test_image_create_rejects_unverified_capability(self) -> None:
        await self._bind(role="IMAGE_GENERATION", verified=False)
        with self.assertRaises(AssistantApplicationError) as raised:
            await self.images.create(_image_command())
        self.assertEqual("MODEL_CAPABILITY_UNVERIFIED", raised.exception.code)
        self.assertEqual(422, raised.exception.status_code)

    async def test_research_create_is_idempotent_and_hides_secrets(self) -> None:
        await self._bind(role="X_SEARCH")
        first, created = await self.research.create(_research_command())
        second, created_again = await self.research.create(_research_command())
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual("ACCEPTED", first["status"])
        snapshot = self.state.runs.rows[0].model_snapshot_json
        self.assertNotIn("secret", snapshot)
        self.assertNotIn("config_file", snapshot)
        self.assertNotIn("api_endpoint", snapshot)
        self.assertNotIn("model_params", snapshot)
        self.assertTrue(snapshot["supports_x_search"])

    async def test_worker_blocks_second_upstream_call(self) -> None:
        await self._bind(role="X_SEARCH")
        view, _ = await self.research.create(_research_command())
        run = self.state.runs.rows[0]
        mark_upstream_attempted(run)
        self.assertTrue(already_attempted(run))
        handled = await self.worker.process_once()
        self.assertTrue(handled)
        self.assertEqual(0, self.generative.research_calls)
        self.assertEqual("FAILED", run.status)
        self.assertEqual("PROVIDER_UNAVAILABLE", run.error_code)
        self.assertIsNone(public_result(run.result_json).get("upstream_attempted"))
        self.assertEqual(view["run_id"], str(run.run_id))

    async def test_worker_blocks_when_provider_request_id_exists(self) -> None:
        await self._bind(role="X_SEARCH")
        await self.research.create(_research_command(idempotency_key="idem-research-2", request_id="req-12"))
        run = self.state.runs.rows[0]
        run.provider_request_id = "already-sent"
        handled = await self.worker.process_once()
        self.assertTrue(handled)
        self.assertEqual(0, self.generative.research_calls)
        self.assertEqual("FAILED", run.status)

    async def test_image_worker_stores_asset_ids_and_object_bytes(self) -> None:
        await self._bind(role="IMAGE_GENERATION")
        view, created = await self.images.create(_image_command())
        self.assertTrue(created)
        handled = await self.worker.process_once()
        self.assertTrue(handled)
        self.assertEqual(1, self.generative.image_calls)
        run = self.state.runs.rows[0]
        self.assertEqual("COMPLETED", run.status)
        self.assertEqual({"asset_ids"}, set(run.result_json.keys()))
        self.assertEqual(1, len(run.result_json["asset_ids"]))
        self.assertNotIn("content", run.result_json)
        asset = self.state.media_assets.rows[0]
        self.assertEqual(
            f"media/{DOMAIN_ID}/{run.run_id}/{asset.asset_id}.png",
            asset.object_key,
        )
        self.assertIsNone(getattr(asset, "content", None))
        self.assertEqual(b"\x89PNG-test-bytes", self.store.objects[asset.object_key])
        self.assertEqual(str(asset.asset_id), run.result_json["asset_ids"][0])
        public = public_result(run.result_json)
        self.assertNotIn("upstream_attempted", public)
        self.assertEqual(view["kind"], "IMAGE_GENERATION")

    async def test_image_worker_surfaces_provider_error_message(self) -> None:
        await self._bind(role="IMAGE_GENERATION")
        await self.images.create(_image_command(idempotency_key="idem-image-fail", request_id="req-image-fail"))

        async def failed(_request):
            return ImageGenerationResult(
                status="FAILED",
                artifacts=(),
                error_code="PROVIDER_UNAVAILABLE",
                error_message="Imagine backend is currently unavailable.",
            )

        self.generative.generate_image = failed
        handled = await self.worker.process_once()
        self.assertTrue(handled)
        run = self.state.runs.rows[0]
        self.assertEqual("FAILED", run.status)
        self.assertEqual("PROVIDER_UNAVAILABLE", run.error_code)
        self.assertEqual("Imagine backend is currently unavailable.", run.error_message)

    async def test_research_delete_removes_owned_run_and_hides_from_others(self) -> None:
        await self._bind(role="X_SEARCH")
        view, _ = await self.research.create(_research_command())
        run_id = UUID(view["run_id"])
        await self.research.delete(domain_id=DOMAIN_ID, run_id=run_id, actor_id=ACTOR_ID)
        self.assertEqual([], self.state.runs.rows)
        self.assertEqual([], self.state.run_events.rows)
        with self.assertRaises(AssistantApplicationError) as raised:
            await self.research.get(domain_id=DOMAIN_ID, run_id=run_id, actor_id=ACTOR_ID)
        self.assertEqual("RUN_NOT_FOUND", raised.exception.code)

    async def test_image_delete_removes_run_and_object_bytes(self) -> None:
        await self._bind(role="IMAGE_GENERATION")
        view, _ = await self.images.create(_image_command())
        handled = await self.worker.process_once()
        self.assertTrue(handled)
        run_id = UUID(view["run_id"])
        object_key = self.state.media_assets.rows[0].object_key
        self.assertIn(object_key, self.store.objects)
        await self.images.delete(domain_id=DOMAIN_ID, run_id=run_id, actor_id=ACTOR_ID)
        self.assertEqual([], self.state.runs.rows)
        self.assertEqual([], self.state.media_assets.rows)
        self.assertEqual([], self.state.prompt_revisions.rows)
        self.assertNotIn(object_key, self.store.objects)
        with self.assertRaises(AssistantApplicationError) as raised:
            await self.images.delete(domain_id=DOMAIN_ID, run_id=run_id, actor_id="other-user")
        self.assertEqual("RUN_NOT_FOUND", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
