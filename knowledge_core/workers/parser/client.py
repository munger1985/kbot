"""HTTP client for the KC Parser Worker lease protocol."""

from dataclasses import dataclass
from typing import Any

import aiohttp

from platform_core.config.settings import get_knowledge_core_config, get_parser_config
from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class KcParserProtocolError(RuntimeError):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(f"{code}: {message}")
        self.code, self.status = code, status


@dataclass(frozen=True)
class ParseTask:
    job_id: int
    lease_owner: str
    lease_until: str
    input_fingerprint: str
    document_version_id: int
    parse_view_id: int
    source_read_url: str
    detected_mime_type: str
    view_kind: str
    parse_config_fingerprint: str
    policy_snapshot: dict[str, Any]


class KcParseClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int = 600,
        caller_service: str | None = None,
        audience: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None
        self._caller_service = caller_service or get_parser_config().service_name
        self._audience = audience or get_knowledge_core_config().service_name

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def claim(self, *, worker_id: str, lease_seconds: int) -> list[ParseTask]:
        payload = await self._request("POST", f"{INTERNAL_API_V1}/knowledge/parse-tasks/claim", json={
            "worker_id": worker_id, "max_tasks": 1, "lease_seconds": lease_seconds,
        })
        return [ParseTask(**task) for task in payload["tasks"]]

    async def heartbeat(self, task: ParseTask, *, lease_seconds: int) -> None:
        await self._request("POST", f"{INTERNAL_API_V1}/knowledge/parse-tasks/{task.job_id}/heartbeat", json={
            "worker_id": task.lease_owner,
            "input_fingerprint": task.input_fingerprint,
            "lease_seconds": lease_seconds,
        })

    async def upload_artifact(
        self, task: ParseTask, *, name: str, payload: Any,
        sha256: str, schema: str, generator: str,
    ) -> dict[str, str]:
        return await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/parse-tasks/{task.job_id}/artifacts/{name}",
            json={
                "worker_id": task.lease_owner,
                "input_fingerprint": task.input_fingerprint,
                "sha256": sha256, "schema_name": schema,
                "generator": generator, "payload": payload,
            },
        )

    async def submit_evidence(self, task: ParseTask, items: list[dict[str, Any]]) -> int:
        payload = await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/parse-tasks/{task.job_id}/evidence-batches",
            json={
                "worker_id": task.lease_owner,
                "input_fingerprint": task.input_fingerprint,
                "items": items,
            },
        )
        return int(payload["inserted"])

    async def complete(
        self, task: ParseTask, *, artifact_manifest: dict[str, Any],
        output_fingerprint: str, quality_report: dict[str, Any], quality_score: float,
    ) -> int:
        payload = await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/parse-tasks/{task.job_id}/complete",
            json={
                "worker_id": task.lease_owner,
                "input_fingerprint": task.input_fingerprint,
                "artifact_manifest": artifact_manifest,
                "output_fingerprint": output_fingerprint,
                "quality_score": quality_score,
                "quality_report": quality_report,
            },
        )
        return int(payload["evidence_count"])

    async def fail(
        self, task: ParseTask, *, failure_class: str,
        failure_code: str, failure_message: str,
        artifact_manifest: dict[str, Any] | None = None,
    ) -> None:
        await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/parse-tasks/{task.job_id}/fail",
            json={
                "worker_id": task.lease_owner,
                "input_fingerprint": task.input_fingerprint,
                "failure_class": failure_class,
                "failure_code": failure_code,
                "failure_message": failure_message[:1000],
                "artifact_manifest": artifact_manifest,
            },
        )

    async def download_source(self, task: ParseTask) -> bytes:
        session = self._require_session()
        params = {
            "worker_id": task.lease_owner,
            "input_fingerprint": task.input_fingerprint,
        }
        async with session.get(
            task.source_read_url,
            params=params,
            headers=self._auth_headers(),
        ) as response:
            if response.status >= 400:
                raise KcParserProtocolError("SOURCE_READ_FAILED", await response.text(), response.status)
            return await response.read()

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        session = self._require_session()
        headers = {
            **self._auth_headers(),
            **kwargs.pop("headers", {}),
        }
        async with session.request(
            method,
            f"{self._base_url}{path}",
            headers=headers,
            **kwargs,
        ) as response:
            if response.status >= 400:
                try:
                    body = await response.json()
                    detail = body.get("detail", body)
                    code = detail.get("code", "KC_PROTOCOL_ERROR") if isinstance(detail, dict) else "KC_PROTOCOL_ERROR"
                    message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                except Exception:
                    code, message = "KC_PROTOCOL_ERROR", await response.text()
                raise KcParserProtocolError(code, message, response.status)
            return await response.json()

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("KcParseClient must be used as an async context manager")
        return self._session

    def _auth_headers(self) -> dict[str, str]:
        """为每次请求签发新的短期内部身份令牌。"""
        return build_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller_service,
        )
