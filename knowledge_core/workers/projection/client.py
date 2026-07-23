"""HTTP client for KC INDEX and PROFILE worker protocols."""

from typing import Any

import aiohttp

from platform_core.config.settings import get_app_config, get_knowledge_core_config
from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class KcIndexProfileProtocolError(RuntimeError):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(f"{code}: {message}")
        self.code, self.status = code, status


class KcIndexProfileClient:
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
        self._caller_service = caller_service or get_app_config().service_name
        self._audience = audience or get_knowledge_core_config().service_name

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def claim_index(self, *, worker_id: str, lease_seconds: int) -> list[dict[str, Any]]:
        return await self._claim("index-tasks", worker_id=worker_id, lease_seconds=lease_seconds)

    async def claim_profile(self, *, worker_id: str, lease_seconds: int) -> list[dict[str, Any]]:
        return await self._claim("profile-tasks", worker_id=worker_id, lease_seconds=lease_seconds)

    async def claim_purge(self, *, worker_id: str, lease_seconds: int) -> list[dict[str, Any]]:
        return await self._claim("purge-tasks", worker_id=worker_id, lease_seconds=lease_seconds)

    async def run_index(self, task: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
        return await self._run(
            "index-tasks", task, {"batch_size": batch_size},
        )

    async def run_profile(self, task: dict[str, Any]) -> dict[str, Any]:
        return await self._run("profile-tasks", task, {})

    async def run_purge(self, task: dict[str, Any]) -> dict[str, Any]:
        return await self._run("purge-tasks", task, {})

    async def heartbeat_index(self, task: dict[str, Any], *, lease_seconds: int) -> None:
        await self._heartbeat("index-tasks", task, lease_seconds=lease_seconds)

    async def heartbeat_profile(self, task: dict[str, Any], *, lease_seconds: int) -> None:
        await self._heartbeat("profile-tasks", task, lease_seconds=lease_seconds)

    async def heartbeat_purge(self, task: dict[str, Any], *, lease_seconds: int) -> None:
        await self._heartbeat("purge-tasks", task, lease_seconds=lease_seconds)

    async def fail_index(self, task: dict[str, Any], *, failure_code: str, message: str) -> None:
        await self._fail("index-tasks", task, failure_code=failure_code, message=message)

    async def fail_profile(self, task: dict[str, Any], *, failure_code: str, message: str) -> None:
        await self._fail("profile-tasks", task, failure_code=failure_code, message=message)

    async def fail_purge(self, task: dict[str, Any], *, failure_code: str, message: str) -> None:
        await self._fail("purge-tasks", task, failure_code=failure_code, message=message)

    async def _claim(self, kind: str, *, worker_id: str, lease_seconds: int) -> list[dict[str, Any]]:
        payload = await self._request("POST", f"{INTERNAL_API_V1}/knowledge/{kind}/claim", json={
            "worker_id": worker_id, "max_tasks": 1, "lease_seconds": lease_seconds,
        })
        return list(payload.get("tasks") or [])

    async def _run(self, kind: str, task: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "worker_id": task["worker_id"],
            "input_fingerprint": task["input_fingerprint"],
            **extra,
        }
        return await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/{kind}/{task['job_id']}/run", json=payload,
        )

    async def _heartbeat(self, kind: str, task: dict[str, Any], *, lease_seconds: int) -> None:
        await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/{kind}/{task['job_id']}/heartbeat", json={
                "worker_id": task["worker_id"],
                "input_fingerprint": task["input_fingerprint"],
                "lease_seconds": lease_seconds,
            },
        )

    async def _fail(self, kind: str, task: dict[str, Any], *, failure_code: str, message: str) -> None:
        await self._request(
            "POST", f"{INTERNAL_API_V1}/knowledge/{kind}/{task['job_id']}/fail", json={
                "worker_id": task["worker_id"],
                "input_fingerprint": task["input_fingerprint"],
                "failure_class": "TRANSIENT",
                "failure_code": failure_code,
                "failure_message": message[:1000],
            },
        )

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("KcIndexProfileClient must be used as an async context manager")
        headers = {
            **build_internal_auth_headers(
                audience=self._audience,
                caller_service=self._caller_service,
            ),
            **kwargs.pop("headers", {}),
        }
        async with self._session.request(
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
                raise KcIndexProfileProtocolError(code, message, response.status)
            body = await response.json()
            return body if isinstance(body, dict) else {}
