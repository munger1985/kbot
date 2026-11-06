"""飞书 OAuth 与消息协议；只返回安全分类，不暴露响应正文。"""

from __future__ import annotations

import asyncio
import math
import re
from time import monotonic

import httpx

from .delivery import DeliveryStatus as Status, SendResult, canonical_json
from .external_config import ExternalDeliveryConfig

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_.:-]{1,256}$")
_RATE_CODES = {99991400, 99991401, 99991402, 230020}
_TOKEN_CODES = {99991661, 99991663, 99991664, 99991668}
_BLOCK_CODES = {99991672, 99991679, 230001, 230002, 230006, 230013}


def safe_identifier(value):
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else None


def _retry_after(response):
    raw = response.headers.get("x-ogw-ratelimit-reset") or response.headers.get("retry-after")
    try:
        value = float(raw)
        return max(1.0, value) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class FeishuSender:
    """同一连接令牌刷新合并；调用期间不持有数据库事务。"""

    def __init__(self, *, config: ExternalDeliveryConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=config.request_timeout_seconds, follow_redirects=False)
        self._owns_client = client is None
        self._tokens = {}
        self._locks = {}

    async def close(self):
        if self._owns_client:
            await self.client.aclose()
        self._tokens.clear()

    def _key(self, connection):
        return (str(connection["id"]), connection["app_id"], connection["credential_version"], "im:message:send_as_bot")

    async def _token(self, connection, secret):
        if self.config.fixture_mode and not secret.startswith("fixture-"):
            raise ValueError("NOTIFICATION_REAL_SECRET_TO_FIXTURE_DENIED")
        key = self._key(connection)
        async with self._locks.setdefault(key, asyncio.Lock()):
            cached = self._tokens.get(key)
            if cached and monotonic() < cached[1]:
                return cached[0]
            # client_secret_post；不得再同时设置 HTTP Basic。
            response = await self.client.post(self.config.token_url, data={
                "grant_type": "client_credentials", "client_id": connection["app_id"], "client_secret": secret,
                "scope": "im:message:send_as_bot",
            }, timeout=self.config.request_timeout_seconds)
            if response.status_code == 429:
                raise TokenFailure("FEISHU_TOKEN_RATE_LIMIT", retry_after=_retry_after(response))
            if response.status_code >= 500:
                raise TokenFailure("FEISHU_TOKEN_UNAVAILABLE", retry_after=_retry_after(response))
            if response.status_code != 200:
                raise TokenFailure("FEISHU_CREDENTIAL_REJECTED", blocked=True)
            try:
                body = response.json()
                if isinstance(body, dict) and body.get("code") in _RATE_CODES:
                    raise TokenFailure("FEISHU_TOKEN_RATE_LIMIT", retry_after=_retry_after(response))
                token, expiry = body["access_token"], body["expires_in"]
                if (not isinstance(token, str) or not token or len(token) > 65536
                    or not isinstance(expiry, (int, float)) or isinstance(expiry, bool) or not math.isfinite(expiry) or expiry <= 0
                    or str(body.get("token_type", "")).lower() != "bearer"
                    or body.get("code", 0) != 0):
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                raise TokenFailure("FEISHU_TOKEN_RESPONSE_INVALID", blocked=True) from None
            self._tokens[key] = (token, monotonic() + max(0, expiry - min(60, expiry / 10)))
            return token

    async def check(self, *, connection, secret):
        try:
            await self._token(connection, secret)
            return SendResult(Status.ACCEPTED)
        except TokenFailure as exc:
            return SendResult(Status.BLOCKED if exc.blocked else Status.RETRY_WAIT, exc.code,
                              retry_after_seconds=exc.retry_after)
        except (httpx.HTTPError, ValueError):
            return SendResult(Status.BLOCKED, "FEISHU_TOKEN_UNAVAILABLE")

    async def send(self, *, connection, secret, target_type, external_id, content, provider_uuid):
        if target_type not in {"open_id", "chat_id"} or not 1 <= len(provider_uuid) <= 50:
            return SendResult(Status.BLOCKED, "FEISHU_REQUEST_INVALID")
        try:
            token = await self._token(connection, secret)
        except TokenFailure as exc:
            return SendResult(Status.BLOCKED if exc.blocked else Status.RETRY_WAIT, exc.code,
                              retry_after_seconds=exc.retry_after)
        except (httpx.HTTPError, ValueError):
            return SendResult(Status.RETRY_WAIT, "FEISHU_TOKEN_UNAVAILABLE")
        try:
            response = await self.client.post(
                self.config.message_origin + "/open-apis/im/v1/messages",
                params={"receive_id_type": target_type},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": external_id, "msg_type": "interactive",
                      "content": canonical_json(content), "uuid": provider_uuid},
                timeout=self.config.request_timeout_seconds,
            )
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
            return SendResult(Status.RETRY_WAIT, "FEISHU_CONNECT_FAILED")
        except httpx.HTTPError:
            return SendResult(Status.UNKNOWN, "FEISHU_RESPONSE_UNCERTAIN")
        request_id = safe_identifier(response.headers.get("x-tt-logid"))
        if response.status_code == 429:
            return SendResult(Status.RETRY_WAIT, "FEISHU_RATE_LIMIT", request_id=request_id,
                              retry_after_seconds=_retry_after(response))
        if response.status_code >= 500:
            return SendResult(Status.UNKNOWN, "FEISHU_RESPONSE_UNCERTAIN", request_id=request_id)
        if response.status_code == 401:
            self._tokens.pop(self._key(connection), None)
            return SendResult(Status.RETRY_WAIT, "FEISHU_TOKEN_EXPIRED", request_id=request_id)
        try:
            body = response.json()
            code = body["code"]
            if isinstance(code, bool) or not isinstance(code, int):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            rejected = 400 <= response.status_code < 500 and response.status_code != 408
            return SendResult(Status.BLOCKED if rejected else Status.UNKNOWN,
                "FEISHU_REQUEST_REJECTED" if rejected else "FEISHU_RESPONSE_INVALID", request_id=request_id)
        if 200 <= response.status_code < 300 and code == 0:
            data = body.get("data")
            receipt = safe_identifier(data.get("message_id")) if isinstance(data, dict) else None
            receipt = receipt if receipt and receipt.startswith("om_") and len(receipt) > 3 else None
            accepted = response.status_code == 200 and receipt is not None
            return SendResult(Status.ACCEPTED if accepted else Status.UNKNOWN,
                              None if accepted else "FEISHU_RECEIPT_MISSING" if not receipt else "FEISHU_RESPONSE_UNCERTAIN",
                              message_id=receipt if accepted else None, request_id=request_id)
        if code in _RATE_CODES:
            return SendResult(Status.RETRY_WAIT, "FEISHU_RATE_LIMIT", request_id=request_id,
                              retry_after_seconds=_retry_after(response))
        if code in _TOKEN_CODES or response.status_code == 401:
            self._tokens.pop(self._key(connection), None)
            return SendResult(Status.RETRY_WAIT, "FEISHU_TOKEN_EXPIRED", request_id=request_id)
        # 未识别的业务拒绝需要人工核对；不输出厂商 msg（可能包含身份信息）。
        return SendResult(Status.BLOCKED, "FEISHU_PERMISSION_DENIED" if code in _BLOCK_CODES else "FEISHU_REQUEST_REJECTED",
                          request_id=request_id)


class TokenFailure(Exception):
    def __init__(self, code, *, blocked=False, retry_after=None):
        super().__init__(code)
        self.code, self.blocked, self.retry_after = code, blocked, retry_after
