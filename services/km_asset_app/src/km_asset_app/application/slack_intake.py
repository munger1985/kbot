"""Slack Events API 的验签、筛选与持久化接入。"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from km_asset_app.entities import SlackInboxEntity
from platform_core.identity import uuid7


_SLACK_MAILTO_PATTERN = re.compile(
    r"<mailto:[^|>]+\|(?P<label>[^>]+)>",
    re.IGNORECASE,
)


def _visible_message_text(value: object) -> str:
    """恢复 Slack 界面可见文本，不改变邮箱本身或原始 Event 报文。"""
    text = str(value or "")
    return _SLACK_MAILTO_PATTERN.sub(
        lambda match: match.group("label"),
        text,
    ).strip()


class SlackWebhookError(ValueError):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class SlackIntakeResult:
    receipt_id: UUID | None
    accepted: bool
    duplicate: bool
    challenge: str | None = None
    ignored: bool = False


def verify_slack_signature(
    *,
    signing_secret: str,
    timestamp: str,
    signature: str,
    raw_body: bytes,
    now: int | None = None,
    max_age_seconds: int = 300,
) -> bool:
    """使用 Slack v0 HMAC，并拒绝超出时间窗口的重放请求。"""
    try:
        request_timestamp = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = int(time.time()) if now is None else now
    if abs(current - request_timestamp) > max_age_seconds:
        return False
    base = b"v0:" + timestamp.encode("utf-8") + b":" + raw_body
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"), base, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def parse_message_event(payload: dict[str, Any]) -> dict[str, str] | None:
    """只接受人工发送的 message 与 app_mention 新消息。"""
    if payload.get("type") != "event_callback":
        return None
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    subtype = str(event.get("subtype") or "")
    if subtype in {"bot_message", "message_changed", "message_deleted"}:
        return None
    if event.get("bot_id"):
        return None
    event_type = str(event.get("type") or "")
    if event_type not in {"message", "app_mention"}:
        return None
    event_ts = str(event.get("event_ts") or event.get("ts") or "")
    message_ts = str(event.get("ts") or event_ts)
    values = {
        "event_id": str(payload.get("event_id") or ""),
        "workspace_id": str(payload.get("team_id") or ""),
        "event_type": event_type,
        "channel_id": str(event.get("channel") or ""),
        "slack_user_id": str(event.get("user") or ""),
        "message_text": _visible_message_text(event.get("text")),
        "event_ts": event_ts,
        "root_thread_ts": str(event.get("thread_ts") or message_ts),
        # Slack 可为同一条 @mention 同时投递 message 与 app_mention。
        # 两类事件的 client_msg_id 并不总是同时存在，而 ts 精确标识
        # 原始 Slack 消息；统一使用它可避免生成两个 Inbox 和两条等待回复。
        "message_identity": message_ts,
    }
    if any(not values[key] for key in values):
        return None
    return values


class SlackIntakeService:
    def __init__(self, *, uow_factory: Callable, slack_config):
        self._uow_factory = uow_factory
        self._config = slack_config

    async def receive(
        self,
        *,
        raw_body: bytes,
        headers: dict[str, str],
    ) -> SlackIntakeResult:
        if not self._config.enabled:
            raise SlackWebhookError(
                "SLACK_INTEGRATION_DISABLED", "Slack 集成未启用", 404
            )
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SlackWebhookError(
                "SLACK_PAYLOAD_INVALID", "Slack 请求不是有效 JSON", 400
            ) from exc
        payload_type = str(payload.get("type") or "")
        workspace_id = str(payload.get("team_id") or "")
        workspace = (
            self._config.workspace(workspace_id) if workspace_id else None
        )
        if workspace is not None:
            signing_secrets = (workspace.require_signing_secret(),)
        elif payload_type == "url_verification" and not workspace_id:
            # Slack 的 URL Verification 不保证携带 team_id。Signing Secret
            # 属于 Slack App，因此允许使用任一已配置 Workspace 的
            # App Secret 验签。
            signing_secrets = tuple(
                item.require_signing_secret()
                for item in self._config.workspaces
            )
        else:
            raise SlackWebhookError(
                "SLACK_WORKSPACE_UNKNOWN", "Slack Workspace 未注册", 404
            )
        verified = any(
            verify_slack_signature(
                signing_secret=signing_secret,
                timestamp=headers.get("x-slack-request-timestamp", ""),
                signature=headers.get("x-slack-signature", ""),
                raw_body=raw_body,
            )
            for signing_secret in signing_secrets
        )
        if not verified:
            raise SlackWebhookError(
                "SLACK_SIGNATURE_INVALID", "Slack 请求签名无效", 401
            )
        if payload_type == "url_verification":
            challenge = payload.get("challenge")
            if not isinstance(challenge, str) or not challenge:
                raise SlackWebhookError(
                    "SLACK_CHALLENGE_INVALID",
                    "Slack URL Verification 缺少 challenge",
                    400,
                )
            return SlackIntakeResult(
                receipt_id=None,
                accepted=True,
                duplicate=False,
                challenge=challenge,
            )
        parsed = parse_message_event(payload)
        if parsed is None:
            return SlackIntakeResult(
                receipt_id=None,
                accepted=True,
                duplicate=False,
                ignored=True,
            )
        message_key = hashlib.sha256(
            ":".join(
                (
                    parsed["workspace_id"],
                    parsed["channel_id"],
                    parsed["slack_user_id"],
                    parsed["message_identity"],
                )
            ).encode("utf-8")
        ).hexdigest()
        async with self._uow_factory() as uow:
            existing = await uow.slack.get_inbox_by_event_id(parsed["event_id"])
            if existing is None:
                existing = await uow.slack.get_inbox_by_message_key(message_key)
            if existing is not None:
                return SlackIntakeResult(
                    receipt_id=existing.inbox_id,
                    accepted=True,
                    duplicate=True,
                )
            entity = SlackInboxEntity(
                inbox_id=uuid7(),
                event_id=parsed["event_id"],
                message_key=message_key,
                workspace_id=parsed["workspace_id"],
                event_type=parsed["event_type"],
                channel_id=parsed["channel_id"],
                slack_user_id=parsed["slack_user_id"],
                event_ts=parsed["event_ts"],
                root_thread_ts=parsed["root_thread_ts"],
                message_text=parsed["message_text"],
                raw_body_hash=hashlib.sha256(raw_body).hexdigest(),
                raw_payload_json=payload,
                status="RECEIVED",
                attempt_count=0,
            )
            try:
                # add_inbox 会 flush；并发重复事件可能在这里就触发唯一约束，
                # 因此 flush 与 commit 必须由同一个冲突处理覆盖。
                await uow.slack.add_inbox(entity)
                await uow.commit()
            except IntegrityError:
                pass
            else:
                return SlackIntakeResult(
                    receipt_id=entity.inbox_id,
                    accepted=True,
                    duplicate=False,
                )
        async with self._uow_factory() as uow:
            existing = await uow.slack.get_inbox_by_event_id(parsed["event_id"])
            if existing is None:
                existing = await uow.slack.get_inbox_by_message_key(message_key)
            if existing is None:
                raise SlackWebhookError(
                    "SLACK_INTAKE_CONFLICT",
                    "Slack 事件并发接入失败",
                    409,
                )
            return SlackIntakeResult(
                receipt_id=existing.inbox_id,
                accepted=True,
                duplicate=True,
            )
