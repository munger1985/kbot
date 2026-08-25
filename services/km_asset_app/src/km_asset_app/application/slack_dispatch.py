"""Slack Inbox/Outbox 的可恢复后台处理。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import aiohttp
from loguru import logger
from sqlalchemy.exc import IntegrityError

from km_asset_app.application.slack_assets import (
    assemble_slack_asset_cards,
)
from km_asset_app.application.slack_rendering import (
    build_callback_payload,
    render_slack_replies,
    slack_visible_payload,
    waiting_message,
)
from km_asset_app.entities import SlackDeliveryEntity, SlackThreadEntity
from platform_clients import KmPortalClientError
from platform_core.identity import uuid7


_TERMINAL_RUN_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "EXPIRED"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:128]


class SlackDispatchService:
    def __init__(
        self,
        *,
        uow_factory: Callable,
        main_api_client,
        slack_config,
        worker_id: str,
        http_session: aiohttp.ClientSession,
        callback_debug_log_path: Path | None = None,
    ):
        self._uow_factory = uow_factory
        self._main_api_client = main_api_client
        self._config = slack_config
        self._worker_id = worker_id
        self._http_session = http_session
        self._callback_debug_log_path = callback_debug_log_path

    async def run_once(self) -> bool:
        worked = await self._process_one_inbox()
        delivered = await self._process_one_delivery()
        return worked or delivered

    async def _process_one_inbox(self) -> bool:
        async with self._uow_factory() as uow:
            inbox = await uow.slack.claim_inbox(
                worker_id=self._worker_id,
                lease_seconds=self._config.lease_seconds,
            )
            if inbox is None:
                return False
            inbox_id = inbox.inbox_id
            status = inbox.status
            await uow.commit()
        try:
            if status == "RECEIVED":
                await self._start_run(inbox_id)
            else:
                await self._check_run(inbox_id)
        except Exception as exc:
            logger.exception("Slack Inbox 处理失败：inbox_id={}", inbox_id)
            await self._record_inbox_error(inbox_id, exc)
        return True

    async def _start_run(self, inbox_id: UUID) -> None:
        async with self._uow_factory() as uow:
            inbox = await uow.slack.get_inbox(inbox_id)
            workspace = self._config.workspace(inbox.workspace_id)
            if workspace is None:
                raise RuntimeError("Slack Workspace 配置已移除")
            # ORM 实体只在当前 UOW 中访问，提交后仅使用标量快照，
            # 避免 expire_on_commit 导致后台 Worker 触发 DetachedInstanceError。
            workspace_id = inbox.workspace_id
            channel_id = inbox.channel_id
            slack_user_id = inbox.slack_user_id
            root_thread_ts = inbox.root_thread_ts
            event_id = inbox.event_id
            message_text = inbox.message_text
            callback_sent_at = inbox.callback_sent_at
            waiting = await uow.slack.get_delivery(
                inbox_id=inbox_id, delivery_type="WAITING"
            )
            if waiting is None:
                await uow.slack.add_delivery(
                    SlackDeliveryEntity(
                        delivery_id=uuid7(),
                        inbox_id=inbox.inbox_id,
                        workspace_id=workspace_id,
                        channel_id=channel_id,
                        slack_user_id=slack_user_id,
                        thread_ts=root_thread_ts,
                        delivery_type="WAITING",
                        payload_json={
                            "channel": channel_id,
                            "thread_ts": root_thread_ts,
                            "text": (
                                f"<@{slack_user_id}> "
                                f"{waiting_message(message_text)}"
                            ),
                        },
                        status="PENDING",
                        attempt_count=0,
                    )
                )
            thread = await uow.slack.get_thread(
                workspace_id=workspace_id,
                channel_id=channel_id,
                root_thread_ts=root_thread_ts,
                slack_user_id=slack_user_id,
            )
            conversation_id = (
                thread.conversation_id if thread is not None else None
            )
            await uow.commit()

        if conversation_id is None:
            conversation = await self._main_api_client.create_conversation(
                payload={
                    "agent_id": str(workspace.agent_id),
                    "title": "Slack 对话",
                    "retention_policy": "DEFAULT",
                },
            )
            conversation_id = UUID(str(conversation["conversation_id"]))
            try:
                async with self._uow_factory() as uow:
                    existing = await uow.slack.get_thread(
                        workspace_id=workspace_id,
                        channel_id=channel_id,
                        root_thread_ts=root_thread_ts,
                        slack_user_id=slack_user_id,
                    )
                    if existing is None:
                        await uow.slack.add_thread(
                            SlackThreadEntity(
                                thread_id=uuid7(),
                                workspace_id=workspace_id,
                                channel_id=channel_id,
                                root_thread_ts=root_thread_ts,
                                slack_user_id=slack_user_id,
                                domain_id=workspace.domain_id,
                                agent_id=workspace.agent_id,
                                conversation_id=conversation_id,
                            )
                        )
                        await uow.commit()
                    else:
                        conversation_id = existing.conversation_id
            except IntegrityError:
                # 同一 Slack thread 的并发首消息只允许一个映射获胜。
                async with self._uow_factory() as uow:
                    thread = await uow.slack.get_thread(
                        workspace_id=workspace_id,
                        channel_id=channel_id,
                        root_thread_ts=root_thread_ts,
                        slack_user_id=slack_user_id,
                    )
                    if thread is None:
                        raise
                    conversation_id = thread.conversation_id
        if conversation_id is None:
            raise RuntimeError("Slack Thread 缺少 Conversation 映射")
        try:
            conversation = await self._main_api_client.get_conversation(
                conversation_id=conversation_id,
            )
        except KmPortalClientError as exc:
            # 旧版 Slack 通过内部 Runtime 创建的会话不属于当前公开
            # API Key 用户。部署新链路后自动换成 Main API 会话，避免
            # 要求运维人员手工清理历史线程映射。
            if exc.status_code not in {403, 404}:
                raise
            conversation = await self._main_api_client.create_conversation(
                payload={
                    "agent_id": str(workspace.agent_id),
                    "title": "Slack 对话",
                    "retention_policy": "DEFAULT",
                }
            )
            replacement_id = UUID(str(conversation["conversation_id"]))
            async with self._uow_factory() as uow:
                current_thread = await uow.slack.get_thread(
                    workspace_id=workspace_id,
                    channel_id=channel_id,
                    root_thread_ts=root_thread_ts,
                    slack_user_id=slack_user_id,
                )
                if current_thread is None:
                    raise RuntimeError("Slack Thread 映射不存在")
                current_thread.conversation_id = replacement_id
                await uow.commit()
            conversation_id = replacement_id
        receipt = await self._main_api_client.create_conversation_turn(
            conversation_id=conversation_id,
            payload={
                "input": message_text,
                "expected_conversation_version": conversation["row_version"],
                # 与 KM Portal 前端保持一致：检索范围、执行模式与安全
                # 等级均由 Main API 决定，Slack 不做任何路由判断。
                "client_metadata": {
                    "source": "SLACK",
                    "workspace_id": workspace_id,
                    "channel_id": channel_id,
                    "event_id": event_id,
                    "thread_ts": root_thread_ts,
                },
                "images": [],
            },
            idempotency_key=f"slack:{workspace_id}:{event_id}",
        )
        if self._config.external_callback.enabled and callback_sent_at is None:
            await self._send_external_callback(
                bot_token=workspace.require_bot_token(),
                slack_user_id=slack_user_id,
                message_text=message_text,
                workspace_id=workspace_id,
                event_id=event_id,
            )
        async with self._uow_factory() as uow:
            current = await uow.slack.get_inbox(inbox_id)
            current_thread = await uow.slack.get_thread(
                workspace_id=workspace_id,
                channel_id=channel_id,
                root_thread_ts=root_thread_ts,
                slack_user_id=slack_user_id,
            )
            current.conversation_id = conversation_id
            current.turn_id = UUID(str(receipt["turn_id"]))
            current.run_id = UUID(str(receipt["run_id"]))
            current.status = "RUNNING"
            current.callback_sent_at = (
                datetime.now(UTC)
                if self._config.external_callback.enabled
                else None
            )
            current.lease_owner = None
            current.lease_until = None
            current.updated_at = datetime.now(UTC)
            if current_thread is not None:
                current_thread.last_active_at = datetime.now(UTC)
            await uow.commit()

    async def _check_run(self, inbox_id: UUID) -> None:
        async with self._uow_factory() as uow:
            inbox = await uow.slack.get_inbox(inbox_id)
            workspace = self._config.workspace(inbox.workspace_id)
            if workspace is None or inbox.run_id is None:
                raise RuntimeError("Slack Inbox 缺少 Workspace 或 Run")
            run_id = inbox.run_id
            workspace_id = inbox.workspace_id
            channel_id = inbox.channel_id
            slack_user_id = inbox.slack_user_id
            root_thread_ts = inbox.root_thread_ts
        summary = await self._main_api_client.get_run(run_id=run_id)
        status = str(summary.get("status") or "")
        if status not in _TERMINAL_RUN_STATUSES:
            async with self._uow_factory() as uow:
                current = await uow.slack.get_inbox(inbox_id)
                current.lease_owner = None
                current.lease_until = datetime.now(UTC) + timedelta(
                    seconds=self._config.outbox_poll_interval_seconds
                )
                await uow.commit()
            return
        if status == "COMPLETED":
            artifact = await self._main_api_client.get_result(run_id=run_id)
            asset_cards = await assemble_slack_asset_cards(
                artifact=artifact,
                main_api_client=self._main_api_client,
                run_id=run_id,
                limit=self._config.reply.max_references,
            )
            payloads = render_slack_replies(
                channel_id=channel_id,
                user_id=slack_user_id,
                thread_ts=root_thread_ts,
                artifact=artifact,
                reply_config=self._config.reply,
                asset_cards=asset_cards,
            )
        else:
            payloads = [
                {
                    "channel": channel_id,
                    "thread_ts": root_thread_ts,
                    "text": (
                        f"<@{slack_user_id}> "
                        "KBot was unable to process this request. "
                        "Please try again later."
                    ),
                }
            ]
        async with self._uow_factory() as uow:
            current = await uow.slack.get_inbox(inbox_id)
            for payload in payloads[:1]:
                delivery_type = "FINAL"
                existing = await uow.slack.get_delivery(
                    inbox_id=inbox_id,
                    delivery_type=delivery_type,
                )
                if existing is None:
                    await uow.slack.add_delivery(
                        SlackDeliveryEntity(
                            delivery_id=uuid7(),
                            inbox_id=inbox_id,
                            workspace_id=workspace_id,
                            channel_id=channel_id,
                            slack_user_id=slack_user_id,
                            thread_ts=root_thread_ts,
                            delivery_type=delivery_type,
                            payload_json=payload,
                            status="PENDING",
                            attempt_count=0,
                        )
                    )
            current.status = "COMPLETED" if status == "COMPLETED" else "FAILED"
            current.lease_owner = None
            current.lease_until = None
            current.updated_at = datetime.now(UTC)
            await uow.commit()

    async def _process_one_delivery(self) -> bool:
        async with self._uow_factory() as uow:
            delivery = await uow.slack.claim_delivery(
                worker_id=self._worker_id,
                lease_seconds=self._config.lease_seconds,
            )
            if delivery is None:
                return False
            delivery_id = delivery.delivery_id
            inbox = await uow.slack.get_inbox(delivery.inbox_id)
            event_id = inbox.event_id if inbox is not None else str(delivery.inbox_id)
            inbox_id = delivery.inbox_id
            workspace_id = delivery.workspace_id
            delivery_type = delivery.delivery_type
            payload = dict(delivery.payload_json)
            await uow.commit()
        workspace = self._config.workspace(workspace_id)
        if workspace is None:
            await self._record_delivery_error(
                delivery_id,
                RuntimeError("Slack Workspace 配置已移除"),
            )
            return True
        try:
            body = await self._post_slack(
                bot_token=workspace.require_bot_token(),
                payload=payload,
                workspace_id=workspace_id,
                event_key=event_id,
                delivery_type=delivery_type.lower(),
            )
            async with self._uow_factory() as uow:
                current = await uow.slack.get_delivery(
                    inbox_id=inbox_id,
                    delivery_type=delivery_type,
                )
                current.status = "DELIVERED"
                current.slack_message_ts = str(body.get("ts") or "") or None
                current.delivered_at = datetime.now(UTC)
                current.lease_owner = None
                current.lease_until = None
                await uow.commit()
        except Exception as exc:
            await self._record_delivery_error(delivery_id, exc)
        return True

    async def _post_slack(
        self,
        *,
        bot_token: str,
        payload: dict[str, Any],
        workspace_id: str,
        event_key: str,
        delivery_type: str,
    ) -> dict[str, Any]:
        try:
            self._dump_slack_payload(
                payload, workspace_id, event_key, delivery_type
            )
        except Exception:
            logger.exception(
                "Slack 原始回复调试报文写入失败：event_id={}",
                event_key,
            )
        visible_payload = slack_visible_payload(payload)
        async with self._http_session.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=visible_payload,
        ) as response:
            body = await response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Slack chat.postMessage 失败：{body.get('error', 'unknown')}")
        return body

    async def _send_external_callback(
        self,
        *,
        bot_token: str,
        slack_user_id: str,
        message_text: str,
        workspace_id: str,
        event_id: str,
    ) -> None:
        name, email = await self._fetch_user_info(bot_token, slack_user_id)
        payload = build_callback_payload(
            user_id=slack_user_id,
            username=name,
            user_email=email,
            user_question=message_text,
            request_date=datetime.now(UTC).date(),
        )
        if self._config.debug.callback_payload_log_enabled:
            try:
                self._append_callback_debug(
                    {
                        "logged_at": datetime.now(UTC).isoformat(),
                        "workspace_id": workspace_id,
                        "event_id": event_id,
                        "callback_url": self._config.external_callback.url,
                        "callback_payload": payload,
                    }
                )
            except Exception:
                logger.exception(
                    "Slack Callback 调试报文写入失败：event_id={}",
                    event_id,
                )
        try:
            async with self._http_session.post(
                self._config.external_callback.url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self._config.external_callback.timeout_seconds
                ),
            ) as response:
                if 200 <= response.status < 300:
                    logger.info(
                        "Slack external callback 调用成功：event_id={} status={}",
                        event_id,
                        response.status,
                    )
                else:
                    logger.warning(
                        "Slack external callback 调用失败：event_id={} status={}",
                        event_id,
                        response.status,
                    )
        except Exception:
            # 与 3.3 保持旁路通知语义，Callback 故障不能中断 Slack 问答。
            logger.exception(
                "Slack external callback 调用异常：event_id={}",
                event_id,
            )

    async def _fetch_user_info(self, token: str, user_id: str) -> tuple[str, str]:
        try:
            async with self._http_session.get(
                "https://slack.com/api/users.info",
                headers={"Authorization": f"Bearer {token}"},
                params={"user": user_id},
            ) as response:
                body = await response.json()
        except Exception:
            logger.exception("Slack users.info 调用异常：user_id={}", user_id)
            return "", ""
        if not body.get("ok"):
            logger.warning(
                "Slack users.info 失败：user_id={} error={}",
                user_id,
                body.get("error"),
            )
            return "", ""
        user = body.get("user") or {}
        profile = user.get("profile") or {}
        return str(user.get("real_name") or ""), str(profile.get("email") or "")

    def _dump_slack_payload(
        self,
        payload: dict[str, Any],
        workspace_id: str,
        event_key: str,
        delivery_type: str,
    ) -> None:
        if not self._config.debug.slack_reply_dump_enabled:
            return
        directory = Path(self._config.debug.slack_reply_dump_dir)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise RuntimeError("Slack 回复调试目录不能是符号链接")
        os.chmod(directory, 0o700)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        filename = (
            f"{_safe_name(workspace_id)}_{_safe_name(event_key)}_"
            f"{_safe_name(delivery_type)}_{timestamp}.json"
        )
        path = directory / filename
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    def _append_callback_debug(self, record: dict[str, Any]) -> None:
        path = self._callback_debug_log_path
        if path is None:
            raise RuntimeError("Slack Callback 调试日志路径未配置")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _record_inbox_error(self, inbox_id: UUID, exc: Exception) -> None:
        async with self._uow_factory() as uow:
            inbox = await uow.slack.get_inbox(inbox_id)
            inbox.error_code = type(exc).__name__[:64]
            inbox.error_message = str(exc)[:2000]
            inbox.attempt_count += 1
            inbox.lease_owner = None
            if inbox.attempt_count >= self._config.max_delivery_attempts:
                inbox.status = "FAILED"
                inbox.lease_until = None
                final = await uow.slack.get_delivery(
                    inbox_id=inbox_id,
                    delivery_type="FINAL",
                )
                if final is None:
                    await uow.slack.add_delivery(
                        SlackDeliveryEntity(
                            delivery_id=uuid7(),
                            inbox_id=inbox.inbox_id,
                            workspace_id=inbox.workspace_id,
                            channel_id=inbox.channel_id,
                            slack_user_id=inbox.slack_user_id,
                            thread_ts=inbox.root_thread_ts,
                            delivery_type="FINAL",
                            payload_json={
                                "channel": inbox.channel_id,
                                "thread_ts": inbox.root_thread_ts,
                                "text": (
                                    f"<@{inbox.slack_user_id}> "
                                    "KBot was unable to process this request. "
                                    "Please try again later."
                                ),
                            },
                            status="PENDING",
                            attempt_count=0,
                        )
                    )
            else:
                delay = min(300, 2 ** min(inbox.attempt_count, 8))
                inbox.lease_until = datetime.now(UTC) + timedelta(seconds=delay)
            await uow.commit()

    async def _record_delivery_error(self, delivery_id: UUID, exc: Exception) -> None:
        async with self._uow_factory() as uow:
            delivery = await uow.session.get(SlackDeliveryEntity, delivery_id)
            delivery.error_code = type(exc).__name__[:128]
            delivery.error_message = str(exc)[:2000]
            delivery.lease_owner = None
            delivery.lease_until = None
            if delivery.attempt_count >= self._config.max_delivery_attempts:
                delivery.status = "FAILED"
            else:
                delay = min(300, 2 ** min(delivery.attempt_count, 8))
                delivery.next_attempt_at = datetime.now(UTC) + timedelta(seconds=delay)
            await uow.commit()
