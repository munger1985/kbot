"""
Slack Events API controller.

Handles signature verification, event parsing, deduplication, agent-driven
replies, and outbound Slack API communication via aiohttp.

The request flow:
1. Slack POSTs an event to ``/slack/events``
2. Router parses the raw body and headers, delegates here
3. ``handle_event`` verifies, parses, deduplicates, and launches a background
   ``asyncio.create_task`` to call the kbot agent and reply to Slack
4. The HTTP handler returns 200 OK immediately (under 3 seconds)
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
from uuid import uuid4

import aiohttp
from fastapi import BackgroundTasks
from loguru import logger

from api.controllers.agent_controller import agent_controller
from api.schemas.agent_schema import AgentChatForm
from api.schemas.base_response import SuccessResponse
from core.config.settings import get_slack_config
from utils.sse import parse_sse_doc_results, parse_sse_for_answer
from utils.thread import detect_language

# ---------------------------------------------------------------------------
# In-memory deduplication sets
# ---------------------------------------------------------------------------
# Slack may deliver the same event_id multiple times (retries, edge cases).
_seen_event_ids: set[str] = set()
_MAX_SEEN_EVENTS = 10_000

# Prevent duplicate replies when the same user message triggers both a
# ``message`` event and an ``app_mention`` event.
_replied_keys: set[str] = set()
_MAX_REPLIED_KEYS = 5_000

# ---------------------------------------------------------------------------
# Signature verification (migrated from test-slack/demo_signature.py)
# ---------------------------------------------------------------------------

def verify_slack_signature(
    signing_secret: str,
    timestamp: str,
    signature: str,
    raw_body: str,
    max_age_seconds: int = 300,
) -> bool:
    """Verify a Slack-signed request using HMAC-SHA256.

    1. Reject timestamps older than *max_age_seconds* (replay protection).
    2. Build the expected signature string ``v0:{timestamp}:{body}``.
    3. Compute HMAC-SHA256 and compare with constant-time comparison.

    Args:
        signing_secret: Slack App Signing Secret from app configuration.
        timestamp: ``X-Slack-Request-Timestamp`` header value.
        signature: ``X-Slack-Signature`` header value (starts with ``v0=``).
        raw_body: Raw request body as a string.
        max_age_seconds: Maximum allowed clock drift before rejecting.

    Returns:
        ``True`` if the signature is valid.
    """
    # --- Replay attack check ---
    try:
        req_ts = int(timestamp)
    except (ValueError, TypeError):
        logger.warning("Slack verification: invalid timestamp format")
        return False

    drift = abs(int(time.time()) - req_ts)
    if drift > max_age_seconds:
        logger.warning(f"Slack verification: timestamp expired (drift={drift}s)")
        return False

    # --- HMAC-SHA256 computation ---
    basestring = f"v0:{timestamp}:{raw_body}"
    expected = "v0=" + hmac.new(
        key=signing_secret.encode("utf-8"),
        msg=basestring.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        logger.warning("Slack verification: signature mismatch")
        return False

    return True


# ---------------------------------------------------------------------------
# Event deduplication helpers
# ---------------------------------------------------------------------------

def _is_duplicate_event(event_id: str) -> bool:
    """Return ``True`` if *event_id* was already processed."""
    if event_id in _seen_event_ids:
        return True
    _seen_event_ids.add(event_id)
    if len(_seen_event_ids) > _MAX_SEEN_EVENTS:
        _seen_event_ids.clear()
    return False


def _mark_replied(channel_id: str, user_id: str, event_ts: str) -> bool:
    """Return ``True`` if this (channel, user, approx-ts) was already replied to.

    If not seen before, record it and return ``False``.
    """
    # Use the whole-second portion of the timestamp to tolerate sub-second
    # variation between ``message`` and ``app_mention`` events for the same
    # user utterance.
    base_ts = event_ts.split(".")[0] if "." in event_ts else event_ts
    key = f"{channel_id}:{base_ts}:{user_id}"

    if key in _replied_keys:
        return True
    _replied_keys.add(key)
    if len(_replied_keys) > _MAX_REPLIED_KEYS:
        _replied_keys.clear()
    return False


# ---------------------------------------------------------------------------
# Slack API outbound communication
# ---------------------------------------------------------------------------

async def _send_slack_reply(
    bot_token: str,
    channel_id: str,
    user_id: str,
    text: str,
    thread_ts: str | None = None,
    blocks: list[dict] | None = None,
) -> bool:
    """Post a threaded reply to Slack via ``chat.postMessage``.

    Args:
        bot_token: Slack Bot User OAuth token (xoxb-...).
        channel_id: Slack channel ID.
        user_id: Slack user ID to @-mention.
        text: Message body (the agent's answer).
        thread_ts: If set, reply in-thread to the message with this timestamp.
        blocks: Optional Slack Block Kit blocks appended after *text*.

    Returns:
        ``True`` if Slack accepted the message.
    """
    payload: dict = {
        "channel": channel_id,
        "text": f"<@{user_id}> {text}",
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts
    if blocks:
        payload["blocks"] = blocks

    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    cfg = get_slack_config()
    timeout = aiohttp.ClientTimeout(total=cfg.api_timeout)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://slack.com/api/chat.postMessage",
                headers=headers,
                json=payload,
            ) as resp:
                body = await resp.json()
                if body.get("ok"):
                    logger.info(
                        "Slack reply sent | channel={} | user={} | thread={}",
                        channel_id,
                        user_id,
                        thread_ts or "N/A",
                    )
                    return True
                else:
                    logger.error(
                        "Slack chat.postMessage error: {} | details={} | "
                        "block_count={}",
                        body.get("error", "unknown"),
                        body.get("errors", ""),
                        len(payload.get("blocks", [])),
                    )
                    return False
    except aiohttp.ClientConnectorError as e:
        logger.error("Slack API connection failed: {}", str(e))
        return False
    except aiohttp.ServerTimeoutError:
        logger.error("Slack API request timed out after {}s", cfg.api_timeout)
        return False
    except Exception as e:
        logger.error("Slack API unexpected error: {}", str(e))
        return False


# ---------------------------------------------------------------------------
# Event parsing (migrated from test-slack/slack_event_server.py)
# ---------------------------------------------------------------------------

def parse_slack_event(payload: dict) -> dict | None:
    """Extract key fields from a Slack Events API payload.

    Only human-sent message-like events (``message``, ``app_mention``) are
    returned.  Every guard from the original ``test-slack/slack_event_server.py``
    ``send_reply_to_user`` is mirrored here to prevent infinite reply loops.

    Args:
        payload: Parsed JSON body of the Slack request.

    Returns:
        A dict with keys ``event_id``, ``user_id``, ``channel_id``, ``text``,
        ``event_ts``, ``event_type``, ``subtype``; or ``None`` if the event
        should be ignored.
    """
    if payload.get("type") != "event_callback":
        return None

    event = payload.get("event", {})

    # ── 1. 不回复机器人消息 ──────────────────────────────────
    # subtype=bot_message 明确标记了由机器人发出的消息。
    # 这是 Slack 原生的反死循环机制——所有通过 chat.postMessage (bot token)
    # 发出的消息都会带上此标记。
    subtype = event.get("subtype", "")
    if subtype == "bot_message":
        logger.debug("Skipping bot_message event")
        return None

    # bot_id 兜底：部分边缘场景下 subtype 可能缺失但 bot_id 存在。
    if event.get("bot_id"):
        logger.debug("Skipping event from bot (bot_id={})", event.get("bot_id"))
        return None

    # 跳过消息编辑（message_changed），只处理新消息。
    if subtype == "message_changed":
        logger.debug("Skipping message_changed event (edit, not new message)")
        return None

    # ── 2. 只处理消息和 @提及事件 ────────────────────────────
    event_type = event.get("type", "")
    if not (event_type.startswith("message") or event_type == "app_mention"):
        return None

    channel_id = event.get("channel", "")
    text = event.get("text", "")

    if not channel_id or not text:
        return None

    # ── 3. 必须有 event_ts，用于后续去重和线程回复 ────────────
    event_ts = event.get("event_ts", "")
    if not event_ts:
        logger.debug("Skipping event without event_ts")
        return None

    return {
        "event_id": payload.get("event_id", ""),
        "user_id": event.get("user", ""),
        "channel_id": channel_id,
        "text": text,
        "event_ts": event_ts,
        "event_type": event_type,
        "subtype": subtype,
    }


# ---------------------------------------------------------------------------
# Background agent processing (fire-and-forget)
# ---------------------------------------------------------------------------

# Waiting / "please wait" message, localised by the user's question language.
_WAITING_MESSAGES: dict[str, str] = {
    "zh": "您的问题 KM 助手正在搜集材料分析中，请稍等.",
    "ja": "ご質問の内容について、KMアシスタントが情報を収集し分析しています。少々お待ちください。",
    "ko": "문의하신 내용에 대해 KM 어시스턴트가 자료를 수집하고 분석 중입니다. 잠시만 기다려 주세요.",
    "th": "KM Assistant กำลังรวบรวมข้อมูลและวิเคราะห์คำถามของท่าน กรุณารอสักครู่",
    "hi": "आपके प्रश्न का विश्लेषण करने के लिए KM असिस्टेंट सामग्री एकत्र कर रहा है, कृपया प्रतीक्षा करें।",
    "vi": "Trợ lý KM đang thu thập tài liệu và phân tích câu hỏi của bạn, vui lòng đợi trong giây lát.",
    "en": "KM Assistant is gathering materials and analyzing your question, please wait.",
}


def _build_asset_blocks(biz_metadata_list: list[dict]) -> list[dict]:
    """Build Slack Block Kit blocks from doc_results biz_metadata.

    Deduplicates by ``asset_title``.  Each field independently checked —
    empty fields cause their corresponding block or entry to be omitted.
    """
    if not biz_metadata_list:
        return []

    seen_titles: set[str] = set()
    blocks: list[dict] = []

    for bm in biz_metadata_list:
        title = (bm.get("asset_title") or "").strip()
        briefing = (bm.get("solution_briefing") or "").strip()
        url = (bm.get("original_asset_url") or "").strip()
        contributor = (bm.get("contributor") or "").strip()

        if not any([title, briefing, url, contributor]):
            continue

        if title and title in seen_titles:
            continue
        if title:
            seen_titles.add(title)
            # Limit to first 3 assets.
            if len(seen_titles) > 3:
                break

        blocks.append({"type": "divider"})

        if title:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Asset Title:* {title}",
                },
            })

        if briefing:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Solution Briefing:* {briefing}",
                },
            })

        has_contributor = bool(contributor)
        has_url = bool(url)
        if has_contributor or has_url:
            section: dict = {"type": "section"}
            fields: list[dict] = []

            if has_contributor:
                fields.append({
                    "type": "mrkdwn",
                    "text": (
                        f"*Contributor:*\n"
                        f"<mailto:{contributor}|{contributor}>"
                    ),
                })

            if has_url:
                fields.append({
                    "type": "mrkdwn",
                    "text": "[VPN required] please visit us:",
                })
                section["accessory"] = {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "KM Link"},
                    "url": url,
                    "action_id": "open_km_resource",
                }

            section["fields"] = fields
            blocks.append(section)

    return blocks


def _get_waiting_message(text: str) -> str:
    """Return a localised "please wait" message matching the language of *text*."""
    lang = detect_language(text)
    return _WAITING_MESSAGES.get(lang, _WAITING_MESSAGES["en"])


async def _process_event_background(parsed: dict) -> None:
    """Call the kbot agent and send the reply to Slack.

    Runs inside ``asyncio.create_task`` after the HTTP handler has already
    returned ``200 OK``. This function is entirely self-contained:

    1. Detect the user's language and send an immediate "please wait" reply.
    2. Build an ``AgentChatForm`` from the parsed Slack event.
    3. Call ``agent_controller.agent_chat_nonstream`` (direct Python call).
    4. Extract clean answer text from the raw SSE output.
    5. Send the final answer back to Slack via ``chat.postMessage``.
    6. Manually run background tasks for memory persistence.
    """
    # --- Resolve secrets ---
    slack_cfg = get_slack_config()
    bot_token = os.getenv("SLACK_BOT_TOKEN") or slack_cfg.bot_token

    if not bot_token:
        logger.error("SLACK_BOT_TOKEN is not configured — cannot reply to Slack")
        return

    # --- 1. Send immediate "please wait" message (in thread) ---
    waiting_text = _get_waiting_message(parsed["text"])
    await _send_slack_reply(
        bot_token=bot_token,
        channel_id=parsed["channel_id"],
        user_id=parsed["user_id"],
        text=waiting_text,
        thread_ts=parsed["event_ts"],
    )

    # --- 2. Build agent request ---
    # Each Slack user gets their own session so conversation context is
    # scoped per-user rather than per-channel.
    form = AgentChatForm(
        session_id=f"slack_{parsed['user_id']}",
        by=parsed["user_id"],
        agent_id=slack_cfg.agent_id,
        question=parsed["text"],
        security_level=0,
        tags=[],
    )

    # --- Call the agent ---
    # We pass our own BackgroundTasks instance because we are in a fire-and-
    # forget context where FastAPI's request-scoped background-task lifecycle
    # does not apply. Tasks (memory persistence) are manually executed below.
    bt = BackgroundTasks()
    try:
        response = await agent_controller.agent_chat_nonstream(form, bt)
    except Exception:
        logger.exception(
            "Agent chat failed for Slack user {} | question: {}",
            parsed["user_id"],
            parsed["text"][:80],
        )
        return

    # --- Run background tasks for memory persistence ---
    # Starlette's BackgroundTasks stores BackgroundTask objects, each with
    # .func, .args, .kwargs attributes.
    if bt.tasks:
        for task in bt.tasks:
            try:
                if asyncio.iscoroutinefunction(task.func):
                    await task.func(*task.args, **task.kwargs)
                else:
                    task.func(*task.args, **task.kwargs)
            except Exception:
                logger.warning(
                    "Memory-persistence background task failed for Slack "
                    "session slack_{}",
                    parsed["user_id"],
                )

    # --- Extract clean answer from SSE ---
    sse_text = response.data if isinstance(response, SuccessResponse) else ""
    raw_sse = str(sse_text) if sse_text else ""
    answer = parse_sse_for_answer(raw_sse)

    if not answer:
        logger.info(
            "Empty answer from agent for Slack user {} — not replying",
            parsed["user_id"],
        )
        return

    # --- Assemble Slack message ---
    # Parse doc_results only when the answer suggests documents were found.
    asset_blocks: list[dict] = []
    if "No relevant information is available" not in answer:
        biz_list = parse_sse_doc_results(raw_sse)
        asset_blocks = _build_asset_blocks(biz_list)

    if not asset_blocks:
        # No cards — send plain text only (no blocks, so ``text`` is rendered).
        await _send_slack_reply(
            bot_token=bot_token,
            channel_id=parsed["channel_id"],
            user_id=parsed["user_id"],
            text=answer,
            thread_ts=parsed["event_ts"],
        )
        return

    # Cards present — build blocks: answer sections first, then asset cards.
    # Use ``plain_text`` to avoid escaping issues and split long text into
    # chunks that fit Slack's 3000-char section limit.
    _SLACK_TEXT_LIMIT = 3000
    answer_blocks: list[dict] = []
    for i in range(0, len(answer), _SLACK_TEXT_LIMIT):
        chunk = answer[i:i + _SLACK_TEXT_LIMIT]
        answer_blocks.append({
            "type": "section",
            "text": {"type": "plain_text", "text": chunk},
        })

    full_blocks = answer_blocks + asset_blocks

    # --- Reply to Slack ---
    await _send_slack_reply(
        bot_token=bot_token,
        channel_id=parsed["channel_id"],
        user_id=parsed["user_id"],
        text=answer,
        thread_ts=parsed["event_ts"],
        blocks=full_blocks,
    )


# ---------------------------------------------------------------------------
# Controller singleton
# ---------------------------------------------------------------------------

class SlackController:
    """High-level controller for Slack Events API requests.

    Provides two entry points:

    * ``handle_event`` — for Event API callbacks (messages, mentions, etc.)
    * ``handle_interaction`` — for interactive component callbacks

    Neither method requires kbot authentication — Slack performs its own
    request signing.
    """

    async def handle_event(
        self,
        raw_body: str,
        headers: dict[str, str],
        payload: dict,
    ) -> tuple[bool, str | None]:
        """Process an incoming Slack Events API callback.

        Steps (all complete within milliseconds — the heavy work is deferred
        to ``asyncio.create_task``):

        1. URL verification challenge detection.
        2. Signature verification.
        3. Event deduplication.
        4. Event parsing.
        5. Reply-key deduplication.
        6. Launch background agent processing.

        Args:
            raw_body: Raw request body string (for signature verification).
            headers: Lowercased request headers.
            payload: Parsed JSON body.

        Returns:
            ``(ok, challenge)`` tuple. When *challenge* is set the caller
            should return it directly (URL verification). When *ok* is
            ``False`` the caller should return 401.
        """
        # --- URL verification (no signature required) ---
        if payload.get("type") == "url_verification":
            challenge = payload.get("challenge", "")
            logger.info("Slack URL verification challenge received")
            return True, challenge

        # --- Signature verification ---
        slack_cfg = get_slack_config()
        signing_secret = os.getenv("SLACK_SIGNING_SECRET") or slack_cfg.signing_secret

        if signing_secret:
            timestamp = headers.get("x-slack-request-timestamp", "")
            signature = headers.get("x-slack-signature", "")
            if not verify_slack_signature(signing_secret, timestamp, signature, raw_body):
                return False, None
        else:
            logger.warning(
                "SLACK_SIGNING_SECRET is not configured — "
                "skipping signature verification (insecure)"
            )

        # --- Event deduplication ---
        event_id = payload.get("event_id", "")
        if event_id and _is_duplicate_event(event_id):
            logger.debug("Duplicate Slack event_id={}, returning OK", event_id)
            return True, None

        # --- Parse event ---
        parsed = parse_slack_event(payload)
        if parsed is None:
            return True, None  # Non-message events are silently accepted.

        logger.info(
            "Slack event | user={} | channel={} | type={} | text={}",
            parsed["user_id"],
            parsed["channel_id"],
            parsed["event_type"],
            parsed["text"][:80],
        )

        # --- Reply-key deduplication ---
        if _mark_replied(parsed["channel_id"], parsed["user_id"], parsed["event_ts"]):
            logger.debug("Already replied to this Slack message — skipping")
            return True, None

        # --- Launch background processing ---
        retry_num = headers.get("x-slack-retry-num", "")
        if retry_num:
            logger.info(
                "Slack retry (num={}) — skipping background processing",
                retry_num,
            )
        else:
            asyncio.create_task(_process_event_background(parsed))

        return True, None

    async def handle_interaction(
        self,
        raw_body: str,
        headers: dict[str, str],
    ) -> bool:
        """Process a Slack interactive component callback.

        Currently logs the interaction payload for observability. Future
        iterations can add agent-driven interactive workflows.

        Args:
            raw_body: Raw form-encoded ``payload`` value.
            headers: Lowercased request headers.

        Returns:
            ``True`` if the request was processed successfully.
        """
        slack_cfg = get_slack_config()
        signing_secret = os.getenv("SLACK_SIGNING_SECRET") or slack_cfg.signing_secret

        if signing_secret:
            timestamp = headers.get("x-slack-request-timestamp", "")
            signature = headers.get("x-slack-signature", "")
            if not verify_slack_signature(signing_secret, timestamp, signature, raw_body):
                logger.warning("Slack interaction signature verification failed")
                return False

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            logger.error("Cannot parse Slack interaction payload JSON")
            return False

        logger.info(
            "Slack interaction | type={} | user={}",
            payload.get("type"),
            payload.get("user", {}).get("id", ""),
        )
        return True


# Module-level singleton matching the existing controller pattern.
slack_controller = SlackController()
