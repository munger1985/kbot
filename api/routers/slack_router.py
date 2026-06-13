"""
Slack Events API router.

Exposes two endpoints that Slack calls directly:

- ``POST /slack/events`` — Event API callbacks (messages, mentions, etc.)
- ``POST /slack/interactions`` — Interactive component callbacks

Key design decisions:
- **No FastAPI auth dependency.** Slack uses its own HMAC-SHA256 signing.
- **No ``/api`` prefix.** Slack's Event Subscription URL must be an exact
  path without a prefix.
- **Returns 200 OK immediately.** The agent call + Slack reply happen in a
  background ``asyncio.create_task`` so we stay within Slack's 3‑second
  response window.
"""

import json

from fastapi import APIRouter, Request, Response
from loguru import logger

from api.controllers.slack_controller import slack_controller

router = APIRouter(tags=["Slack Integration"])


@router.post("/slack/events")
async def slack_events(request: Request):
    """Primary endpoint for Slack Events API callbacks.

    Handles:

    * ``url_verification`` — Returns the challenge value as ``text/plain``
      (no signature verification required by Slack for this payload type).
    * ``event_callback`` — Signature verification, deduplication, and
      background agent processing.
    """
    raw_body = await request.body()
    body_str = raw_body.decode("utf-8")

    # Normalize header keys to lowercase for case-insensitive access.
    headers = {k.lower(): v for k, v in request.headers.items()}

    # --- Parse JSON ---
    try:
        payload = json.loads(body_str)
    except json.JSONDecodeError:
        logger.error("Slack events: invalid JSON body")
        return Response("Bad Request", status_code=400)

    # --- Delegate ---
    ok, challenge = await slack_controller.handle_event(body_str, headers, payload)

    if challenge is not None:
        # URL verification — return the challenge value as plain text.
        return Response(content=challenge, media_type="text/plain")

    if not ok:
        return Response("Unauthorized", status_code=401)

    return Response("OK", status_code=200)


@router.post("/slack/interactions")
async def slack_interactions(request: Request):
    """Endpoint for Slack interactive component callbacks.

    Slack sends interaction payloads (block_actions, view_submission, etc.)
    as ``application/x-www-form-urlencoded`` with a ``payload`` field
    containing the JSON body.
    """
    form_data = await request.form()
    raw_payload = form_data.get("payload")

    if not raw_payload:
        return Response("Bad Request", status_code=400)

    headers = {k.lower(): v for k, v in request.headers.items()}

    ok = await slack_controller.handle_interaction(str(raw_payload), headers)
    if not ok:
        return Response("Unauthorized", status_code=401)

    return Response("OK", status_code=200)
