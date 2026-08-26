from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


health_file = Path("/var/lib/kbot/oracle-alert/health.json")
if not health_file.is_file():
    raise SystemExit(1)
payload = json.loads(health_file.read_text(encoding="utf-8"))
checked_at = datetime.fromisoformat(payload["checked_at"])
if checked_at.tzinfo is None:
    checked_at = checked_at.replace(tzinfo=timezone.utc)
fresh = checked_at >= datetime.now(timezone.utc) - timedelta(minutes=2)
sys.exit(0 if payload.get("status") == "healthy" and fresh else 1)
