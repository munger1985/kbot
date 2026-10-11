"""故障进程状态文件。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state(path: Path, **values: Any) -> None:
    """用原子替换写入不含凭据的进程状态。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def print_fault_status(path: Path, database: str, running: bool) -> None:
    """输出适合人工查看的单库故障状态。"""

    state = read_state(path)
    if state is None:
        print(f"{database}当前没有故障模拟记录。")
        return
    state_name = str(state.get("status", "unknown"))
    if not running and state_name in {"starting", "active", "stopping"}:
        state_name = "进程已退出"
    labels = {
        "starting": "正在启动",
        "active": "运行中",
        "stopping": "正在停止",
        "stopped": "已停止并恢复",
        "expired": "已到期并自动恢复",
        "failed": "启动或运行失败",
    }
    label = labels.get(state_name, state_name)
    print(
        f"数据库={database}，状态={label}，故障类型={state.get('fault_type', '未知')}，"
        f"运行标识={state.get('run_id', '未知')}，PID={state.get('pid', '未知')}，"
        f"安全时限={state.get('ttl_seconds', '未知')}秒"
    )
    if state.get("finished_at"):
        print(f"结束时间={state['finished_at']}")
