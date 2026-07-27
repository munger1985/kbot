"""跨服务领域标识生成。"""

from __future__ import annotations

import secrets
import threading
import time
from uuid import UUID


_UUID7_RANDOM_MASK = (1 << 74) - 1
_uuid7_lock = threading.Lock()
_last_unix_ms = -1
_last_random = 0


def uuid7() -> UUID:
    """生成单进程内单调递增的 RFC 9562 UUIDv7。"""
    global _last_random, _last_unix_ms

    unix_ms = time.time_ns() // 1_000_000
    with _uuid7_lock:
        if unix_ms > _last_unix_ms:
            _last_unix_ms = unix_ms
            _last_random = secrets.randbits(74)
        else:
            unix_ms = _last_unix_ms
            _last_random = (_last_random + 1) & _UUID7_RANDOM_MASK
            if _last_random == 0:
                _last_unix_ms += 1
                unix_ms = _last_unix_ms

        random_a = _last_random >> 62
        random_b = _last_random & ((1 << 62) - 1)

    value = (
        (unix_ms & ((1 << 48) - 1)) << 80
        | 0x7 << 76
        | random_a << 64
        | 0b10 << 62
        | random_b
    )
    return UUID(int=value)
