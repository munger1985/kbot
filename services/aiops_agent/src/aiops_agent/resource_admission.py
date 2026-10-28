"""共享准入的短退避；不持 UoW、不重试已经执行的数据库查询。"""

import asyncio


RETRYABLE_RESOURCE_CODES = frozenset({
    "RESOURCE_QUOTA_GLOBAL_EXHAUSTED", "RESOURCE_QUOTA_TARGET_EXHAUSTED",
    "RESOURCE_QUOTA_SOURCE_EXHAUSTED", "RESOURCE_QUOTA_UNAVAILABLE",
})
ADMISSION_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.5, 0.5)


async def wait_for_resource(operation, *, retry_errors):
    """最多七次原子准入、合计1.75秒退避；调用方仍负责整个动作的绝对期限。"""
    for attempt in range(len(ADMISSION_RETRY_DELAYS) + 1):
        try:
            return await operation()
        except retry_errors as exc:
            if exc.code not in RETRYABLE_RESOURCE_CODES or attempt == len(ADMISSION_RETRY_DELAYS):
                raise
            await asyncio.sleep(ADMISSION_RETRY_DELAYS[attempt])
