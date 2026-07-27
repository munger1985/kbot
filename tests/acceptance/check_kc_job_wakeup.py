"""实库验收 Oracle DBMS_ALERT 提交后唤醒 KC Worker。"""

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 路径初始化必须先于项目包导入。
from knowledge_core.adapters.oracle_job_wakeup import (
    OracleDbmsAlertListener,
    OracleDbmsAlertPublisher,
)
from knowledge_core.config import get_knowledge_core_settings
from knowledge_core.ports.job_wakeup import PARSE_WAKEUP_CHANNEL
from platform_core.database.oracle import create_database_runtime


async def main() -> None:
    settings = get_knowledge_core_settings()
    runtime = create_database_runtime(settings)
    listener = OracleDbmsAlertListener(
        settings=settings,
        channel=PARSE_WAKEUP_CHANNEL,
    )
    try:
        waiting = asyncio.create_task(listener.wait(10))
        await asyncio.sleep(0.25)
        async with runtime.session_factory() as session:
            async with session.begin_nested():
                await OracleDbmsAlertPublisher().signal(
                    session,
                    {PARSE_WAKEUP_CHANNEL},
                )
            await session.commit()
        awakened = await asyncio.wait_for(waiting, timeout=5)
        if not awakened:
            raise RuntimeError("DBMS_ALERT 已返回，但未收到唤醒信号")
        print("KC DBMS_ALERT 提交后唤醒验证通过")
    finally:
        await listener.close()
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
