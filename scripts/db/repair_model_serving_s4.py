"""将已有本地 Oracle 模型目录对齐到 S4 行版本结构。"""

import asyncio

from sqlalchemy import text

from platform_core.database import create_database_runtime


async def main() -> None:
    runtime = create_database_runtime()
    try:
        async with runtime.session_factory() as session:
            exists = int((await session.execute(text(
                "SELECT COUNT(*) FROM USER_TAB_COLUMNS "
                "WHERE TABLE_NAME = 'KBOT_AI_MODEL' "
                "AND COLUMN_NAME = 'ROW_VERSION'"
            ))).scalar_one())
            if not exists:
                await session.execute(text(
                    "ALTER TABLE KBOT_AI_MODEL ADD ("
                    "ROW_VERSION NUMBER(19) DEFAULT 1 NOT NULL)"
                ))
            invalid = int((await session.execute(text(
                "SELECT COUNT(*) FROM KBOT_AI_MODEL "
                "WHERE ROW_VERSION IS NULL OR ROW_VERSION < 1"
            ))).scalar_one())
            if invalid:
                raise RuntimeError("模型目录存在无效 ROW_VERSION")
            await session.commit()
        print("Model Serving S4 Oracle 结构对齐完成：ROW_VERSION 可用")
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
