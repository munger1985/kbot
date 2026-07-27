"""基于 Oracle DBMS_ALERT 的 KC 任务唤醒适配器。"""

from collections.abc import Set

import oracledb
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.config import Settings


class OracleDbmsAlertPublisher:
    """在当前事务中发送 Alert；Oracle 仅在提交后交付。"""

    async def signal(
        self,
        session: AsyncSession,
        channels: Set[str],
    ) -> None:
        for channel in sorted(channels):
            await session.execute(
                text(
                    "BEGIN DBMS_ALERT.SIGNAL("
                    ":channel_name, :message_text); END;"
                ),
                {
                    "channel_name": channel,
                    "message_text": "READY",
                },
            )


class OracleDbmsAlertListener:
    """使用独立 Oracle 连接等待 Alert，不参与业务事务。"""

    def __init__(self, *, settings: Settings, channel: str):
        oracle = settings.database.oracle
        self._connect_kwargs = {
            "user": oracle.username,
            "password": oracle.require_password(),
            "host": oracle.host,
            "port": oracle.port,
            "service_name": oracle.service_name,
        }
        self._channel = channel
        self._connection: oracledb.AsyncConnection | None = None

    async def _connect(self) -> oracledb.AsyncConnection:
        if self._connection is not None:
            return self._connection
        connection = await oracledb.connect_async(**self._connect_kwargs)
        cursor = connection.cursor()
        try:
            await cursor.callproc("DBMS_ALERT.REGISTER", [self._channel])
        finally:
            cursor.close()
        self._connection = connection
        return connection

    async def wait(self, timeout_seconds: float) -> bool:
        try:
            connection = await self._connect()
            cursor = connection.cursor()
            try:
                message = cursor.var(str, 1800)
                status = cursor.var(int)
                await cursor.callproc(
                    "DBMS_ALERT.WAITONE",
                    [
                        self._channel,
                        message,
                        status,
                        max(1, int(timeout_seconds)),
                    ],
                )
                status_value = status.getvalue()
                return (
                    status_value is not None
                    and int(status_value) == 0
                )
            finally:
                cursor.close()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            try:
                cursor = connection.cursor()
                try:
                    await cursor.callproc(
                        "DBMS_ALERT.REMOVE", [self._channel]
                    )
                finally:
                    cursor.close()
            except Exception:
                # 连接中断时无需再注销，Oracle 会清理该 Session 的注册。
                pass
        finally:
            await connection.close()
