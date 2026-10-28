"""MySQL 保留查询连接所有权后才取消，禁止对可复用的外部会话编号盲目 KILL。"""

import asyncio
from contextlib import asynccontextmanager
from uuid import UUID

from loguru import logger


MYSQL_CANCEL_TIMEOUT_SECONDS = 3.0
MYSQL_LOCAL_CLEANUP_SECONDS = 1.0


class OwnedMySQLExecution:
    def __init__(self):
        self.connection = None
        self.server_uuid = None
        self.session_id = None
        self.stopping = False
        self.stop_requested = asyncio.Event()
        self.termination_confirmed = False
        self.expired = asyncio.Event()
        self.query_finished = asyncio.Event()
        self.release_connection = asyncio.Event()

    def attach(self, connection, identity):
        self.connection = connection
        self.server_uuid = dict(identity).get("server_uuid")
        self.session_id = connection.thread_id()

    def ensure_active(self):
        if self.stopping:
            raise asyncio.CancelledError

    async def pause(self, seconds):
        try:
            await asyncio.wait_for(self.stop_requested.wait(), seconds)
        except TimeoutError:
            return
        raise asyncio.CancelledError

    @asynccontextmanager
    async def statement_window(self, seconds):
        # 定时器通知外层；不能先取消 aiomysql 的读包协程，否则它会提前关闭原连接。
        handle = asyncio.get_running_loop().call_later(seconds, self.expired.set)
        try:
            yield
        finally:
            handle.cancel()

    async def finish(self, connection):
        self.query_finished.set()
        try:
            if self.stopping:
                # 查询自然结束与取消并发时仍持有原连接，防止自身关闭导致编号被复用。
                await self.release_connection.wait()
        finally:
            connection.close()

    async def run(self, operation, *, open_control_connection):
        task = asyncio.create_task(operation(self), name="aiops-mysql-owned-query")
        timeout = asyncio.create_task(self.expired.wait(), name="aiops-mysql-statement-deadline")
        try:
            await asyncio.wait((task, timeout), return_when=asyncio.FIRST_COMPLETED)
            if self.expired.is_set():
                raise TimeoutError
            return await task
        except (asyncio.CancelledError, TimeoutError):
            self.stopping = True
            self.stop_requested.set()
            try:
                await self._stop_query(open_control_connection)
            finally:
                self.release_connection.set()
                # KILL 的结果已收敛或已明确未知后，才取消本地读包和释放原连接。
                if not task.done():
                    task.cancel()
                async with asyncio.timeout(MYSQL_LOCAL_CLEANUP_SECONDS):
                    await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            timeout.cancel()
            await asyncio.gather(timeout, return_exceptions=True)

    async def _stop_query(self, open_control_connection):
        connection = self.connection
        control = None
        try:
            # 没有完整原生身份不发 KILL；原连接已关闭也不再使用可能被复用的编号。
            if (connection is None or connection.closed or not self.server_uuid
                or isinstance(self.session_id, bool) or not isinstance(self.session_id, int)
                or not 0 < self.session_id < 2**64 or connection.thread_id() != self.session_id):
                logger.warning("MySQL 只读查询清理：termination_status=TERMINATION_UNCONFIRMED")
                return
            async with asyncio.timeout(MYSQL_CANCEL_TIMEOUT_SECONDS):
                control = await open_control_connection()
                async with control.cursor() as cursor:
                    await cursor.execute("SELECT @@server_uuid")
                    rows = await cursor.fetchmany(1)
                    if (len(rows) != 1 or len(rows[0]) != 1
                        or str(UUID(str(rows[0][0]))) != self.server_uuid
                        or connection.closed or connection.thread_id() != self.session_id):
                        logger.warning("MySQL 只读查询清理：termination_status=OWNERSHIP_UNCONFIRMED")
                        return
                    # 同凭据、同物理实例、仍由本次操作持有的独占连接；没有用户可传入的会话号。
                    await cursor.execute(f"KILL QUERY {self.session_id}")
                await self.query_finished.wait()
            self.termination_confirmed = True
            logger.info("MySQL 只读查询清理：termination_status=QUERY_TERMINATION_ACKNOWLEDGED")
        except asyncio.CancelledError:
            logger.warning("MySQL 只读查询清理：termination_status=TERMINATION_UNCONFIRMED")
            raise
        except Exception:
            # 不重试 KILL；网络/权限/代理身份不能确认时不能假设目标语句已停止。
            logger.warning("MySQL 只读查询清理：termination_status=TERMINATION_UNCONFIRMED")
        finally:
            if control is not None:
                control.close()
