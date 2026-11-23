"""按已登记资源装配本机备份和独立主机演练。"""

import asyncio
from contextlib import asynccontextmanager

from aiops_agent.adapters.file_local_backup import FileLocalBackupBackend
from aiops_agent.adapters.local_drill import LocalDrillBackend, machine_id
from aiops_agent.adapters.oracle_local_backup import OracleLocalBackupBackend
from aiops_agent.application.data_protection_execution import DrillRejected
from aiops_agent.workers.data_protection import DataProtectionWorker


@asynccontextmanager
async def local_drill_worker(app, runtime, backend, *, interval_seconds):
    """在同一 Worker 入口只运行数据保护循环，不装配普通任务。"""
    component = DataProtectionWorker(
        uow_factory=runtime.uow_factory, backend=backend, interval_seconds=interval_seconds,
    )
    await runtime.start()
    task = asyncio.create_task(component.run_forever(), name="aiops-data-protection")

    async def ready_check():
        checks = await runtime.check_database_connection()
        checks["component:data-protection"] = "failed" if task.done() else "ok"
        return checks

    app.state.ready_check = ready_check
    try:
        yield
    finally:
        component.stop()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


class LocalProtectionBackend:
    def __init__(self, *backends, drill=None):
        self._routes = {}
        for backend in backends:
            for resource_id in backend.backup_resource_ids:
                if resource_id in self._routes:
                    raise ValueError("同一数据保护资源不能登记多个本机备份执行位置")
                self._routes[resource_id] = backend
        self.backup_resource_ids = frozenset(self._routes)
        self._drill = drill
        self.drill_resource_ids = drill.drill_resource_ids if drill else frozenset()

    def _backend(self, resource_id):
        backend = self._routes.get(resource_id)
        if backend is None:
            raise DrillRejected("该资源未登记本机备份执行位置")
        return backend

    async def preflight(self, *, scope, actor_id, resource_id, settings, operation):
        if operation == "DRILL" and resource_id in self.drill_resource_ids:
            return await self._drill.preflight(
                scope=scope, actor_id=actor_id, resource_id=resource_id, settings=settings, operation=operation,
            )
        if operation != "BACKUP":
            raise DrillRejected("该资源未登记当前主机的演练入口")
        return await self._backend(resource_id).preflight(
            scope=scope, actor_id=actor_id, resource_id=resource_id, settings=settings, operation=operation,
        )

    async def backup(self, work, invocation):
        if work.operation != "BACKUP":
            raise DrillRejected("本机备份适配不提供恢复入口")
        return await self._backend(work.resource_id).backup(work, invocation)

    async def drill_runner(self, work):
        if work.resource_id not in self.drill_resource_ids:
            raise DrillRejected("该资源未登记当前主机的演练入口")
        return await self._drill.drill_runner(work)


def local_protection_backend(*, settings, uow_factory, credentials, worker=False):
    oracle = settings.oracle_local_backups
    files = settings.file_local_backups
    registered = getattr(settings, "local_drills", {})
    local_id = machine_id() if worker and registered else None
    drills = {key: binding for key, binding in registered.items() if binding.machine_id == local_id}
    if drills:
        # 演练主机只持有本机演练入口，不装配源端备份适配。
        return LocalProtectionBackend(
            drill=LocalDrillBackend(bindings=drills, uow_factory=uow_factory, credentials=credentials),
        )
    if not oracle and not files:
        return None
    return LocalProtectionBackend(
        OracleLocalBackupBackend(bindings=oracle, uow_factory=uow_factory, credentials=credentials),
        FileLocalBackupBackend(bindings=files, uow_factory=uow_factory, credentials=credentials),
    )
