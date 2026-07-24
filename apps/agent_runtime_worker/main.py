"""Agent Runtime 无状态 Worker 独立入口。"""

import asyncio
import signal

import aiohttp
from loguru import logger

from agent_runtime.application import (
    AgentDelegationReconciler,
    AgentRuntimeService,
)
from agent_runtime.config import get_agent_runtime_settings
from agent_runtime.domain.planning import PlanLimits, PlanValidator
from agent_runtime.domain.skills import SkillRegistry
from agent_runtime.persistence import create_agent_runtime_uow
from agent_runtime.runtime import AgentRuntimeWorker
from agent_runtime.specialists import register_builtin_skills
from platform_clients import (
    AIModelClient,
    AIOpsClientAuth,
    AIOpsDelegationClient,
    KnowledgeCoreClient,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager
from platform_core.security import (
    create_auth_context_codec,
    create_service_identity_codec,
)


async def main() -> None:
    settings = get_agent_runtime_settings()
    config = settings.worker
    LogManager(LogConfig(
        service_name=config.service_name,
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    db_runtime = create_database_runtime()
    client_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=max(
                settings.knowledge_core.timeout_seconds,
                settings.aiops.timeout_seconds,
            )
        )
    )
    try:
        knowledge_core_client = KnowledgeCoreClient(
            base_url=settings.knowledge_core.base_url,
            caller_service=config.service_name,
            audience=settings.knowledge_core.audience,
            timeout_seconds=settings.knowledge_core.timeout_seconds,
            session=client_session,
        )
        model_client = AIModelClient(
            caller_service=config.service_name,
            llm_config=settings.llm,
        )
        aiops_client = AIOpsDelegationClient(
            base_url=settings.aiops.base_url,
            auth=AIOpsClientAuth(
                caller_service=config.service_name,
                audience=settings.aiops.audience,
                scopes=("aiops.delegate",),
                auth_context_codec=create_auth_context_codec(),
                service_identity_codec=create_service_identity_codec(),
            ),
            timeout_seconds=settings.aiops.timeout_seconds,
            session=client_session,
        )
        skill_registry = register_builtin_skills(
            SkillRegistry(),
            knowledge_core_client=knowledge_core_client,
            model_client=model_client,
            service_name=config.service_name,
        )
        limits = PlanLimits(
            max_tasks=config.max_tasks_per_run,
            max_parallel_tasks=config.max_parallel_tasks,
            max_total_retries=config.max_total_retries,
            max_task_timeout_seconds=config.max_task_timeout_seconds,
        )
        runtime_service = AgentRuntimeService(
            uow_factory=create_agent_runtime_uow(
                db_runtime.session_factory
            ),
            plan_validator=PlanValidator(
                skill_exists=skill_registry.contains,
                capability_exists=lambda service, capability: (
                    service == "aiops_agent"
                    and capability == "diagnosis"
                ),
                public_artifact_types={"GROUNDED_ANSWER"},
            ),
            plan_limits=limits,
            skill_registry=skill_registry,
        )
        worker = AgentRuntimeWorker(
            runtime_service=runtime_service,
            skill_registry=skill_registry,
            worker_id=config.worker_id,
            lease_seconds=config.lease_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        reconciler = AgentDelegationReconciler(
            uow_factory=create_agent_runtime_uow(
                db_runtime.session_factory
            ),
            aiops_client=aiops_client,
            reconciler_id=f"{config.worker_id}:delegation",
            lease_seconds=config.lease_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )

        def stop_services() -> None:
            worker.stop()
            reconciler.stop()

        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stop_services)
        logger.info(
            "正在启动 Agent Runtime Worker：worker_id={}",
            config.worker_id,
        )
        await asyncio.gather(
            worker.run_forever(),
            reconciler.run_forever(),
        )
    finally:
        await client_session.close()
        await db_runtime.close()
        logger.info("Agent Runtime Worker 资源已释放")


if __name__ == "__main__":
    asyncio.run(main())
