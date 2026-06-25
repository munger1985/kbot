# services/basic/ops_agent_conf_service.py

from loguru import logger
from typing import Any

from dao.entities.ops_agent_conf import OpsAgentConfEntity
from dao.repositories.ops_agent_conf_repo import OpsAgentConfRepository
from core.database.oracle import get_session



class OpsAgentConfService:
    """
    智能运维智能体资产配置服务类
    提供 AIOps 机器人与物理数据库实例之间的多对多动态绑定、自愈变更权限管控以及拓扑树侦测功能。
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str
    """

    def __init__(self):
        pass

    @property
    def db_session(self):
        """获取元数据库异步会话对象"""
        return get_session()

    async def bind_instance_to_agent(
        self,
        agent_id: int,
        instance_id: str,
        username: str,
        is_mutation_allowed: bool = False,
        require_approval: bool = True,
        max_daily_execution: int = 10
    ) -> None:
        """建立智能体与物理数据库实例的管理绑定关系, 并植入控制面安全闸门参数"""
        async with self.db_session as session:
            repo = OpsAgentConfRepository(session)
            # 检查是否已经存在绑定关系
            existing = await repo.get_by_agent_and_instance(agent_id, instance_id)
            if existing:
                logger.warning(f"[OpsCMDB] 绑定关系已存在, 自动升级为更新策略模式 | Agent: {agent_id}, Instance: {instance_id}")
                await repo.update(
                    existing.id,
                    is_mutation_allowed=is_mutation_allowed,
                    require_approval=require_approval,
                    max_daily_execution=max_daily_execution,
                    updated_by=username
                )
                return

            config_entity = OpsAgentConfEntity(
                agent_id=agent_id,
                instance_id=instance_id,
                is_mutation_allowed=is_mutation_allowed,
                require_approval=require_approval,
                max_daily_execution=max_daily_execution,
                created_by=username,
                updated_by=username
            )
            await repo.create(config_entity)
            logger.success(f"[OpsCMDB] 智能体资产绑定关系创建成功 | Agent: {agent_id} -> Instance: {instance_id}")

    async def update_binding_policy(self, config_id: int, username: str, **kwargs) -> None:
        """动态调整运行期自愈控制开关"""
        async with self.db_session as session:
            repo = OpsAgentConfRepository(session)
            kwargs["updated_by"] = username
            await repo.update(config_id, **kwargs)
            logger.success(f"[OpsCMDB] 成功修正智能体自愈门禁控制策略 | ConfigID: {config_id}")

    async def unbind_instance_from_agent(self, agent_id: int, instance_id: str) -> None:
        """解除智能体与某物理实例的管理托管关系"""
        async with self.db_session as session:
            repo = OpsAgentConfRepository(session)
            await repo.unbind_instance(agent_id, instance_id)
            logger.info(f"[OpsCMDB] 成功解除智能体托管资产关系 | Agent: {agent_id}, Instance: {instance_id}")

    async def get_bound_instances_by_agent(self, agent_id: int) -> list[dict[str, Any]]:
        """
        【流式网关核心函数】
        一键拉取当前运维机器人名下的所有在线活跃物理库资产, 并将多维 JOIN 实体扁平化清洗为
        Dict 字典流, 供编排器直接装载注入到强类型控制总线 ctx。
        """
        async with self.db_session as session:
            repo = OpsAgentConfRepository(session)
            raw_tuples = await repo.get_bound_instances_by_agent(agent_id)

            cleaned_instances = []
            for conf_entity, instance_entity in raw_tuples:
                instance_dto = {
                    # --- 资产物理特征 ---
                    "instance_id": instance_entity.instance_id,
                    "instance_name": instance_entity.instance_name,
                    "db_type": instance_entity.db_type,
                    "version_code": instance_entity.version_code,
                    "environment": instance_entity.environment,
                    "security_level": instance_entity.security_level,
                    "host": instance_entity.host,
                    "port": instance_entity.port,
                    "service_name": instance_entity.service_name,
                    "database_name": instance_entity.database_name,
                    "db_role": instance_entity.db_role,

                    # --- 对应智能体的控制策略闸门 ---
                    "relation_config_id": conf_entity.id,
                    "is_mutation_allowed": conf_entity.is_mutation_allowed,
                    "require_approval": conf_entity.require_approval,
                    "max_daily_execution": conf_entity.max_daily_execution
                }
                cleaned_instances.append(instance_dto)

            return cleaned_instances

    async def get_instance_detail_by_id(self, instance_id: str) -> dict[str, Any] | None:
        """
        【流式网关核心函数 - 精准单查版】
        依据前端选定的唯一 instance_id 精准锁定单台物理库资产,
        并将 JOIN 实体扁平化清洗为单体 Dict 字典, 供编排器直接装载注入到强类型控制总线 ctx。
        """
        async with self.db_session as session:
            repo = OpsAgentConfRepository(session)
            raw_tuple = await repo.get_bound_instance_by_id(instance_id)

            if not raw_tuple:
                logger.warning(f"[OpsAgentConfService] 资产锁定失败, 系统中未注册该实例 | InstanceID: {instance_id}")
                return None

            conf_entity, instance_entity = raw_tuple

            instance_dto = {
                "instance_id": instance_entity.instance_id,
                "instance_name": instance_entity.instance_name,
                "db_type": instance_entity.db_type,
                "version_code": instance_entity.version_code,
                "environment": instance_entity.environment,
                "security_level": instance_entity.security_level,
                "host": instance_entity.host,
                "port": instance_entity.port,
                "service_name": instance_entity.service_name,
                "database_name": instance_entity.database_name,
                "db_role": instance_entity.db_role,
                "monitor_type": instance_entity.monitor_type,
                "prometheus_instance_label": instance_entity.prometheus_instance_label,

                "relation_config_id": conf_entity.id,
                "is_mutation_allowed": conf_entity.is_mutation_allowed,
                "require_approval": conf_entity.require_approval,
                "max_daily_execution": conf_entity.max_daily_execution
            }

            return instance_dto
