# dao/repositories/ops_agent_conf_repo.py

from sqlalchemy import select, update, delete, and_
from typing import Sequence

from dao.entities.ops_agent_conf import OpsAgentConfEntity
from dao.entities.ops_db_instance import OpsDbInstanceEntity
from core.exceptions import APIException, DatabaseException, DataNotFoundException
from .base_repo import BaseRepository


class OpsAgentConfRepository(BaseRepository[OpsAgentConfEntity]):
    """
    智能运维智能体资产配置关系仓库
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str
    """

    async def create(self, ops_agent_conf: OpsAgentConfEntity) -> None:
        """创建智能体与运维资产绑定配置"""
        try:
            self.session.add(ops_agent_conf)
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException("创建运维智能体资产配置绑定失败", original_error=e)

    async def update(self, config_id: int, **kwargs) -> None:
        """根据配置 ID 更新运维自愈与限流规则配置"""
        try:
            await self.session.execute(
                update(OpsAgentConfEntity)
                .where(OpsAgentConfEntity.id == config_id)
                .values(**kwargs)
            )
            await self.session.flush()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException("更新运维智能体资产配置失败", original_error=e)

    async def delete(self, config_id: int) -> None:
        """根据配置 ID 彻底删除/解除绑定关系"""
        try:
            await self.session.execute(
                delete(OpsAgentConfEntity)
                .where(OpsAgentConfEntity.id == config_id)
            )
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException("解绑运维资产配置失败", original_error=e)

    async def get(self, config_id: int) -> OpsAgentConfEntity:
        """根据配置 ID 获取特定的运维绑定与策略快照"""
        try:
            result = await self.session.execute(
                select(OpsAgentConfEntity).where(OpsAgentConfEntity.id == config_id)
            )
            config = result.scalar_one_or_none()
            if not config:
                raise DataNotFoundException(f"运维资产绑定关系配置 {config_id} 不存在")
            return config
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException("获取运维智能体资产配置失败", original_error=e)

    async def get_by_agent_and_instance(self, agent_id: int, instance_id: str) -> OpsAgentConfEntity | None:
        """
        根据 agent_id (int) 和 instance_id (str) 精准获取门禁控制策略
        (用于在具体执行 KillSession 或变更技能前, 秒级判定该 Agent 是否具备当前实例的 mutation 权限)
        """
        try:
            result = await self.session.execute(
                select(OpsAgentConfEntity).where(
                    and_(
                        OpsAgentConfEntity.agent_id == agent_id,
                        OpsAgentConfEntity.instance_id == instance_id
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"联合查询智能体实例门禁异常 | Agent: {agent_id}, Instance: {instance_id}", original_error=e)

    async def get_bound_instances_by_agent(self, agent_id: int) -> Sequence[tuple[OpsAgentConfEntity, OpsDbInstanceEntity]]:
        """
        核心联动: 一键拉取当前运维脑托管的所有硬核物理实例资产 (带有强控制开关)
        供编排器（Orchestrator）在初始化强类型总线 ctx 时做资产拓扑回填与锚定使用。
        agent_id: int
        """
        try:
            stmt = (
                select(OpsAgentConfEntity, OpsDbInstanceEntity)
                .join(OpsDbInstanceEntity, OpsAgentConfEntity.instance_id == OpsDbInstanceEntity.instance_id)
                .where(OpsAgentConfEntity.agent_id == agent_id)
                .where(OpsDbInstanceEntity.status == "active")
            )
            result = await self.session.execute(stmt)
            return result.tuples().all()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"拉取智能体 [{agent_id}] 绑定的物理实例集群拓扑失败", original_error=e)

    async def unbind_instance(self, agent_id: int, instance_id: str) -> bool:
        """根据 agent_id (int) 与 instance_id (str) 显式快捷解绑关系"""
        try:
            await self.session.execute(
                delete(OpsAgentConfEntity).where(
                    and_(
                        OpsAgentConfEntity.agent_id == agent_id,
                        OpsAgentConfEntity.instance_id == instance_id
                    )
                )
            )
            return True
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"解除智能体 [{agent_id}] 与实例 [{instance_id}] 的关系失败", original_error=e)

    async def get_bound_instance_by_id(self, instance_id: str) -> tuple[OpsAgentConfEntity, OpsDbInstanceEntity] | None:
        """
        根据实例 ID (str) 精准 JOIN 查询单条配置与实例实体
        """
        try:
            stmt = (
                select(OpsAgentConfEntity, OpsDbInstanceEntity)
                .join(OpsDbInstanceEntity, OpsAgentConfEntity.instance_id == OpsDbInstanceEntity.instance_id)
                .where(OpsDbInstanceEntity.instance_id == instance_id)
            )
            result = await self.session.execute(stmt)
            return result.tuples().first()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"根据实例 ID [{instance_id}] 查询绑定关系失败", original_error=e)
