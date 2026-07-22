# dao/repositories/ops_db_instance_repo.py

from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Sequence
from dao.entities.ops_db_instance import OpsDbInstanceEntity
from platform_core.exceptions import APIException, DatabaseException, DataNotFoundException, DataConflictException
from .base_repo import BaseRepository


class OpsDbInstanceRepository(BaseRepository[OpsDbInstanceEntity]):
    """
    运维实例资产仓库
    使用构造注入的 self.session 管理会话, 完美对接双轨制自愈系统
    """

    async def create(self, instance: OpsDbInstanceEntity) -> str:
        """登记新的物理数据库实例"""
        try:
            # 1. 检查物理寻址唯一性冲突 (防止同一 IP + Port 被重复登记)
            result = await self.session.execute(
                select(OpsDbInstanceEntity)
                .where(
                    and_(
                        OpsDbInstanceEntity.host == instance.host,
                        OpsDbInstanceEntity.port == instance.port
                    )
                )
            )
            if result.scalar_one_or_none():
                raise DataConflictException(
                    f"物理实例地址冲突: 目标库 {instance.host}:{instance.port} 已经登记在册"
                )

            # 2. 如果指定了特定的唯一 ID, 检查 ID 冲突
            if instance.instance_id:
                id_check = await self.session.execute(
                    select(OpsDbInstanceEntity).where(OpsDbInstanceEntity.instance_id == instance.instance_id)
                )
                if id_check.scalar_one_or_none():
                    raise DataConflictException(f"实例 ID 冲突: {instance.instance_id} 已存在")

            # 3. 添加到上下文中并刷新
            self.session.add(instance)
            await self.session.flush()
            return instance.instance_id

        except DataConflictException as e:
            raise e
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException("登记物理资产实例失败", original_error=e)

    async def get_by_id(self, instance_id: str) -> OpsDbInstanceEntity:
        """根据实例 ID 精确查找资产元数据"""
        try:
            result = await self.session.execute(
                select(OpsDbInstanceEntity)
                .where(OpsDbInstanceEntity.instance_id == instance_id)
            )
            entity = result.scalar_one_or_none()
            if not entity:
                raise DataNotFoundException(f"CMDB资产库中未找到实例 ID 为 [{instance_id}] 的节点")
            return entity
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"根据ID检索实例资产失败: {instance_id}", original_error=e)

    async def get_by_cluster(self, cluster_id: str) -> Sequence[OpsDbInstanceEntity]:
        """根据集群 ID 批量捞取完整的物理拓扑节点（用于自愈程序识别主备关系）"""
        try:
            result = await self.session.execute(
                select(OpsDbInstanceEntity)
                .where(OpsDbInstanceEntity.cluster_id == cluster_id)
                .order_by(OpsDbInstanceEntity.db_role.asc())
            )
            return result.scalars().all()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"检索集群高可用拓扑失败: {cluster_id}", original_error=e)

    async def get_all_active(self) -> Sequence[OpsDbInstanceEntity]:
        """获取所有活跃的运维实例（不含已下线的）"""
        try:
            result = await self.session.execute(
                select(OpsDbInstanceEntity)
                .where(OpsDbInstanceEntity.status != "offline")
                .order_by(OpsDbInstanceEntity.environment.asc(),
                          OpsDbInstanceEntity.db_type.asc())
            )
            return result.scalars().all()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException("获取所有活跃运维实例失败", original_error=e)

    async def update_instance(self, instance_id: str, **kwargs) -> None:
        """更新运维实例资产信息"""
        try:
            result = await self.session.execute(
                update(OpsDbInstanceEntity)
                .where(OpsDbInstanceEntity.instance_id == instance_id)
                .values(**kwargs)
            )
            # Oracle 不支持 RETURNING in UPDATE via SQLAlchemy the same way,
            # but execute() with update() still works — just no scalar returned
            await self.session.flush()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"更新运维实例失败: {instance_id}", original_error=e)

    async def delete_instance(self, instance_id: str) -> None:
        """下线并物理删除资产"""
        try:
            entity = await self.get_by_id(instance_id)
            await self.session.delete(entity)
            await self.session.flush()
        except Exception as e:
            if isinstance(e, (APIException, DataNotFoundException)):
                raise
            raise DatabaseException(f"下线物理实例资产失败: {instance_id}", original_error=e)

    async def find_by_prometheus_label(self, label: str) -> OpsDbInstanceEntity | None:
        """通过 Prometheus instance label 查找实例 (用于告警 webhook 自动匹配)。"""
        stmt = (
            select(OpsDbInstanceEntity)
            .where(OpsDbInstanceEntity.prometheus_instance_label == label)
            .where(OpsDbInstanceEntity.status == "active")
            .fetch(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_by_zabbix_host(self, host_name: str) -> OpsDbInstanceEntity | None:
        """通过 Zabbix host name 查找实例 (用于告警 webhook 自动匹配)。"""
        stmt = (
            select(OpsDbInstanceEntity)
            .where(OpsDbInstanceEntity.zabbix_host_name == host_name)
            .where(OpsDbInstanceEntity.status == "active")
            .fetch(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
