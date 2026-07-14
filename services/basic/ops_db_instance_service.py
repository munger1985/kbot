# services/basic/ops_db_instance_service.py

from typing import Any
from loguru import logger
from core.exceptions import NotFoundError, InternalServerError
from dao.repositories.ops_db_instance_repo import OpsDbInstanceRepository
from dao.entities.ops_db_instance import OpsDbInstanceEntity
from core.security import CryptoToolkit
from core.database.oracle import get_session


class OpsDBInstanceService:
    """
    配置管理数据库服务 (CMDB Service)
    专职打理物理实例资产盘点、凭证托管以及自愈风控控制。
    """

    def __init__(self):
        self.crypto = CryptoToolkit()

    @property
    def session(self):
        return get_session()

    async def get_instance_by_id(self, instance_id: str) -> dict[str, Any]:
        """
        供物理引擎驱动(Driver)直接调用, 提供实时解密后的高权凭证快照。
        """
        logger.debug(f"[CMDBService] 正在为执行微服务加载实例凭证: {instance_id}")

        try:
            async with self.session as session:
                repo = OpsDbInstanceRepository(session)
                entity = await repo.get_by_id(instance_id)

            # 运维生命线: 如果实例处于离线状态, 拒绝暴露凭证
            if entity.status == "offline":
                logger.critical(f"[CMDBService] 警告: 实例 {instance_id} 已处于下线状态, 拒绝提供物理连接。")
                raise NotFoundError(f"实例 [{instance_id}] 已在CMDB中被强制下线封禁")

            # 核心安全防护: 将强加密的物理密码在内存中实时解密
            decrypted_password = self._decrypt_password(entity.encrypted_password)

            return {
                "instance_id": entity.instance_id,
                "instance_name": entity.instance_name,
                "db_type": entity.db_type,
                "environment": entity.environment,
                "db_role": entity.db_role,
                "version_code": entity.version_code,
                "monitor_type": entity.monitor_type,
                "prometheus_instance_label": entity.prometheus_instance_label,
                "zabbix_host_name": entity.zabbix_host_name,
                "connection_config": {
                    "host": entity.host,
                    "port": entity.port,
                    "user": entity.ops_user,
                    "password": decrypted_password,
                    "database": entity.database_name,
                    "service_name": entity.service_name,
                    "dsn": entity.dsn,
                    "charset": entity.charset
                }
            }

        except NotFoundError as e:
            raise e
        except Exception as e:
            logger.exception(f"[CMDBService] 物理资产检索链路崩溃 | 实例: {instance_id}")
            raise InternalServerError(f"元资产中心寻址崩溃: {str(e)}")

    async def get_cluster_topology(self, cluster_id: str) -> list[dict[str, Any]]:
        """获取高可用集群的主备拓扑网络"""
        logger.info(f"[CMDBService] 正在加载高可用拓扑网格, 集群: {cluster_id}")

        async with self.session as session:
            repo = OpsDbInstanceRepository(session)
            nodes = await repo.get_by_cluster(cluster_id)

        topo_tree = []
        for node in nodes:
            topo_tree.append({
                "instance_id": node.instance_id,
                "instance_name": node.instance_name,
                "role": node.db_role,
                "endpoint": f"{node.host}:{node.port}",
                "status": node.status
            })
        return topo_tree

    async def get_all_instances(self) -> list[dict[str, Any]]:
        """获取所有活跃的运维实例列表（不含密码），供前端选择器使用"""
        async with self.session as session:
            repo = OpsDbInstanceRepository(session)
            instances = await repo.get_all_active()
            return [{
                "instance_id": inst.instance_id,
                "instance_name": inst.instance_name,
                "db_type": inst.db_type,
                "version_code": inst.version_code,
                "environment": inst.environment,
                "security_level": inst.security_level,
                "host": inst.host,
                "port": inst.port,
                "service_name": inst.service_name,
                "database_name": inst.database_name,
                "db_role": inst.db_role,
                "monitor_type": inst.monitor_type,
                "status": inst.status,
            } for inst in instances]

    async def get_instance_for_ui(self, instance_id: str) -> dict[str, Any] | None:
        """获取单个实例详情（不含密码），供前端管理页面使用"""
        async with self.session as session:
            repo = OpsDbInstanceRepository(session)
            inst = await repo.get_by_id(instance_id)
            if not inst:
                return None
            return {
                "instance_id": inst.instance_id,
                "instance_name": inst.instance_name,
                "environment": inst.environment,
                "security_level": inst.security_level,
                "db_type": inst.db_type,
                "version_code": inst.version_code,
                "charset": inst.charset,
                "host": inst.host,
                "port": inst.port,
                "service_name": inst.service_name,
                "database_name": inst.database_name,
                "dsn": inst.dsn,
                "db_role": inst.db_role,
                "monitor_type": inst.monitor_type,
                "prometheus_instance_label": inst.prometheus_instance_label,
                "zabbix_host_name": inst.zabbix_host_name,
                "cluster_id": inst.cluster_id,
                "ops_user": inst.ops_user,
                "secret_vault_key": inst.secret_vault_key,
                "status": inst.status,
                "created_at": inst.created_at.isoformat() if inst.created_at else None,
                "updated_at": inst.updated_at.isoformat() if inst.updated_at else None,
            }

    async def update_instance(self, instance_id: str, data: dict[str, Any]) -> None:
        """更新运维实例信息, 若含密码则重新加密"""
        update_data = {k: v for k, v in data.items() if v is not None}
        if "password" in update_data:
            update_data["encrypted_password"] = self._encrypt_password(update_data.pop("password"))
        if not update_data:
            return
        async with self.session as session:
            repo = OpsDbInstanceRepository(session)
            await repo.update_instance(instance_id, **update_data)
            logger.success(f"[CMDB] 实例 {instance_id} 信息已更新")

    async def delete_instance(self, instance_id: str) -> None:
        """删除运维实例"""
        async with self.session as session:
            repo = OpsDbInstanceRepository(session)
            await repo.delete_instance(instance_id)
            logger.info(f"[CMDB] 实例 {instance_id} 已删除")

    async def register_new_instance(self, asset_data: dict[str, Any]) -> str:
        """新资产录入（由 DBA 控制面或自动化资产发现脚本调用）"""
        encrypted_pwd = self._encrypt_password(asset_data["password"])

        entity = OpsDbInstanceEntity(
            instance_id=asset_data.get("instance_id"),
            instance_name=asset_data["instance_name"],
            environment=asset_data.get("environment", "dev"),
            security_level=asset_data.get("security_level", 3),
            db_type=asset_data["db_type"],
            version_code=asset_data["version_code"],
            charset=asset_data.get("charset", "utf8mb4"),
            host=asset_data["host"],
            port=asset_data["port"],
            service_name=asset_data.get("service_name"),
            database_name=asset_data.get("database_name"),
            dsn=asset_data.get("dsn"),
            db_role=asset_data.get("db_role", "primary"),
            cluster_id=asset_data.get("cluster_id"),
            ops_user=asset_data["ops_user"],
            encrypted_password=encrypted_pwd,
            secret_vault_key=asset_data.get("secret_vault_key"),
            status=asset_data.get("status", "active")
        )
        async with self.session as session:
            repo = OpsDbInstanceRepository(session)
            return await repo.create(entity)

    def _decrypt_password(self, encrypted_str: str) -> str:
        """调用 AES-256-GCM 安全套件解密"""
        return self.crypto.decrypt(encrypted_str)

    def _encrypt_password(self, raw_str: str) -> str:
        """调用 AES-256-GCM 安全套件加密"""
        return self.crypto.encrypt(raw_str)
