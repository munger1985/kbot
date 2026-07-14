# dao/entities/ops_db_instance.py

from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime, CheckConstraint, text
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity


class OpsDbInstanceEntity(BaseEntity):
    """智能运维线 - 物理与云端数据库实例资产配置 (CMDB 核心表映射)"""

    __tablename__ = "kbot_ops_db_instance"

    instance_id: Mapped[str] = mapped_column(String(36), primary_key=True, comment="运维全域唯一实例ID (UUID v7 格式)")
    instance_name: Mapped[str] = mapped_column(String(128), comment="实例直观可读名称")
    environment: Mapped[str] = mapped_column(String(16), default="dev", comment="环境分级隔离策略: prod, stg, dev")
    security_level: Mapped[int] = mapped_column(Integer, default=3, comment="安全控制等级,数值越高触发变更自愈时的审批流越严格")
    db_type: Mapped[str] = mapped_column(String(32), comment="数据库引擎内核类型: oracle, postgresql, mysql")
    version_code: Mapped[int] = mapped_column(Integer, default=0, comment="精准内核版本数字代码,例如: 26000000")
    charset: Mapped[str | None] = mapped_column(String(32), default="utf8mb4", comment="数据库实例配置的默认字符集")
    host: Mapped[str] = mapped_column(String(256), comment="物理机/虚拟机 IP 地址或高可用对外域名")
    port: Mapped[int] = mapped_column(Integer, comment="数据库内核监听服务的物理端口号")
    service_name: Mapped[str | None] = mapped_column(String(128), comment="Oracle专属: 存放 Service Name 或 SID")
    database_name: Mapped[str | None] = mapped_column(String(128), comment="PostgreSQL/MySQL专属: 默认绑定的物理数据库名")
    dsn: Mapped[str | None] = mapped_column(String(512), comment="自定义高可用高级连接串描述符")
    db_role: Mapped[str] = mapped_column(String(32), default="primary", comment="物理角色: primary(主), standby(备), cluster_node")
    monitor_type: Mapped[str] = mapped_column(String(32), default="prometheus", comment="监控数据源类型: prometheus, zabbix, none")
    prometheus_instance_label: Mapped[str | None] = mapped_column(String(256), comment="对应 Prometheus 采集时的 instance 标签值")
    zabbix_host_name: Mapped[str | None] = mapped_column(String(256), comment="Zabbix 监控主机名称")
    cluster_id: Mapped[str | None] = mapped_column(String(36), comment="高可用集群级关联ID")
    ops_user: Mapped[str] = mapped_column(String(64), comment="大模型运维通道专属的高权/监控账号")
    encrypted_password: Mapped[str] = mapped_column(String(512), comment="经过强加密后的运维账号物理密码密文")
    secret_vault_key: Mapped[str | None] = mapped_column(String(256), comment="第三方密囊系统(如 HashiCorp Vault)的凭证索引 Key")
    status: Mapped[str] = mapped_column(String(16), default="active", comment="生命周期状态: active, maintenance, offline")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc), comment="资产登记入库时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment="最近一次变更同步时间")

    # Oracle CHECK 约束
    __table_args__ = (
        CheckConstraint("environment IN ('prod', 'stg', 'dev')", name="ck_ops_db_env"),
        CheckConstraint("db_role IN ('primary', 'standby', 'cluster_node')", name="ck_ops_db_role"),
        CheckConstraint("status IN ('active', 'maintenance', 'offline')", name="ck_ops_db_status"),
        {"comment": "智能运维线 - 物理与云端数据库实例资产配置表 (CMDB)"}
    )

    def __repr__(self) -> str:
        return f"<OpsDbInstance(id={self.instance_id}, name='{self.instance_name}', role='{self.db_role}', env='{self.environment}')>"
