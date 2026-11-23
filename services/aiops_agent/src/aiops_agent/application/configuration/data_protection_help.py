"""数据保护权限说明和经验目标，不作现场预检结论。"""

from platform_core.contracts.aiops.data_protection import (
    ProtectionAdvice, ProtectionCheck, ProtectionPlanSettings,
)


TOOLS = {"ORACLE": "RMAN", "POSTGRESQL": "PGBACKREST", "MYSQL": "XTRABACKUP"}
BACKUP_DATABASE_HELP = {
    "RMAN": "首版定时备份需 DBA 授权本机 OS 用户以 SYSBACKUP 登录，不能只填远程数据库账号；整库在线备份需 ARCHIVELOG 和 SPFILE。",
    "PGBACKREST": "专用账号须经本机 socket 连接 postgres，读取实例设置，并可执行 pg_control_system、pg_backup_start/stop、pg_switch_wal 和 pg_create_restore_point。要求开启 WAL 归档；pg_monitor 不等于备份权限。",
    "XTRABACKUP": "本机全备需专用账号直接授予 PROCESS、RELOAD、LOCK TABLES 和 REPLICATION CLIENT；MySQL 8 另需 BACKUP_ADMIN，读取 performance_schema 下的 log_status、replication_group_members；8.4 另需读取 keyring_component_status。REPLICATION CLIENT 不等于 Binlog 流读取权。",
}
BACKUP_HOST_HELP = {
    "RMAN": "AIOps API 和 Worker 需以该 OS 用户运行，并能执行登记 Oracle Home 内的 SQL*Plus 和 RMAN。",
    "PGBACKREST": "AIOps API 和 Worker 需能访问登记的本机 socket、读取源数据文件，运行 pgBackRest 2.55.0 或更新版本；使用既有本机 stanza 的 repo1。",
    "XTRABACKUP": "AIOps API 和 Worker 需能访问登记的本机 socket、读取 datadir，运行与 MySQL 代际匹配的 XtraBackup 与 innochecksum。",
}
RESTORE_DATABASE_HELP = {
    "RMAN": "仅接受独立 NOMOUNT 辅助实例密码文件内的 SYSBACKUP，能在本机执行辅助恢复及读取状态；不使用源库身份。",
    "PGBACKREST": "恢复后使用备份内已有的验证角色，通过本机 socket 和 SCRAM 认证连接验证库，读取恢复状态、执行 pg_control_checkpoint() 并读取已选验证表。",
    "XTRABACKUP": "恢复后使用备份内已有的验证账号，通过本机 socket 登录；需要查看实例状态、验证库目录和已选验证表，不使用生产复制权限。",
}
EXPERIENCE = {
    "CRITICAL": (900, 60), "HIGH": (3600, 300), "MEDIUM": (14400, 3600),
    "LOWER": (28800, 14400), "LOW": (86400, 86400),
}
TOPOLOGY_NOTES = {
    "UNKNOWN": "主备关系未确认，不能判断保护能力。",
    "SINGLE": "单库需核对备份间隔与连续日志；只有周期备份不能承诺分钟级 RPO。",
    "ASYNC_REPLICA": "异步主备需核对传输和回放延迟，不能认为零丢失。",
    "SYNC_REPLICA": "同步切换不等于空库重建，不能替代备份恢复演练。",
}


def protection_advice(settings: ProtectionPlanSettings) -> ProtectionAdvice:
    rto, rpo = EXPERIENCE.get(settings.importance, (None, None))
    return ProtectionAdvice(
        rto_seconds=rto, rpo_seconds=rpo,
        explanation="经验建议，不是服务承诺或实测。" + TOPOLOGY_NOTES[settings.topology],
    )


def configuration_checks(
    settings: ProtectionPlanSettings, *, has_backup_identity: bool, has_restore_identity: bool,
) -> tuple[ProtectionCheck, ...]:
    """未绑定现场脚本/预检时不得把填写完成显示成可执行。"""
    return (
        ProtectionCheck(area="DATABASE", operation="BACKUP",
                        status="UNVERIFIED" if has_backup_identity else "MISSING",
                        guidance=BACKUP_DATABASE_HELP[settings.tool]),
        ProtectionCheck(area="HOST", operation="BACKUP",
                        status="UNVERIFIED",
                        guidance=BACKUP_HOST_HELP[settings.tool]),
        ProtectionCheck(area="STORAGE", operation="BACKUP",
                        status="UNVERIFIED" if (settings.storage_key if settings.tool == "PGBACKREST" else settings.backup_directory) else "MISSING",
                        guidance="核对备份仓库、批准范围内的写入与校验权限、保留和空间。"),
        ProtectionCheck(area="NETWORK_TOOL", operation="BACKUP", status="UNVERIFIED",
                        guidance="确认现有执行通道与数据库、工具版本及仓库可达性。"),
        ProtectionCheck(area="DATABASE", operation="DRILL",
                        status="UNVERIFIED" if has_restore_identity else "MISSING",
                        guidance=RESTORE_DATABASE_HELP[settings.tool]),
        ProtectionCheck(area="HOST", operation="DRILL",
                        status="UNVERIFIED" if settings.drill_destination_key and settings.restore_template_key else "MISSING",
                        guidance="独立演练机的现有 Worker 须以已登记的非 root 专用系统用户运行，能执行本机工具并独占空目标及工作目录；只有数据库账号不够。"),
        ProtectionCheck(area="STORAGE", operation="DRILL",
                        status="UNVERIFIED" if settings.storage_key or settings.backup_directory else "MISSING",
                        guidance="批准的备份仓库须只读挂载并与演练写入目录分开；备份和工作副本所需空间由 DBA 准备。"),
        ProtectionCheck(area="NETWORK_TOOL", operation="DRILL", status="UNVERIFIED",
                        guidance="由 DBA 确认演练主机不能访问生产主库或备库，并安装匹配登记版本的原生工具；Worker 不探测生产网络。"),
        ProtectionCheck(area="VERIFY", operation="DRILL", status="UNVERIFIED",
                        guidance="演练只验证数据库。PG/MySQL 在恢复后才能验证账号和对象权限；预检通过不等于已验证登录或应用可用。"),
    )
