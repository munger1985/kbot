"""备份与演练执行入口的最小业务边界，不实现远程通道。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from re import fullmatch
from typing import Protocol

from platform_core.contracts.aiops.data_protection import ProtectionBackupKind, ProtectionTool


@dataclass(frozen=True)
class BackupArtifact:
    """引用不可变备份集及当时已可用的归档；不是可随时间变化的目录。"""

    storage_key: str
    reference: str
    available_at: datetime
    checksum_verified: bool
    kind: ProtectionBackupKind = "FULL"
    dependencies: tuple[str, ...] = ()
    # 原生备份的文件清单摘要与检查点；不含备份内容或凭据。
    fingerprint: str | None = None
    baseline_fingerprint: str | None = None
    minimum_scn: int | None = None
    end_lsn: int | None = None
    start_lsn: int | None = None


FILE_BACKUP_REFERENCE = r"aiops-[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}"
PG_FULL_LABEL = r"[0-9]{8}-[0-9]{6}F"
PG_INCREMENTAL_LABEL = PG_FULL_LABEL + r"_[0-9]{8}-[0-9]{6}I"


def valid_backup_chain(tool: ProtectionTool, artifact: BackupArtifact) -> bool:
    refs = (*artifact.dependencies, artifact.reference)
    if not all(isinstance(ref, str) for ref in refs) or len(refs) != len(set(refs)):
        return False
    if tool == "PGBACKREST":
        return bool(fullmatch(PG_FULL_LABEL, refs[0]) and (
            (artifact.kind == "FULL" and len(refs) == 1)
            or (artifact.kind == "INCREMENTAL" and len(refs) >= 2 and all(
                fullmatch(PG_INCREMENTAL_LABEL, ref) and ref.startswith(refs[0] + "_") for ref in refs[1:]
            ))
        ))
    if not all(fullmatch(FILE_BACKUP_REFERENCE, ref) for ref in refs):
        return False
    if artifact.kind == "FULL":
        return len(refs) == 1
    if artifact.kind != "INCREMENTAL" or len(refs) != 2 or not _sha256(artifact.fingerprint):
        return False
    if tool == "RMAN":
        return (_sha256(artifact.baseline_fingerprint) and type(artifact.minimum_scn) is int
                and artifact.minimum_scn > 0)
    return (tool == "XTRABACKUP" and type(artifact.start_lsn) is int and type(artifact.end_lsn) is int
            and 0 < artifact.start_lsn < artifact.end_lsn)


def _sha256(value: str | None) -> bool:
    return isinstance(value, str) and fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class DrillDestination:
    """独立登记的演练目标；实例别名必须归一为相同的实际独占资源身份。"""

    key: str
    instance_identity: str
    data_path: str


@dataclass(frozen=True)
class DrillPreflight:
    """受控执行位置的只读观测；不能用页面表单构造。"""

    destination: DrillDestination
    dedicated_drill_environment: bool
    no_production_overlap: bool
    isolated_from_production: bool
    empty_and_idle: bool
    permissions_verified: bool
    tool_compatible: bool
    capacity_verified: bool
    backup_readable: bool


@dataclass(frozen=True)
class DrillRequest:
    tool: ProtectionTool
    destination: DrillDestination
    storage_key: str
    backup: BackupArtifact
    simulated_failure_at: datetime


@dataclass(frozen=True)
class DrillVerification:
    database_verified: bool
    verified_at: datetime | None
    recovery_coordinate: str | None
    recovered_at: datetime | None
    recovered_at_latest: datetime | None = None
    recovery_coordinate_kind: str | None = None
    retrieval_seconds: int | None = None
    restore_seconds: int | None = None
    verification_seconds: int | None = None

    def evidence(self, tool: ProtectionTool, *, not_before: datetime, not_after: datetime) -> dict:
        kinds = {"RMAN": {"SCN"}, "PGBACKREST": {"TIMELINE_LSN"}, "XTRABACKUP": {"GTID", "BINLOG"}}
        valid_coordinate = (self.recovery_coordinate_kind in kinds[tool] and self.recovery_coordinate
                            and fullmatch(r"[a-zA-Z0-9:._/, -]{1,256}", self.recovery_coordinate))
        verified = bool(
            self.database_verified is True and self.verified_at and _aware(self.verified_at)
            and not_before <= self.verified_at <= not_after and valid_coordinate
        )
        return {
            "database_verified": verified,
            "verified_at": _iso(self.verified_at) if verified else None,
            "recovery_coordinate": self.recovery_coordinate if verified else None,
            "recovery_coordinate_kind": self.recovery_coordinate_kind if verified else None,
            "recovered_at_earliest": _iso(self.recovered_at) if verified else None,
            "recovered_at_latest": _iso(self.recovered_at_latest or self.recovered_at) if verified else None,
            "retrieval_seconds": _duration(self.retrieval_seconds),
            "restore_seconds": _duration(self.restore_seconds),
            "verification_seconds": _duration(self.verification_seconds),
            "message": "数据库恢复与恢复坐标已验证" if verified else "恢复结果或恢复坐标未验证",
        }


class DrillRunner(Protocol):
    async def probe_drill(self, request: DrillRequest) -> DrillPreflight:
        """只读取演练环境与备份存储，不连接生产库。"""

    async def restore_and_verify(self, request: DrillRequest) -> DrillVerification:
        """仅用批准模板及演练凭据；写入口须绑定已核验目的地和 T0 前的备份集。"""


class DrillRejected(RuntimeError):
    """门禁拒绝，恢复写步骤尚未被调用。"""

    def __init__(self, message: str = "", *, code: str = "UNVERIFIED"):
        super().__init__(message)
        self.code = code


def _proper_destination(destination: DrillDestination) -> bool:
    path = PurePosixPath(destination.data_path)
    return bool(destination.key and destination.instance_identity and path.is_absolute()
                and path.name and ".." not in path.parts and str(path) == destination.data_path
                and not destination.data_path.startswith("//"))


def _aware(value: datetime) -> bool:
    return value.utcoffset() is not None


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None and _aware(value) else None


def _duration(value: int | None) -> int | None:
    return value if type(value) is int and value >= 0 else None


async def execute_isolated_drill(
    request: DrillRequest, runner: DrillRunner, *, before_write: Callable[[], Awaitable[None]] | None = None,
) -> DrillVerification:
    """调用者须先获得目的地互斥；写入前再用执行端观测核对。"""
    if not _proper_destination(request.destination):
        raise DrillRejected("未登记明确的空演练实例及数据目录", code="DESTINATION_CHANGED")
    if (not _aware(request.simulated_failure_at) or not _aware(request.backup.available_at)
        or request.backup.available_at.astimezone(UTC) > request.simulated_failure_at.astimezone(UTC)
        or not request.backup.reference or request.backup.checksum_verified is not True
        or not request.storage_key or request.backup.storage_key != request.storage_key):
        raise DrillRejected("没有可用且符合故障时点的备份证据", code="BACKUP_UNAVAILABLE")
    facts = await runner.probe_drill(request)
    if facts.destination != request.destination:
        raise DrillRejected("演练端实例身份或规范路径发生变化", code="DESTINATION_CHANGED")
    if any(item is not True for item in (
        facts.dedicated_drill_environment, facts.no_production_overlap, facts.isolated_from_production,
    )):
        raise DrillRejected("未证实是与生产主库和备库隔离的演练环境", code="ISOLATION_UNVERIFIED")
    if facts.empty_and_idle is not True:
        raise DrillRejected("目标非空、已占用或空状态未验证", code="NOT_EMPTY")
    if any(item is not True for item in (
        facts.permissions_verified, facts.tool_compatible, facts.capacity_verified, facts.backup_readable,
    )):
        raise DrillRejected("权限、兼容性、容量或备份读取未验证")
    if before_write is not None:
        await before_write()
    return await runner.restore_and_verify(request)
