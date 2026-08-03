"""验收 AIOps SQLAlchemy Entity 与当前 Oracle Catalog。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from sqlalchemy import DateTime, Integer, Numeric, String, Text, text

# 支持从仓库根目录直接执行本脚本。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiops_agent.entities import (
    CredentialEntity,
    ApprovalTokenEntity,
    ChangeProposalEntity,
    ExecutionEntity,
    HitlEntity,
    InboxEntity,
    InspectionFireEntity,
    InspectionPlanEntity,
    InspectionTargetEntity,
    MonitorSourceEntity,
    OpsAlertEntity,
    OpsArtifactEntity,
    OpsEventEntity,
    OpsRunEntity,
    OpsRunEventEntity,
    OpsTaskEntity,
    OutboxEntity,
    PolicyEntity,
    ReportEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetMonitorEntity,
)
from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.persistence.orm import (
    OracleNativeJSON,
    UniversalTimestamp,
    UniversalVector,
    UUIDv7Type,
)
from tests.support.oracle_preflight import require_oracle_listener


AIOPS_ENTITY_CLASSES = (
    CredentialEntity,
    TargetEntity,
    PolicyEntity,
    TargetBindingEntity,
    MonitorSourceEntity,
    TargetMonitorEntity,
    OpsEventEntity,
    OpsAlertEntity,
    OpsRunEntity,
    OpsTaskEntity,
    OpsArtifactEntity,
    OpsRunEventEntity,
    ChangeProposalEntity,
    HitlEntity,
    ApprovalTokenEntity,
    ExecutionEntity,
    InspectionPlanEntity,
    InspectionTargetEntity,
    InspectionFireEntity,
    ReportEntity,
    InboxEntity,
    OutboxEntity,
)
HITL_STATUSES = {
    "PENDING",
    "ANSWERED",
    "APPROVED",
    "REJECTED",
    "SKIPPED",
    "EXPIRED",
    "CANCELLED",
}


@dataclass(frozen=True)
class ColumnContract:
    """与数据库方言无关的列物理契约。"""

    family: str
    length: int | None
    precision: int | None
    scale: int | None
    timezone: bool | None
    nullable: bool


def entity_column_contract(column) -> ColumnContract:
    column_type = column.type
    if isinstance(column_type, UUIDv7Type):
        return ColumnContract("RAW", 16, None, None, None, column.nullable)
    if isinstance(column_type, UniversalTimestamp):
        return ColumnContract(
            "TIMESTAMP",
            None,
            None,
            None,
            column_type.timezone,
            column.nullable,
        )
    if isinstance(column_type, UniversalVector):
        return ColumnContract(
            "VECTOR", None, None, None, None, column.nullable
        )
    if isinstance(column_type, OracleNativeJSON):
        return ColumnContract(
            "JSON", None, None, None, None, column.nullable
        )
    if isinstance(column_type, Text):
        return ColumnContract("CLOB", None, None, None, None, column.nullable)
    if isinstance(column_type, DateTime):
        return ColumnContract(
            "TIMESTAMP",
            None,
            None,
            None,
            column_type.timezone,
            column.nullable,
        )
    if isinstance(column_type, Integer):
        return ColumnContract(
            "NUMBER", None, None, None, None, column.nullable
        )
    if isinstance(column_type, String):
        return ColumnContract(
            "VARCHAR2",
            column_type.length,
            None,
            None,
            None,
            column.nullable,
        )
    if isinstance(column_type, Numeric):
        return ColumnContract(
            "NUMBER",
            None,
            column_type.precision,
            column_type.scale,
            None,
            column.nullable,
        )
    raise TypeError(
        f"未支持的 Entity 列类型：{column.table.name}.{column.name} "
        f"{column_type!r}"
    )


def catalog_column_contract(row) -> ColumnContract:
    data_type = str(row.data_type)
    if data_type == "RAW":
        family = "RAW"
        length = int(row.data_length)
        timezone = None
    elif data_type == "VARCHAR2":
        family = "VARCHAR2"
        length = int(row.char_length)
        timezone = None
    elif data_type == "CLOB":
        family = "CLOB"
        length = None
        timezone = None
    elif data_type.startswith("TIMESTAMP"):
        family = "TIMESTAMP"
        length = None
        timezone = "WITH TIME ZONE" in data_type
    elif data_type == "NUMBER":
        family = "NUMBER"
        length = None
        timezone = None
    elif data_type == "VECTOR":
        family = "VECTOR"
        length = None
        timezone = None
    elif data_type == "JSON":
        family = "JSON"
        length = None
        timezone = None
    else:
        raise TypeError(
            f"未支持的 Oracle 列类型：{row.table_name}.{row.column_name} "
            f"{data_type}"
        )
    return ColumnContract(
        family,
        length,
        (
            int(row.data_precision)
            if family == "NUMBER" and row.data_precision is not None
            else None
        ),
        (
            int(row.data_scale)
            if family == "NUMBER" and row.data_scale is not None
            else None
        ),
        timezone,
        row.nullable == "Y",
    )


async def check_schema() -> list[str]:
    """返回全部 Schema 漂移；空列表表示契约一致。"""
    settings = get_settings()
    oracle = settings.database.oracle
    require_oracle_listener(host=oracle.host, port=oracle.port)
    runtime = create_database_runtime(settings)
    try:
        async with runtime.session_factory() as session:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT table_name, column_name, data_type, data_length,
                               data_precision, data_scale, char_length, nullable
                        FROM user_tab_columns
                        WHERE table_name LIKE 'KBOT_OPS_%'
                        ORDER BY table_name, column_id
                        """
                    )
                )
            ).all()
            hitl_status_constraint = (
                await session.execute(
                    text(
                        """
                        SELECT search_condition_vc
                        FROM user_constraints
                        WHERE constraint_name = 'CK_OPS_HITL_STATUS'
                          AND table_name = 'KBOT_OPS_HITL'
                        """
                    )
                )
            ).scalar_one_or_none()
    finally:
        await runtime.close()

    catalog = {
        (row.table_name, row.column_name): catalog_column_contract(row)
        for row in rows
    }
    expected = {
        (entity.__tablename__, column.name.upper()): entity_column_contract(
            column
        )
        for entity in AIOPS_ENTITY_CLASSES
        for column in entity.__table__.columns
    }
    errors: list[str] = []
    missing = sorted(set(expected) - set(catalog))
    unexpected = sorted(set(catalog) - set(expected))
    if missing:
        errors.append(f"Oracle 缺少 Entity 列：{missing}")
    if unexpected:
        errors.append(f"Oracle 出现未映射列：{unexpected}")
    for key in sorted(set(expected) & set(catalog)):
        if expected[key] != catalog[key]:
            errors.append(
                f"{key[0]}.{key[1]} 不一致："
                f"entity={expected[key]} oracle={catalog[key]}"
            )
    if hitl_status_constraint is None:
        errors.append("Oracle 缺少约束 CK_OPS_HITL_STATUS")
    else:
        actual_statuses = set(
            re.findall(r"'([A-Z_]+)'", str(hitl_status_constraint))
        )
        if actual_statuses != HITL_STATUSES:
            errors.append(
                "CK_OPS_HITL_STATUS 不一致："
                f"expected={sorted(HITL_STATUSES)} "
                f"oracle={sorted(actual_statuses)}"
            )
    return errors


def main() -> int:
    try:
        errors = asyncio.run(check_schema())
    except RuntimeError as exc:
        print(f"AIOps Entity/Oracle Schema Preflight 失败：{exc}")
        return 2
    if errors:
        print("AIOps Entity/Oracle Schema 校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("AIOps Entity/Oracle Schema 校验通过：21 张表逐列一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
