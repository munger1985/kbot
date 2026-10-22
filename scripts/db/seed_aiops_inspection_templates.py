"""为所有已启用 AIOps Domain 幂等写入默认数据库巡检模板。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

from sqlalchemy import func, select, text


ROOT = Path(
    os.environ.get("KBOT_REPO_ROOT")
    or Path(__file__).resolve().parents[2]
).resolve()
for source_root in reversed((
    ROOT / "packages" / "platform_core" / "src",
    ROOT / "services" / "aiops_agent" / "src",
)):
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

from aiops_agent.application.inspections import (
    compile_selected_check_steps,
    default_selected_check_ids,
    load_check_catalog,
)
from aiops_agent.entities import (
    InspectionTemplateEntity,
    InspectionTemplateVersionEntity,
)
from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7


DEFAULT_TEMPLATE_PROFILES = (
    ("日常巡检", "DAILY"),
    ("月度巡检", "MONTHLY"),
)
BOOTSTRAP_ACTOR = "system:inspection-template-bootstrap"


@dataclass(frozen=True)
class DefaultInspectionTemplateSeed:
    """一份可直接持久化的默认巡检模板定义。"""

    display_name: str
    schedule_type: str
    definition: dict[str, object]
    content_hash: str


@dataclass(frozen=True)
class SeedResult:
    """初始化结果摘要。"""

    domain_count: int
    created_count: int
    updated_count: int
    existing_count: int


def _content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_default_template_seeds() -> tuple[DefaultInspectionTemplateSeed, ...]:
    """从当前 Check Catalog 生成跨数据库默认模板快照。"""
    catalog = load_check_catalog()
    seeds: list[DefaultInspectionTemplateSeed] = []
    for display_name, schedule_type in DEFAULT_TEMPLATE_PROFILES:
        catalog_schedule_type = (
            "WEEKLY" if schedule_type == "MONTHLY" else schedule_type
        )
        selected = default_selected_check_ids(catalog_schedule_type)
        definition: dict[str, object] = {
            "schema_version": "INSPECTION_TEMPLATE.v1",
            "display_name": display_name,
            "catalog_hash": catalog.catalog_hash,
            "selected_check_ids": list(selected),
            "evidence_steps": list(
                compile_selected_check_steps(
                    selected,
                    schedule_type=catalog_schedule_type,
                )
            ),
        }
        seeds.append(
            DefaultInspectionTemplateSeed(
                display_name=display_name,
                schedule_type=schedule_type,
                definition=definition,
                content_hash=_content_hash(definition),
            )
        )
    return tuple(seeds)


async def seed_default_inspection_templates(
    *, check_only: bool = False,
) -> SeedResult:
    """锁定每个 AIOps Domain，并仅补写缺少的默认模板。"""
    runtime = create_database_runtime(get_settings())
    seeds = build_default_template_seeds()
    created_count = 0
    updated_count = 0
    existing_count = 0
    try:
        async with runtime.session_factory() as session:
            domains = (
                await session.execute(
                    text(
                        """
                        SELECT domain.DOMAIN_ID
                        FROM KBOT_PLATFORM_DOMAIN domain
                        JOIN KBOT_APP_DOMAIN app_domain
                          ON app_domain.DOMAIN_ID = domain.DOMAIN_ID
                         AND app_domain.APP_ID = 'aiops'
                         AND app_domain.STATUS = 'ACTIVE'
                        WHERE domain.STATUS = 'ACTIVE'
                        ORDER BY domain.DOMAIN_ID
                        """
                    )
                )
            ).scalars().all()
            if not domains:
                raise RuntimeError("没有可初始化的已启用 AIOps Domain")
            for raw_domain_id in domains:
                domain_id = int(raw_domain_id)
                await session.execute(
                    text(
                        "SELECT DOMAIN_ID FROM KBOT_PLATFORM_DOMAIN "
                        "WHERE DOMAIN_ID = :domain_id FOR UPDATE"
                    ),
                    {"domain_id": domain_id},
                )
                rows = list(
                    (
                        await session.scalars(
                            select(InspectionTemplateEntity).where(
                                InspectionTemplateEntity.domain_id == domain_id
                            )
                        )
                    ).all()
                )
                rows_by_name: dict[str, list[InspectionTemplateEntity]] = {}
                for row in rows:
                    rows_by_name.setdefault(str(row.display_name), []).append(row)
                for seed in seeds:
                    matching = rows_by_name.get(seed.display_name, [])
                    if len(matching) > 1:
                        raise RuntimeError(
                            f"Domain {domain_id} 存在重名巡检模板："
                            f"{seed.display_name}"
                        )
                    if matching:
                        row = matching[0]
                        version = await session.get(
                            InspectionTemplateVersionEntity,
                            row.current_version_id,
                        )
                        if version is None:
                            raise RuntimeError(
                                f"Domain {domain_id} 的 {seed.display_name} "
                                "当前版本不存在"
                            )
                        managed_outdated = (
                            str(row.created_by) == BOOTSTRAP_ACTOR
                            and str(version.content_hash) != seed.content_hash
                        )
                        if managed_outdated and check_only:
                            raise RuntimeError(
                                f"Domain {domain_id} 的内置模板 "
                                f"{seed.display_name} 不是当前目录版本"
                            )
                        if managed_outdated:
                            next_version = int(
                                await session.scalar(
                                    select(
                                        func.coalesce(
                                            func.max(
                                                InspectionTemplateVersionEntity.version_no
                                            ),
                                            0,
                                        )
                                    ).where(
                                        InspectionTemplateVersionEntity.inspection_template_id
                                        == row.inspection_template_id
                                    )
                                )
                            ) + 1
                            version_id = uuid7()
                            session.add(
                                InspectionTemplateVersionEntity(
                                    inspection_template_version_id=version_id,
                                    domain_id=domain_id,
                                    inspection_template_id=(
                                        row.inspection_template_id
                                    ),
                                    version_no=next_version,
                                    definition_json=seed.definition,
                                    content_hash=seed.content_hash,
                                    created_by=BOOTSTRAP_ACTOR,
                                )
                            )
                            row.current_version_id = version_id
                            row.updated_by = BOOTSTRAP_ACTOR
                            updated_count += 1
                        else:
                            existing_count += 1
                        continue
                    if check_only:
                        raise RuntimeError(
                            f"Domain {domain_id} 缺少默认巡检模板："
                            f"{seed.display_name}"
                        )
                    template_id = uuid7()
                    version_id = uuid7()
                    session.add(
                        InspectionTemplateEntity(
                            inspection_template_id=template_id,
                            domain_id=domain_id,
                            display_name=seed.display_name,
                            status="ACTIVE",
                            current_version_id=version_id,
                            created_by=BOOTSTRAP_ACTOR,
                            updated_by=BOOTSTRAP_ACTOR,
                        )
                    )
                    session.add(
                        InspectionTemplateVersionEntity(
                            inspection_template_version_id=version_id,
                            domain_id=domain_id,
                            inspection_template_id=template_id,
                            version_no=1,
                            definition_json=seed.definition,
                            content_hash=seed.content_hash,
                            created_by=BOOTSTRAP_ACTOR,
                        )
                    )
                    created_count += 1
            if check_only:
                await session.rollback()
            else:
                await session.commit()
        return SeedResult(
            domain_count=len(domains),
            created_count=created_count,
            updated_count=updated_count,
            existing_count=existing_count,
        )
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验每个 AIOps Domain 是否已有两份默认模板",
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(
            seed_default_inspection_templates(check_only=args.check_only)
        )
    except Exception as exc:
        print(f"默认巡检模板初始化失败：{exc}", file=sys.stderr)
        return 1
    action = "校验通过" if args.check_only else "初始化完成"
    print(
        f"默认巡检模板{action}：Domain={result.domain_count}，"
        f"新增={result.created_count}，更新={result.updated_count}，"
        f"已存在={result.existing_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
