"""将文件 Prompt Catalog 幂等同步到平台数据库。"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from platform_core.identity import uuid7

from .catalog import load_prompt_catalog


async def sync_prompt_catalog(
    connection: AsyncConnection,
    *,
    selected_services: set[str],
    environment: str,
    actor_id: str = "prompt-catalog-sync",
) -> int:
    """初始化文件基准；开发环境只允许更新 FILE_SEED 1.0.0。"""
    catalog = load_prompt_catalog()
    entries = catalog.for_services(selected_services)
    is_development = environment.lower() in {
        "dev",
        "development",
        "debug",
    }
    invalid_entries = [
        f"{entry.prompt_key}@{entry.version}"
        for entry in entries
        if entry.version != "1.0.0"
    ]
    if invalid_entries:
        raise RuntimeError(
            "文件 Prompt 基准版本必须固定为 1.0.0："
            + "、".join(invalid_entries)
        )
    for entry in entries:
        row = (
            await connection.execute(
                text(
                    """
                    SELECT prompt_id, active_version_id
                    FROM KBOT_PLATFORM_PROMPT
                    WHERE prompt_key = :prompt_key
                    FOR UPDATE
                    """
                ),
                {"prompt_key": entry.prompt_key},
            )
        ).one_or_none()
        if row is None:
            definition_created = True
            prompt_id = uuid7().bytes
            active_version_id = None
            await connection.execute(
                text(
                    """
                    INSERT INTO KBOT_PLATFORM_PROMPT (
                        prompt_id, prompt_key, owner_service, purpose,
                        active_version_id, row_version, created_by, updated_by
                    ) VALUES (
                        :prompt_id, :prompt_key, :owner_service, :purpose,
                        NULL, 1, :actor_id, :actor_id
                    )
                    """
                ),
                {
                    "prompt_id": prompt_id,
                    "prompt_key": entry.prompt_key,
                    "owner_service": entry.owner_service,
                    "purpose": entry.purpose,
                    "actor_id": actor_id,
                },
            )
        else:
            definition_created = False
            prompt_id, active_version_id = row
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_PLATFORM_PROMPT
                    SET owner_service = :owner_service,
                        purpose = :purpose,
                        updated_by = :actor_id,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE prompt_id = :prompt_id
                      AND (
                          owner_service <> :owner_service
                          OR purpose <> :purpose
                      )
                    """
                ),
                {
                    "prompt_id": prompt_id,
                    "owner_service": entry.owner_service,
                    "purpose": entry.purpose,
                    "actor_id": actor_id,
                },
            )

        version_row = (
            await connection.execute(
                text(
                    """
                    SELECT prompt_version_id, content_sha256, source
                    FROM KBOT_PLATFORM_PROMPT_VERSION
                    WHERE prompt_id = :prompt_id
                      AND version = :version
                    """
                ),
                {"prompt_id": prompt_id, "version": entry.version},
            )
        ).one_or_none()
        if version_row is not None:
            prompt_version_id, existing_hash, source = version_row
            if str(source) == "FILE_SEED" and is_development:
                await connection.execute(
                    text(
                        """
                        UPDATE KBOT_PLATFORM_PROMPT_VERSION
                        SET content = :content,
                            content_sha256 = :content_sha256,
                            input_variables_json = :input_variables_json,
                            output_schema_ref = :output_schema_ref
                        WHERE prompt_version_id = :prompt_version_id
                        """
                    ),
                    {
                        "content": entry.content,
                        "content_sha256": entry.sha256,
                        "input_variables_json": (
                            "["
                            + ",".join(
                                f'"{value}"'
                                for value in entry.input_variables
                            )
                            + "]"
                        ),
                        "output_schema_ref": entry.output_schema,
                        "prompt_version_id": prompt_version_id,
                    },
                )
            elif (
                str(source) == "FILE_SEED"
                and str(existing_hash) != entry.sha256
            ):
                raise RuntimeError(
                    "非开发环境不能覆盖文件 Prompt 基准："
                    f"{entry.prompt_key}@{entry.version}"
                )
        else:
            prompt_version_id = uuid7().bytes
            await connection.execute(
                text(
                    """
                    INSERT INTO KBOT_PLATFORM_PROMPT_VERSION (
                        prompt_version_id, prompt_id, version, content,
                        content_sha256, input_variables_json,
                        output_schema_ref, status, source, created_by
                    ) VALUES (
                        :prompt_version_id, :prompt_id, :version, :content,
                        :content_sha256, :input_variables_json,
                        :output_schema_ref, :status, 'FILE_SEED', :actor_id
                    )
                    """
                ),
                {
                    "prompt_version_id": prompt_version_id,
                    "prompt_id": prompt_id,
                    "version": entry.version,
                    "content": entry.content,
                    "content_sha256": entry.sha256,
                    "input_variables_json": (
                        "["
                        + ",".join(
                            f'"{value}"' for value in entry.input_variables
                        )
                        + "]"
                    ),
                    "output_schema_ref": entry.output_schema,
                    "status": "ACTIVE" if entry.active else "RETIRED",
                    "actor_id": actor_id,
                },
            )
        if entry.active and definition_created:
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_PLATFORM_PROMPT_VERSION
                    SET status = 'RETIRED'
                    WHERE prompt_id = :prompt_id
                      AND prompt_version_id <> :prompt_version_id
                      AND status = 'ACTIVE'
                    """
                ),
                {
                    "prompt_id": prompt_id,
                    "prompt_version_id": prompt_version_id,
                },
            )
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_PLATFORM_PROMPT
                    SET active_version_id = :prompt_version_id,
                        row_version = row_version + 1,
                        updated_by = :actor_id,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE prompt_id = :prompt_id
                      AND (
                          active_version_id IS NULL
                          OR active_version_id <> :prompt_version_id
                      )
                    """
                ),
                {
                    "prompt_version_id": prompt_version_id,
                    "prompt_id": prompt_id,
                    "actor_id": actor_id,
                },
            )
    await connection.commit()
    return len(entries)
