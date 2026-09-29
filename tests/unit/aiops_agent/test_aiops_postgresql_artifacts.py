"""PostgreSQL pgBadger 外部报告的安全校验与本地存储边界。"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
from pathlib import Path
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from aiops_agent.application.postgresql_artifacts import (
    PgBadgerArtifactRejected,
    PostgreSQLArtifactStore,
    validate_pgbadger_artifact,
)
from aiops_agent.application.workload import WorkloadService
from aiops_agent.contracts.report import ReportContent
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


def test_pgbadger_json_and_gzip_are_normalized_deterministically() -> None:
    plain = validate_pgbadger_artifact(
        file_name="report.json",
        content_type="application/json",
        body=b'{"z":2,"a":1}',
    )
    compressed = validate_pgbadger_artifact(
        file_name="report.json.gz",
        content_type="application/gzip",
        body=gzip.compress(b'{"a":1,"z":2}'),
    )

    assert plain.body == b'{"a":1,"z":2}'
    assert compressed.body == plain.body
    assert compressed.content_hash == plain.content_hash


@pytest.mark.parametrize(
    "body",
    (
        b"<html><script>alert(1)</script></html>",
        b'<html><img src="https://evil.example/x"></html>',
        b'<html><svg onload="alert(1)"></svg></html>',
        b'<html><div style="background:url(https://evil.example/x)"></div></html>',
        b'<html><meta http-equiv="refresh" content="0;url=https://evil.example"></html>',
        b'<html><a href="javascript:alert(1)">x</a></html>',
    ),
)
def test_pgbadger_html_rejects_active_content(body: bytes) -> None:
    with pytest.raises(
        PgBadgerArtifactRejected,
        match="禁止脚本、嵌入对象和外部资源",
    ):
        validate_pgbadger_artifact(
            file_name="report.html",
            content_type="text/html",
            body=body,
        )


def test_pgbadger_rejects_unsupported_type_and_bounded_gzip_expansion() -> None:
    with pytest.raises(PgBadgerArtifactRejected) as unsupported:
        validate_pgbadger_artifact(
            file_name="report.txt",
            content_type="text/plain",
            body=b"not a report",
        )
    assert unsupported.value.code == "PGBADGER_TYPE_UNSUPPORTED"

    with patch(
        "aiops_agent.application.postgresql_artifacts._MAX_UNCOMPRESSED_BYTES",
        16,
    ):
        with pytest.raises(PgBadgerArtifactRejected) as expanded:
            validate_pgbadger_artifact(
                file_name="report.json.gz",
                content_type="application/gzip",
                body=gzip.compress(b'{"payload":"' + b"x" * 64 + b'"}'),
            )
    assert expanded.value.code == "PGBADGER_EXPANDED_SIZE_INVALID"


def test_artifact_store_enforces_permissions_hash_and_root(tmp_path: Path) -> None:
    store = PostgreSQLArtifactStore(tmp_path / "pgbadger")
    artifact = validate_pgbadger_artifact(
        file_name="report.html",
        content_type="text/html",
        body=b"<html><body>pgBadger</body></html>",
    )
    artifact_id = uuid7()

    uri = store.write(artifact_id=artifact_id, artifact=artifact)
    stored = Path(uri.removeprefix("file://"))

    assert stored.stat().st_mode & 0o777 == 0o600
    assert store.read(uri, expected_hash=artifact.content_hash) == artifact.body
    with pytest.raises(ValueError, match="Hash"):
        store.read(uri, expected_hash="0" * 64)
    with pytest.raises(ValueError, match="越界"):
        store.read((tmp_path / "outside.html").as_uri(), expected_hash="0" * 64)

    stored.write_bytes(b"tampered")
    assert hashlib.sha256(stored.read_bytes()).hexdigest() != artifact.content_hash
    with pytest.raises(ValueError, match="Hash"):
        store.read(uri, expected_hash=artifact.content_hash)


def test_import_registers_formal_report_without_promoting_trust(
    tmp_path: Path,
) -> None:
    domain_id, target_id, actor_id = 100, uuid7(), "dba@example.com"
    binding = SimpleNamespace(agent_id=uuid7(), status="ACTIVE")
    active = SimpleNamespace(agent_id=binding.agent_id, binding_id=uuid7())
    target = SimpleNamespace(
        target_id=target_id,
        domain_id=domain_id,
        db_type="POSTGRESQL",
        display_name="生产PG",
        security_level=2,
    )
    uow = SimpleNamespace(
        targets=SimpleNamespace(
            get_scoped=AsyncMock(return_value=target),
            list_agent_bindings=AsyncMock(return_value=[binding]),
        ),
        agents=SimpleNamespace(get_active=AsyncMock(return_value=active)),
        runs=SimpleNamespace(
            get_by_idempotency=AsyncMock(return_value=None),
            add_run=AsyncMock(),
            add_task=AsyncMock(),
            add_artifact=AsyncMock(),
        ),
        inspections=SimpleNamespace(
            add_report=AsyncMock(),
            add_report_sources=AsyncMock(),
        ),
        commit=AsyncMock(),
    )
    service = WorkloadService(
        uow_factory=lambda: _context(uow),
        postgresql_artifact_store=PostgreSQLArtifactStore(
            tmp_path / "pgbadger"
        ),
    )
    start = datetime(2026, 9, 22, tzinfo=UTC)

    view = asyncio.run(
        service.import_pgbadger_artifact(
            domain_id=domain_id,
            actor_id=actor_id,
            target_id=target_id,
            period_start=start,
            period_end=start + timedelta(hours=1),
            file_name="report.json",
            content_type="application/json",
            body=b'{"pgbadger":true}',
            idempotency_key="pgbadger-import-1",
            trace_id="trace-pgbadger-1",
        )
    )

    assert view.report_origin == "EXTERNAL_IMPORTED"
    artifacts = [call.args[0] for call in uow.runs.add_artifact.await_args_list]
    assert [item.artifact_type for item in artifacts] == [
        "POSTGRESQL_PGBADGER_REPORT",
        "REPORT_CONTENT",
    ]
    assert all(item.trust_level == "EXTERNAL_IMPORTED" for item in artifacts)
    content = ReportContent.model_validate(artifacts[1].payload_json)
    assert content.report_type == "POSTGRESQL_PGBADGER"
    assert content.gaps[0]["code"] == "EXTERNAL_SOURCE_NOT_SOURCE_VERIFIED"
    formal = uow.inspections.add_report.await_args.args[0]
    assert formal.content_artifact_id == artifacts[1].artifact_id
    source = uow.inspections.add_report_sources.await_args.args[0][0]
    assert source.source_artifact_id == artifacts[0].artifact_id
    uow.runs.get_artifact_scoped = AsyncMock(return_value=artifacts[0])

    body, content_type, file_name = asyncio.run(
        service.get_report_artifact_content(
            domain_id=domain_id,
            artifact_id=artifacts[0].artifact_id,
        )
    )

    assert body == b'{"pgbadger":true}'
    assert content_type == "application/json"
    assert file_name == "report.json"
