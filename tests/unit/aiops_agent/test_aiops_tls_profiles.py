"""PostgreSQL 受控 TLS Profile 的契约和文件系统边界测试。"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from aiops_agent.executor.tls_profiles import TLSProfileError, TLSProfileResolver
from platform_core.contracts.aiops.configuration import (
    DatabaseCredentialInput,
    TargetCreate,
    TargetEndpoint,
)
from platform_core.contracts.aiops.executor import DiagnosticConnectionProfile


def _credential() -> DatabaseCredentialInput:
    return DatabaseCredentialInput(username="diagnostic", password="secret")


def test_tls_profile_contract_requires_tls_and_rejects_path_refs() -> None:
    with pytest.raises(ValidationError, match="只能在启用TLS时配置"):
        TargetEndpoint(
            host="db.internal",
            port=5432,
            database="app",
            tls_enabled=False,
            tls_profile_ref="prod-ca",
        )
    with pytest.raises(ValidationError):
        DiagnosticConnectionProfile(
            host="db.internal",
            port=5432,
            database="app",
            tls_profile_ref="../prod-ca",
        )


@pytest.mark.parametrize(
    ("db_type", "endpoint"),
    (
        (
            "MYSQL",
            TargetEndpoint(
                host="db.internal",
                port=3306,
                database="app",
                tls_profile_ref="prod-ca",
            ),
        ),
        (
            "ORACLE",
            TargetEndpoint(
                host="db.internal",
                port=1521,
                service="PDB1",
                tls_profile_ref="prod-ca",
            ),
        ),
    ),
)
def test_only_postgresql_target_accepts_tls_profile(db_type, endpoint) -> None:
    with pytest.raises(ValidationError, match="只有PostgreSQL"):
        TargetCreate(
            display_name="database",
            db_type=db_type,
            environment="PROD",
            endpoint=endpoint,
            readonly_connection_enabled=True,
            diagnostic_credential=_credential(),
        )

    postgresql = TargetCreate(
        display_name="postgresql",
        db_type="POSTGRESQL",
        environment="PROD",
        endpoint=TargetEndpoint(
            host="db.internal",
            port=5432,
            database="app",
            tls_profile_ref="prod-ca",
        ),
        readonly_connection_enabled=True,
        diagnostic_credential=_credential(),
    )
    assert postgresql.endpoint is not None
    assert postgresql.endpoint.tls_profile_ref == "prod-ca"


def test_resolver_builds_verified_context_from_fixed_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "prod-ca"
    profile.mkdir(parents=True)
    (profile / "ca.pem").write_text("ca", encoding="utf-8")
    (profile / "client-cert.pem").write_text("cert", encoding="utf-8")
    (profile / "client-key.pem").write_text("key", encoding="utf-8")
    context = Mock()

    with patch(
        "aiops_agent.executor.tls_profiles.ssl.create_default_context",
        return_value=context,
    ) as create_context:
        resolved = TLSProfileResolver(tmp_path / "profiles").resolve("prod-ca")

    assert resolved is context
    create_context.assert_called_once_with(cafile=str(profile / "ca.pem"))
    context.load_cert_chain.assert_called_once_with(
        certfile=str(profile / "client-cert.pem"),
        keyfile=str(profile / "client-key.pem"),
    )
    assert context.check_hostname is True


def test_resolver_rejects_symlinks_and_incomplete_client_pair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profiles"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ca.pem").write_text("ca", encoding="utf-8")
    root.mkdir()
    (root / "linked-profile").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TLSProfileError, match="目录不能是符号链接"):
        TLSProfileResolver(root).resolve("linked-profile")

    profile = root / "prod-ca"
    profile.mkdir()
    (profile / "ca.pem").symlink_to(outside / "ca.pem")
    with pytest.raises(TLSProfileError, match="文件无效"):
        TLSProfileResolver(root).resolve("prod-ca")

    (profile / "ca.pem").unlink()
    (profile / "ca.pem").write_text("ca", encoding="utf-8")
    (profile / "client-cert.pem").write_text("cert", encoding="utf-8")
    with pytest.raises(TLSProfileError, match="必须同时配置"):
        TLSProfileResolver(root).resolve("prod-ca")
