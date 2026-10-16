from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "database" / "oracle" / "operations" / "upgrade_kbot_41_retire_km_asset.sql"


KM_TABLES = {
    "KBOT_KM_SOURCE",
    "KBOT_KM_ASSET",
    "KBOT_KM_ASSET_REVISION",
    "KBOT_KM_ATTACHMENT",
    "KBOT_KM_JOB",
    "KBOT_KM_AGENT",
    "KBOT_KM_AGENT_GRANT",
    "KBOT_KM_AGENT_VERSION",
    "KBOT_KM_SLACK_INBOX",
    "KBOT_KM_SLACK_THREAD",
    "KBOT_KM_SLACK_DELIVERY",
}


def test_kbot_41_upgrade_retires_every_km_private_table() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for table_name in KM_TABLES:
        assert f"drop_table_if_exists('{table_name}')" in source
    assert "drop_view_if_exists('KBOT_V_KM_ASSET_CURRENT')" in source
    assert "drop_view_if_exists('KBOT_V_KM_ASSET_SEARCHABLE')" in source
    assert "CASCADE CONSTRAINTS PURGE" in source


def test_kbot_41_upgrade_guards_dependencies_and_cleans_authorization() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "USER_CONSTRAINTS child_constraint" in source
    assert "USER_DEPENDENCIES dependency_row" in source
    assert "OWNER_APP_ID = 'km_asset'" in source
    assert "DELETE FROM KBOT_APP_API_CREDENTIAL" in source
    assert "DELETE FROM KBOT_PLATFORM_APP WHERE APP_ID = 'km_asset'" in source
    assert "KBot 4.1 KM Asset 退役验证通过" in source


def test_kbot_41_upgrade_preserves_shared_foundations() -> None:
    source = SCRIPT.read_text(encoding="utf-8").upper()

    for forbidden_statement in (
        "DROP TABLE KBOT_KC_",
        "DROP TABLE KBOT_DQ_",
        "DROP TABLE KBOT_AGENT_",
        "DELETE FROM KBOT_KC_",
        "DELETE FROM KBOT_DQ_",
        "DELETE FROM KBOT_AGENT_",
        "DELETE FROM KBOT_PLATFORM_DOMAIN",
        "DELETE FROM KBOT_MANAGED_CREDENTIAL",
    ):
        assert forbidden_statement not in source
