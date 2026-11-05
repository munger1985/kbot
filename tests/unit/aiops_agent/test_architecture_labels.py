"""数据库架构与主机承载、存储依赖独立判断，不根据主备角色伪造复制关系。"""

from types import SimpleNamespace as NS

import pytest

from aiops_agent.application.configuration.business_system_service import _architecture_labels


def database(deployment="SINGLE_INSTANCE"):
    return NS(resource=NS(database=NS(db_type="ORACLE", version_code="19c", deployment_type=deployment)))


def relation(kind, status="CONFIRMED"):
    return NS(relation_type_key=kind, confirmation_status=status)


@pytest.mark.parametrize("kind", ["RUNS_ON", "USES_STORAGE", "DEPENDS_ON", "LOAD_BALANCES"])
def test_infrastructure_relations_do_not_hide_single_instance(kind):
    host = NS(resource=NS(database=None))
    assert _architecture_labels([database(), host], [relation(kind)]) == ("SINGLE_INSTANCE",)


@pytest.mark.parametrize("kind,label", [
    ("ORACLE_RAC", "RAC"), ("DATA_GUARD", "DATA_GUARD"), ("OGG", "OGG"),
    ("MYSQL_REPLICATION", "MYSQL_REPLICATION"), ("POSTGRES_STREAMING", "POSTGRES_STREAMING"),
    ("DEBEZIUM_CDC", "DEBEZIUM_CDC"), ("FLINK_CDC", "FLINK_CDC"), ("DATAX", "DATAX"),
])
def test_confirmed_database_relations_coexist_with_host_relations(kind, label):
    assert _architecture_labels([database(), database()], [relation(kind), relation("RUNS_ON")]) == (label,)


def test_rac_without_confirmed_membership_is_not_reported_as_single_instance():
    assert _architecture_labels([database("RAC")], [relation("RUNS_ON")]) == ()
    assert _architecture_labels([database("RAC"), database("RAC")], [relation("ORACLE_RAC", "SUGGESTED")]) == ()


def test_multiple_databases_without_relationship_do_not_become_a_fake_cluster():
    assert _architecture_labels([database(), database()], [relation("RUNS_ON")]) == ()


def test_rac_and_dg_are_preserved_together():
    assert _architecture_labels([database("RAC"), database("RAC"), database()], [
        relation("ORACLE_RAC"), relation("DATA_GUARD"), relation("RUNS_ON"),
    ]) == ("RAC_DATA_GUARD", "RAC", "DATA_GUARD")
