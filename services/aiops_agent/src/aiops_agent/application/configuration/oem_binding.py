"""OEM 候选与 AIOps 数据库资源的范围及已观测身份校验。"""

from aiops_agent.application.errors import validation_failed


def validate_oem_database_binding(*, target, locator):
    if locator.get("database_type") != target.db_type:
        raise validation_failed("OEM 候选数据库引擎与资源不一致，请重新发现并选择")
    if target.db_type != "ORACLE":
        return
    scope = locator.get("oracle_container_scope")
    expected = getattr(target, "oracle_container_scope", None)
    if scope not in {"CDB_ROOT", "PDB", "NON_CDB"} or expected != scope:
        raise validation_failed("OEM 候选与数据库资源的 CDB/PDB/非 CDB 范围不一致")
    observed_scope = getattr(target, "observed_oracle_container_scope", None)
    if observed_scope is not None and observed_scope != scope:
        raise validation_failed("OEM 候选范围与数据库连接实际观测范围不一致")
    if scope == "PDB":
        actual = str(locator.get("oracle_pdb_name") or "")
        expected_pdb = str(getattr(target, "oracle_pdb_name", None) or "")
        if not actual or actual.casefold() != expected_pdb.casefold():
            raise validation_failed("OEM 候选 PDB 与资源声明的 PDB 名称不一致")
        observed_pdb = getattr(target, "observed_oracle_container_name", None)
        if observed_pdb and actual.casefold() != observed_pdb.casefold():
            raise validation_failed("OEM 候选 PDB 与数据库连接实际观测身份不一致")
    observed_database = getattr(target, "observed_oracle_database_name", None)
    actual_database = locator.get("database_name")
    if observed_database and actual_database and observed_database.casefold() != actual_database.casefold():
        raise validation_failed("OEM 候选与数据库连接实际观测的数据库名称不一致")
    if getattr(target, "deployment_type", None) == "RAC":
        if locator.get("rac_option") != "YES":
            raise validation_failed("RAC 节点必须选择经过确认的 OEM RAC 数据库实例")
        number = locator.get("rac_instance_number")
        if not number or not number.isdecimal() or int(number) != int(target.rac_node_number or 0):
            raise validation_failed("OEM RAC 实例编号与数据库资源节点编号不一致")
