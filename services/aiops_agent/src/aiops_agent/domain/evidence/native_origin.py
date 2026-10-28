"""同一原生数据库家族的保守来源归组；别名或重启不产生独立互证。"""

from aiops_agent.playbooks import canonical_hash


def database_origin_anchor(database_type, identity):
    key = {"POSTGRESQL": "system_identifier", "MYSQL": "server_uuid", "ORACLE": "dbid"}.get(str(database_type))
    if key is None or not identity.get(key):
        return None
    # Oracle RAC/主备保守按 DBID 归组，不能因连接不同或 Artifact 更多而自动增加因果支持。
    return "native-database:" + str(database_type).lower() + ":" + canonical_hash({key: str(identity[key])})
