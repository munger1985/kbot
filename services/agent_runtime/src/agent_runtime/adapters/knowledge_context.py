"""知识 Skill 只接受冻结 App owner，不回落公共 KC。"""

from knowledge_algorithms.contracts.errors import require


def owner_app_id(context):
    agent = context.config_snapshot.get("agent")
    require(type(agent) is dict and agent.get("owner_app_id") in {"knowledge_retrieval", "aiops"},
            "运行快照缺少已登记的 App owner")
    require(getattr(context, "owner_app_id", agent["owner_app_id"]) == agent["owner_app_id"],
            "运行 owner 与冻结 Agent 不一致")
    return agent["owner_app_id"]
