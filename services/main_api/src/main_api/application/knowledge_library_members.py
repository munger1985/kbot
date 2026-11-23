"""对接工作域 IAM：仅返回当前知识 App 的可用成员。"""

APP = "knowledge_retrieval"
USE = APP + ":use"
MANAGE = APP + ":knowledge_core_manage"


async def options(app_state, context):
    access = app_state.access_control_service
    scope = {"app_id": APP, "domain_id": int(context.domain_id)}
    members = await access.list_members_with_permission(**scope, permission_code=USE)
    managers = {str(row["user_id"]) for row in await access.list_members_with_permission(**scope, permission_code=MANAGE)}
    return [{"user_id": str(row["user_id"]), "display_name": row.get("display_name") or str(row["user_id"]),
        "can_manage": str(row["user_id"]) in managers} for row in members]


def grants_allowed(grants, candidates):
    eligible = {entry["user_id"]: entry["can_manage"] for entry in candidates}
    return bool(grants) and all(subject in eligible and ("MANAGE" not in perms or eligible[subject])
        for subject, perms in grants.items()) and any(eligible.get(subject) and "MANAGE" in perms
        for subject, perms in grants.items())
