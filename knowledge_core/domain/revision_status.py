"""Pure aggregation rules for Bundle Revision availability."""


def reduce_revision_status(members) -> str:
    members = list(members)
    manifest = next((item for item in members if item.document_role == "MANIFEST"), None)
    if manifest is None:
        return "FAILED" if members and all(_stable(item.member_status) for item in members) else "PROCESSING"
    if manifest.member_status in {"DECLARED", "RECEIVED", "PARSING", "INDEXING"}:
        return "PROCESSING"
    if manifest.member_status != "READY":
        return "FAILED"
    if any(not _stable(item.member_status) for item in members):
        return "PROCESSING"
    if any(item.member_status in {"FAILED", "SOURCE_UNAVAILABLE", "CANCELLED"} for item in members):
        return "PARTIAL"
    return "READY"


def _stable(status: str) -> bool:
    return status in {"READY", "FAILED", "SOURCE_UNAVAILABLE", "CANCELLED"}
