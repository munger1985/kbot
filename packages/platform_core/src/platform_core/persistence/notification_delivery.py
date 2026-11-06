"""外部投递物理结构的共同定义；类型和表名由平台数据库适配。"""

from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Index, String, Table, Text


def define_external_tables(*, metadata, prefix, id_type, domain_type, json_type, timestamp_type,
                           tenant_type=None, existing_job=None):
    """不建立业务 CHECK／UNIQUE；稳定主键和事务锁维护业务幂等。"""
    def col(name, type_, *args, **kw):
        return Column(name, type_, *args, **kw)

    def scope(resource=False, nullable=False):
        columns = [col("tenant_id", tenant_type, nullable=False)] if tenant_type is not None else []
        return columns + ([col("domain_id", domain_type, nullable=nullable)] if resource else [])

    def audit():
        return [col("created_at", timestamp_type, nullable=False),
                col("updated_at", timestamp_type, nullable=False),
                col("created_by", String(256), nullable=False),
                col("updated_by", String(256), nullable=False),
                col("row_version", BigInteger, nullable=False, default=1)]

    def identity():
        return col("id", id_type, primary_key=True)

    def table(kind, *columns, resource=False, nullable_domain=False):
        name = (prefix + kind).upper() if prefix.isupper() else prefix + kind
        return Table(name, metadata, identity(), *scope(resource, nullable_domain), *columns, *audit())

    connection = table("connection",
        col("name", String(128), nullable=False), col("provider", String(24), nullable=False),
        col("app_id", String(128), nullable=False), col("credential_ref", String(512), nullable=False),
        col("credential_version", BigInteger, nullable=False),
        col("enabled", Boolean, nullable=False, default=False),
        col("health", String(24), nullable=False), col("last_error_code", String(128)),
        col("checked_at", timestamp_type), col("next_send_at", timestamp_type))
    binding = table("binding",
        col("connection_id", id_type, ForeignKey(connection.c.id), nullable=False),
        col("actor_id", String(256)), col("target_type", String(16), nullable=False),
        col("external_id", String(256), nullable=False), col("status", String(24), nullable=False),
        col("verified_at", timestamp_type), col("challenge_hash", String(128)),
        col("challenge_ref", String(512)),
        col("challenge_salt", String(64)), col("challenge_expires_at", timestamp_type),
        col("challenge_attempts", BigInteger, nullable=False, default=0),
        col("challenge_sent_at", timestamp_type), col("next_send_at", timestamp_type))
    policy = table("policy",
        col("connection_id", id_type, ForeignKey(connection.c.id), nullable=False),
        col("event_types", json_type, nullable=False), col("binding_ids", json_type, nullable=False),
        col("actor_ids", json_type, nullable=False),
        col("enabled", Boolean, nullable=False, default=False))
    entry = table("entry",
        col("recipient_actor_id", String(256)), col("audience", String(16), nullable=False),
        col("event_type", String(160), nullable=False), col("route_key", String(160), nullable=False),
        col("route_params", json_type, nullable=False), col("expires_at", timestamp_type, nullable=False),
        resource=True)
    job_columns = [
        col("source_event_id", id_type, nullable=False), col("event_type", String(160), nullable=False),
        col("original_severity", String(16)), col("coalesce_key", String(64)),
        col("connection_id", id_type, ForeignKey(connection.c.id), nullable=False),
        col("connection_version", BigInteger, nullable=False),
        col("binding_id", id_type, ForeignKey(binding.c.id)),
        col("binding_version", BigInteger, nullable=False),
        col("recipient_actor_id", String(256)), col("target_type", String(16), nullable=False),
        col("external_id", String(256), nullable=False), col("purpose", String(24), nullable=False),
        col("template_version", BigInteger, nullable=False), col("generation", BigInteger, nullable=False),
        col("parent_job_id", id_type), col("entry_id", id_type, ForeignKey(entry.c.id)),
        col("content", json_type, nullable=False), col("content_hash", String(64), nullable=False),
        col("provider_uuid", String(50), nullable=False), col("status", String(24), nullable=False),
        col("attempt_count", BigInteger, nullable=False, default=0),
        col("retry_count", BigInteger, nullable=False, default=0),
        col("available_at", timestamp_type, nullable=False), col("expires_at", timestamp_type, nullable=False),
        col("lease_token", id_type), col("lease_expires_at", timestamp_type),
        col("first_sent_at", timestamp_type), col("network_started_at", timestamp_type),
        col("message_id", String(256)), col("last_error_code", String(128)),
    ]
    if existing_job is not None:
        job = existing_job
        job.append_column(col("domain_id", domain_type))
        for column in job_columns:
            job.append_column(column)
    else:
        job = table("job", *job_columns, resource=True, nullable_domain=True)
    attempt = table("attempt",
        col("job_id", id_type, ForeignKey(job.c.id), nullable=False),
        col("attempt_number", BigInteger, nullable=False), col("lease_token", id_type, nullable=False),
        col("started_at", timestamp_type, nullable=False), col("finished_at", timestamp_type),
        col("status", String(24), nullable=False), col("error_code", String(128)),
        col("request_id", String(256)), col("message_id", String(256)),
        col("elapsed_ms", BigInteger), resource=True, nullable_domain=True)
    audit_event = table("audit",
        col("action", String(80), nullable=False), col("object_id", id_type, nullable=False),
        col("outcome", String(128), nullable=False))
    # 全局频控行由部署初始化；缺失时明确阻断，不以竞争性惰性插入取代初始化。
    throttle = Table((prefix + "throttle").upper() if prefix.isupper() else prefix + "throttle", metadata,
        col("id", String(32), primary_key=True), col("next_send_at", timestamp_type, nullable=False))
    for obj, suffix, fields in (
        (connection, "conn_scope", (["tenant_id"] if tenant_type is not None else []) + ["provider", "enabled"]),
        (binding, "bind_scope", ["connection_id", "actor_id"]),
        (job, "job_claim", ["status", "available_at"]),
        (job, "job_scope", ["domain_id", "created_at"]),
        (job, "job_coalesce", ["coalesce_key", "status"]),
        (attempt, "attempt_job", ["job_id", "attempt_number"]),
        (entry, "entry_expiry", ["expires_at"]),
    ):
        Index(("ix_notify_" + suffix).upper() if prefix.isupper() else "ix_notify_" + suffix,
              *(obj.c[field] for field in fields))
    return dict(connection=connection, binding=binding, policy=policy, entry=entry,
                job=job, attempt=attempt, audit=audit_event, throttle=throttle)
