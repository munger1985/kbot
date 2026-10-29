"""按固定原生字段比较来源，不从Target、地址、版本或数据库族标识推断同一运行实例。"""

from datetime import datetime, timezone
from uuid import UUID

from aiops_agent.contracts.physical_identity import (
    PhysicalIdentityEdge, PhysicalIdentityGraph, PhysicalIdentityNode,
)
from aiops_agent.playbooks import canonical_hash


IDENTITY_METRIC_CODE = "db.identity.info"
IDENTITY_TEMPLATE_ID = "prom.database.identity.info"
IDENTITY_TEMPLATE_VERSION = "1"
IDENTITY_FIELDS = {
    "POSTGRESQL": ("system_identifier", "database_oid", "instance_started_at"),
    "MYSQL": ("server_uuid",),
    "ORACLE": ("dbid", "instance_number", "container_id", "instance_started_at"),
}
ANCHORS = {"POSTGRESQL": "system_identifier", "MYSQL": "server_uuid", "ORACLE": "dbid"}
COLLECTOR_LABELS = ("target_key", "job", "instance", "cluster", "prometheus", "replica", "namespace", "pod")


def canonical_native_identity(engine, identity):
    """缺失字段不补值；大整数身份按十进制字符串处理，避免浮点舍入。"""
    output = {}
    for key in IDENTITY_FIELDS.get(engine, ()):
        raw = identity.get(key)
        if raw is None:
            continue
        if not isinstance(raw, str) or not raw or len(raw) > 128 or any(ord(char) < 32 for char in raw):
            return (), ("INVALID_NATIVE_IDENTITY",)
        if key == "server_uuid":
            try:
                parsed = UUID(raw)
                if parsed.int == 0:
                    raise ValueError
                raw = str(parsed)
            except ValueError:
                return (), ("INVALID_NATIVE_IDENTITY",)
        elif key == "instance_started_at":
            try:
                stamp = datetime.fromisoformat(raw)
                raw = stamp.astimezone(timezone.utc).isoformat() if stamp.tzinfo is not None else stamp.isoformat()
            except (ValueError, OverflowError):
                return (), ("INVALID_NATIVE_IDENTITY",)
        else:
            if not raw.isascii() or not raw.isdecimal() or int(raw) < (0 if key == "container_id" else 1):
                return (), ("INVALID_NATIVE_IDENTITY",)
            raw = str(int(raw))
        output[key] = raw
    flags = []
    if ANCHORS.get(engine) not in output:
        flags.append("NATIVE_DATABASE_IDENTITY_MISSING")
    if "instance_started_at" not in output:
        flags.append("BOOT_IDENTITY_UNKNOWN")
    elif datetime.fromisoformat(output["instance_started_at"]).tzinfo is None:
        flags.append("BOOT_TIMEZONE_UNKNOWN")
    return tuple(sorted(output.items())), tuple(flags)


def _engine(identity):
    for engine, anchor in ANCHORS.items():
        if anchor in identity:
            return engine
    return None


def build_physical_identity_graph(evidence):
    nodes, flags = [], []
    for fact in sorted(evidence, key=lambda item: item.evidence_ref):
        if fact.trust_level != "SOURCE_VERIFIED":
            continue
        provenance = fact.provenance
        if (provenance is not None and provenance.physical_identity
            and (fact.tool_id.startswith("db.") or fact.tool_id == "evidence.sql_plan.controlled_estimate")):
            identity = dict(provenance.physical_identity)
            engine = _engine(identity)
            if engine is not None:
                native, quality = canonical_native_identity(engine, identity)
                stamp = provenance.collected_end
                # 只有同连接生产者的采集区间可提供当前时点；历史窗口不补造当前身份。
                if stamp is not None:
                    nodes.append(PhysicalIdentityNode(
                        node_id=canonical_hash((fact.evidence_ref, "direct")), evidence_ref=fact.evidence_ref,
                        input_ref=fact.evidence_ref,
                        target_id=provenance.target_id, origin="DIRECT_DATABASE", database_type=engine,
                        source_id=fact.source_id, source_version=provenance.source_version,
                        native_identity=native, observed_times=(stamp,),
                        quality_flags=(*quality, "BACKEND_AFFINITY_NOT_PROVEN"),
                    ))
        for witness in fact.source_identity_witnesses:
            nodes.append(PhysicalIdentityNode(
                node_id=canonical_hash((fact.evidence_ref, witness.input_ref)), evidence_ref=fact.evidence_ref,
                input_ref=witness.input_ref, target_id=str(witness.target_id), origin="MONITORING_REPORT",
                database_type=witness.database_type, native_identity=witness.native_identity,
                source_id=str(witness.source_id), source_version=str(witness.source_version),
                binding_id=str(witness.binding_id),
                observed_times=witness.sample_times, collector_scope=witness.collector_scope,
                quality_flags=witness.quality_flags,
            ))
    if len(nodes) > 64:
        flags.append("IDENTITY_NODE_BUDGET_EXCEEDED")
        # 两类来源各保留一部分，避免大量直连观察挤掉全部监控节点。
        nodes = [node for origin in ("DIRECT_DATABASE", "MONITORING_REPORT")
                 for node in [item for item in nodes if item.origin == origin][:32]]
    direct = [node for node in nodes if node.origin == "DIRECT_DATABASE"]
    monitors = [node for node in nodes if node.origin == "MONITORING_REPORT"]
    edges = []
    for left in direct:
        for right in monitors:
            if left.target_id != right.target_id:
                continue
            if len(edges) >= 128:
                flags.append("IDENTITY_EDGE_BUDGET_EXCEEDED")
                break
            delta = min((abs((a - b).total_seconds()) for a in left.observed_times for b in right.observed_times), default=None)
            reasons = ["SOURCE_REPORTED_IDENTITY_ONLY", "OTHER_METRIC_SERIES_NOT_ASSIGNED", "WINDOW_CONTINUITY_NOT_PROVEN"]
            if "SAME_QUERY_METRIC_IDENTITY_REPORTED" in right.quality_flags:
                reasons[1] = "ONLY_THIS_NATIVE_QUERY_METRIC_REPORTED"
            matched, conflicting = (), ()
            relation = "UNKNOWN"
            bad = {"INVALID_NATIVE_IDENTITY", "NATIVE_DATABASE_IDENTITY_MISSING", "IDENTITY_SAMPLE_INVALID",
                   "IDENTITY_PROTOCOL_UNVERIFIED", "TRUNCATED", "IDENTITY_SAMPLE_BUDGET_EXCEEDED",
                   "IDENTITY_SCOPE_MISSING", "IDENTITY_SCOPE_MISMATCH"}
            conflicting_witnesses = any(
                other.node_id != right.node_id and other.target_id == right.target_id
                and other.database_type == right.database_type and other.collector_scope == right.collector_scope
                and other.input_ref.split("#", 1)[0] == right.input_ref.split("#", 1)[0]
                and dict(other.native_identity).get(ANCHORS[right.database_type]) != dict(right.native_identity).get(ANCHORS[right.database_type])
                for other in monitors
            )
            if delta is None or delta > 120:
                reasons.append("NO_NEARBY_IDENTITY_SAMPLE")
            elif left.database_type != right.database_type:
                reasons.append("DATABASE_ENGINE_CONFLICT")
                relation = "CONFLICT"
            elif bad.intersection((*left.quality_flags, *right.quality_flags)):
                reasons.append("IDENTITY_WITNESS_INCOMPLETE")
            elif conflicting_witnesses:
                reasons.append("COLLECTOR_IDENTITY_CHANGED_OR_AMBIGUOUS")
            else:
                a, b = dict(left.native_identity), dict(right.native_identity)
                common = set(a) & set(b)
                # Oracle原生启动时刻无时区，不能跨来源比较时区未知的时钟。
                if "BOOT_TIMEZONE_UNKNOWN" in (*left.quality_flags, *right.quality_flags):
                    common.discard("instance_started_at")
                matched = tuple(sorted(key for key in common if a[key] == b[key]))
                conflicting = tuple(sorted(key for key in common if a[key] != b[key]))
                anchor = ANCHORS[left.database_type]
                if conflicting:
                    relation = "CONFLICT"
                    reasons.append("NATIVE_IDENTITY_FIELDS_DIFFER")
                elif anchor in matched:
                    relation = "REPORTED_SERVER_MATCH" if left.database_type == "MYSQL" else "REPORTED_DATABASE_FAMILY_MATCH"
                    reasons.append("BOOT_GENERATION_NOT_PROVEN" if left.database_type == "MYSQL" else "DATABASE_FAMILY_IS_NOT_RUNTIME_NODE")
                else:
                    reasons.append("NATIVE_DATABASE_IDENTITY_MISSING")
            edges.append(PhysicalIdentityEdge(
                database_node=left.node_id, monitoring_node=right.node_id, relation=relation,
                matched_fields=matched, conflicting_fields=conflicting, reasons=tuple(reasons),
                nearest_sample_seconds=delta,
            ))
    return PhysicalIdentityGraph(nodes=tuple(nodes), edges=tuple(edges), quality_flags=tuple(dict.fromkeys(flags)))


def replay_physical_identity_graph(graph, evidence):
    if graph != build_physical_identity_graph(evidence):
        raise ValueError("跨源身份图与原始身份观察不一致")


def physical_identity_model_context(evidence):
    """充分性评估、重规划和回答共用同一确定性来源关系与解释边界。"""
    return {
        "physical_identity_graph": build_physical_identity_graph(evidence).model_dump(mode="json"),
        "identity_interpretation_policy": {
            "source_report_is_not_physical_attestation": True,
            "database_family_is_not_runtime_instance": True,
            "identity_metric_does_not_assign_other_series": True,
            "identity_match_is_not_window_continuity_or_sql_causality": True,
            "client_connection_identity_does_not_prove_backend_affinity": True,
        },
    }
