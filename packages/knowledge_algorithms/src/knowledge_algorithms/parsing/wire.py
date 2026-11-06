"""唯一的新解析线信封；显式展平任务身份，不接受旧嵌套 run 形状。"""

import json

from ..contracts.errors import require
from ..contracts.wire import decode, encode
from .events import PARSE_EVENT_SCHEMA, ParseArtifact, ParseEvent, ParseEventType, ParseRunIdentity


def event_to_wire(event: ParseEvent) -> dict:
    return {"schema_version": PARSE_EVENT_SCHEMA, **encode(event.run),
            "sequence": event.sequence, "batch_key": event.batch_key, "event_type": event.event_type.value,
            "payload_hash": event.payload_hash, "dependency_keys": list(event.dependency_keys),
            "payload": json.loads(event.payload_json), "artifact_handles": encode(event.artifact_handles)}


def event_from_wire(value: dict) -> ParseEvent:
    run_fields = {"owner_binding_ref", "job_ref", "parse_run_ref", "attempt", "fence", "source_sha256", "profile_hash"}
    fields = run_fields | {"schema_version", "sequence", "batch_key", "event_type", "payload_hash",
                           "dependency_keys", "payload", "artifact_handles"}
    require(type(value) is dict and set(value) == fields and value["schema_version"] == PARSE_EVENT_SCHEMA,
            "解析线信封字段或版本无效")
    require(type(value["payload"]) is dict, "解析载荷必须为有界对象")
    run = decode(ParseRunIdentity, {key: value[key] for key in run_fields})
    result = ParseEvent.create(run=run, sequence=decode(int, value["sequence"]),
        batch_key=decode(str, value["batch_key"]), event_type=decode(ParseEventType, value["event_type"]),
        payload=value["payload"], dependency_keys=decode(tuple[str, ...], value["dependency_keys"]),
        artifact_handles=decode(tuple[ParseArtifact, ...], value["artifact_handles"]))
    require(result.payload_hash == value["payload_hash"], "解析线载荷摘要不一致")
    return result
