"""解析事件的纯内存完整性检查；不是持久接收器，不负责 ACK。"""

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re

from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import canonical_json, fingerprint, nonempty


PARSE_EVENT_SCHEMA = "knowledge.parse_event.v1"


def validate_digest(value: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "摘要必须为小写 SHA-256")


def local_key(*, algorithm_version: str, native_ref: str, kind: str) -> str:
    """同一算法及规范来源位置稳定；不承诺跨解析器稳定。"""
    nonempty(algorithm_version, native_ref, kind)
    return fingerprint({"algorithm_version": algorithm_version, "native_ref": native_ref, "kind": kind})


class ParseEventType(str, Enum):
    RUN_STARTED = "RUN_STARTED"
    PAGE_BATCH = "PAGE_BATCH"
    NODE_BATCH = "NODE_BATCH"
    ASSET_BATCH = "ASSET_BATCH"
    REFERENCE_BATCH = "REFERENCE_BATCH"
    STRUCTURE_FINALIZED = "STRUCTURE_FINALIZED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


OBJECT_EVENTS = frozenset({ParseEventType.PAGE_BATCH, ParseEventType.NODE_BATCH,
                           ParseEventType.ASSET_BATCH, ParseEventType.REFERENCE_BATCH})


@dataclass(frozen=True, slots=True)
class ParseRunIdentity:
    owner_binding_ref: str
    job_ref: str
    parse_run_ref: str
    attempt: int
    fence: int
    source_sha256: str
    profile_hash: str

    def __post_init__(self) -> None:
        nonempty(self.owner_binding_ref, self.job_ref, self.parse_run_ref)
        require(type(self.attempt) is int and type(self.fence) is int
                and self.attempt > 0 and self.fence > 0, "任务代次与栅栏必须为正整数")
        validate_digest(self.source_sha256)
        validate_digest(self.profile_hash)


@dataclass(frozen=True, slots=True)
class ParseArtifact:
    handle_ref: str
    object_key: str
    sha256: str
    mime_type: str
    byte_count: int
    expires_at: int

    def __post_init__(self) -> None:
        nonempty(self.handle_ref, self.object_key, self.mime_type)
        validate_digest(self.sha256)
        require("://" not in self.handle_ref and not self.handle_ref.startswith("/")
                and ".." not in self.handle_ref.split("/"), "制品只能使用受控不透明句柄")
        require(type(self.byte_count) is int and self.byte_count > 0
                and type(self.expires_at) is int and self.expires_at > 0, "制品大小或期限无效")


@dataclass(frozen=True, slots=True)
class ParseEvent:
    run: ParseRunIdentity
    sequence: int
    batch_key: str
    event_type: ParseEventType
    payload_json: str
    payload_hash: str
    dependency_keys: tuple[str, ...] = ()
    schema_version: str = PARSE_EVENT_SCHEMA
    artifact_handles: tuple[ParseArtifact, ...] = ()

    def __post_init__(self) -> None:
        require(self.schema_version == PARSE_EVENT_SCHEMA, "不支持的解析事件合同")
        require(type(self.sequence) is int and self.sequence > 0, "事件序号必须为正整数")
        require(isinstance(self.event_type, ParseEventType), "事件类型无效")
        nonempty(self.batch_key, *self.dependency_keys)
        require(len(set(self.dependency_keys)) == len(self.dependency_keys), "依赖键重复")
        validate_digest(self.payload_hash)
        require(len({a.handle_ref for a in self.artifact_handles}) == len(self.artifact_handles), "制品句柄重复")

    @property
    def event_digest(self) -> str:
        return fingerprint(asdict(self))

    @classmethod
    def create(cls, *, run: ParseRunIdentity, sequence: int, batch_key: str,
               event_type: ParseEventType, payload: dict,
               dependency_keys: tuple[str, ...] = (), artifact_handles: tuple[ParseArtifact, ...] = ()) -> "ParseEvent":
        encoded = canonical_json(payload)
        return cls(run, sequence, batch_key, event_type, encoded,
                   sha256(encoded.encode("utf-8")).hexdigest(), dependency_keys, artifact_handles=artifact_handles)


class ParseStreamValidator:
    """有界验证器；故障不推进状态，跨进程恢复仍由 App 持久批次负责。"""

    def __init__(self, run: ParseRunIdentity, *, max_events: int,
                 max_objects: int, max_payload_bytes: int):
        require(all(type(v) is int and v > 0 for v in (
            max_events, max_objects, max_payload_bytes)), "解析验证预算必须为正整数")
        self.run = run
        self.max_events = max_events
        self.max_objects = max_objects
        self.max_payload_bytes = max_payload_bytes
        self._seen: dict[int, str] = {}
        self._batches: set[str] = set()
        self._objects: set[str] = set()
        self._dependencies: set[str] = set()
        self._counts = {kind.value: 0 for kind in OBJECT_EVENTS}
        self._digest = sha256()
        self._finalized = False
        self._terminal = False
        self._completed = False

    @property
    def ordered_digest(self) -> str:
        return self._digest.hexdigest()

    @property
    def event_count(self) -> int:
        return len(self._seen)

    @property
    def object_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def completed(self) -> bool:
        return self._completed

    def accept(self, event: ParseEvent) -> bool:
        if event.run != self.run:
            raise AlgorithmError(ErrorCode.SCOPE_CLOSURE_BROKEN,
                                 "事件任务、源版本、授权归属或栅栏不匹配", stage="parse_event")
        encoded = event.payload_json.encode("utf-8")
        if len(encoded) > self.max_payload_bytes:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "事件载荷超过预算", stage="parse_event")
        require(sha256(encoded).hexdigest() == event.payload_hash, "事件载荷摘要不匹配", stage="parse_event")
        try:
            payload = json.loads(event.payload_json)
            require(isinstance(payload, dict) and canonical_json(payload) == event.payload_json,
                    "事件载荷必须为规范 JSON 对象", stage="parse_event")
        except (ValueError, TypeError) as exc:
            if isinstance(exc, AlgorithmError):
                raise
            raise AlgorithmError(ErrorCode.INVALID_CONTRACT, "事件 JSON 无效", stage="parse_event") from exc
        digest = event.event_digest
        if event.sequence in self._seen:
            if self._seen[event.sequence] != digest:
                raise AlgorithmError(ErrorCode.EVENT_CONFLICT, "同序号事件内容冲突", stage="parse_event")
            return False
        if event.batch_key in self._batches:
            raise AlgorithmError(ErrorCode.EVENT_CONFLICT, "同批次键不能绑定不同序号", stage="parse_event")
        if self._terminal or event.sequence != len(self._seen) + 1:
            raise AlgorithmError(ErrorCode.EVENT_SEQUENCE_INVALID, "事件乱序或任务已经终止", stage="parse_event")
        if len(self._seen) >= self.max_events:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "事件数超过预算", stage="parse_event")
        first = event.sequence == 1
        require(first == (event.event_type == ParseEventType.RUN_STARTED), "首事件必须且只能启动任务")
        new_keys: set[str] = set()
        if event.event_type in OBJECT_EVENTS:
            require(not self._finalized, "结构冻结后不能追加对象")
            objects = payload.get("objects")
            require(isinstance(objects, list) and bool(objects), "对象批次必须非空")
            for item in objects:
                require(isinstance(item, dict) and isinstance(item.get("local_key"), str), "对象缺少本地身份")
                key = item["local_key"]
                nonempty(key)
                require(key not in new_keys and key not in self._objects, "解析对象本地身份重复")
                new_keys.add(key)
        if len(new_keys) + len(self._objects) > self.max_objects:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "解析对象数超过预算", stage="parse_event")
        require(all(a.object_key in new_keys | self._objects for a in event.artifact_handles),
                "制品必须属于本任务已声明的确切对象")
        if len(event.artifact_handles) > self.max_objects or len(canonical_json(asdict(event)).encode("utf-8")) > self.max_payload_bytes + 4096:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "解析信封元数据超过预算", stage="parse_event")
        dependencies = self._dependencies | set(event.dependency_keys)
        if len(dependencies) > self.max_objects:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "引用依赖数超过预算", stage="parse_event")
        if event.event_type == ParseEventType.STRUCTURE_FINALIZED:
            require(not self._finalized, "结构只能冻结一次")
            require(dependencies <= self._objects, "结构存在未解析的对象依赖")
        if event.event_type == ParseEventType.RUN_COMPLETED:
            counts = payload.get("counts")
            valid = (self._finalized and dependencies <= self._objects
                     and type(payload.get("event_count")) is int
                     and payload.get("event_count") == len(self._seen)
                     and payload.get("ordered_digest") == self.ordered_digest
                     and isinstance(counts, dict) and counts == self._counts
                     and all(type(value) is int for value in counts.values())
                     and isinstance(payload.get("quality_report"), dict)
                     and payload["quality_report"].get("status") in {"COMPLETE", "PARTIAL", "FAILED"})
            if not valid:
                raise AlgorithmError(ErrorCode.MANIFEST_INVALID, "最终清单计数、依赖、摘要或质量报告无效", stage="manifest")
        # 所有检查通过后才推进；本步骤不等于落库或确认消费。
        self._seen[event.sequence] = digest
        self._batches.add(event.batch_key)
        self._objects.update(new_keys)
        self._dependencies = dependencies
        if event.event_type in OBJECT_EVENTS:
            self._counts[event.event_type.value] += len(new_keys)
        self._digest.update(bytes.fromhex(digest))
        if event.event_type == ParseEventType.STRUCTURE_FINALIZED:
            self._finalized = True
        if event.event_type in {ParseEventType.RUN_COMPLETED, ParseEventType.RUN_FAILED}:
            self._terminal = True
        if event.event_type == ParseEventType.RUN_COMPLETED:
            self._completed = True
        return True
