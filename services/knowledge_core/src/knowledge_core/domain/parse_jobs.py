"""解析计算任务合同；事件存储只服务本次计算，不是公共知识库。"""

from dataclasses import dataclass

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.contracts import ParseProfile, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseRunIdentity


@dataclass(frozen=True, slots=True)
class ComputeAccess:
    owner: str
    subject_ref: str
    scope_key: str

    def __post_init__(self):
        require(all(isinstance(value, str) and 0 < len(value) <= 512
                    for value in (self.owner, self.subject_ref, self.scope_key)), "计算委托身份无效")
        require(self.owner in {"knowledge_retrieval", "aiops"} and self.subject_ref and self.scope_key,
                "未登记的计算委托范围")


@dataclass(frozen=True, slots=True)
class ParseJob:
    access: ComputeAccess
    run: ParseRunIdentity
    source: SourceDescriptor
    profile: ParseProfile
    expires_at: int

    def __post_init__(self):
        require(type(self.expires_at) is int and self.expires_at > 0, "计算期限无效")
        require(self.run.owner_binding_ref == self.source.owner_binding_ref == self.access.owner
                and self.run.source_sha256 == self.source.sha256 and self.run.profile_hash == self.profile.profile_hash,
                "计算任务的归属、源或配置摘要不一致")

    @property
    def computation_ref(self):
        return fingerprint(encode((self.access, self.run, self.source, self.profile)))
