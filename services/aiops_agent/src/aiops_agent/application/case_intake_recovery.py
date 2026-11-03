"""案例入库恢复材料：固定脱敏投影和确切 KC 受理定位。"""

import hashlib
from uuid import UUID

from pydantic import Field, model_validator

from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from platform_core.contracts.aiops.types import AIOpsContract, Sha256Digest, UUIDv7


class CaseIntakeRecovery(AIOpsContract):
    schema_version: str = "OPS_CASE_INTAKE_RECOVERY.v1"
    intake_id: UUIDv7
    request_hash: Sha256Digest
    source_fingerprint: Sha256Digest
    collection_id: UUID
    title: str = Field(min_length=1, max_length=256)
    profile: DiagnosisCaseProfile
    body_sha256: Sha256Digest
    bundle_id: UUID | None = None
    bundle_revision_id: UUID | None = None

    @model_validator(mode="after")
    def validate_pair(self):
        if self.schema_version != "OPS_CASE_INTAKE_RECOVERY.v1":
            raise ValueError("案例恢复材料版本无效")
        if (self.bundle_id is None) != (self.bundle_revision_id is None):
            raise ValueError("KC 受理必须有确切 Bundle 和 Revision")
        return self

    def body(self, *, prepared, collection_id):
        """只重放原投影，不因模型或 Prompt 变化重新脱敏。"""
        from aiops_agent.application.case_closure import rejected_closure
        from aiops_agent.application.case_knowledge_projection import case_knowledge_markdown
        from aiops_agent.application.case_semantic_redaction import shared_content_hash
        provenance = self.profile.redaction.semantic_provenance if self.profile.redaction else {}
        if (self.intake_id != prepared.intake_id
            or self.request_hash != prepared.request_hash
            or self.source_fingerprint != prepared.source_fingerprint
            or self.collection_id != collection_id
            or self.profile.source != prepared.profile.source
            or provenance.get("status") != "SCANNED"
            or provenance.get("output_hash") != shared_content_hash(self.title, self.profile)):
            raise rejected_closure()
        body = case_knowledge_markdown(title=self.title, profile=self.profile)
        if hashlib.sha256(body).hexdigest() != self.body_sha256:
            raise rejected_closure()
        return body
