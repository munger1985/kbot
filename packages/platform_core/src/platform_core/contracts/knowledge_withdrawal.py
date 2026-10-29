"""知识版本撤回请求与接收方持久化回执。"""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict


class KnowledgeWithdrawalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID


class KnowledgeWithdrawalReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    withdrawal_id: UUID
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    status: Literal["WITHDRAWN"]
    withdrawn_at: AwareDatetime
