"""Runtime 对浏览器的知识结果投影；不输出 App 存储句柄、授权快照或模型内部载荷。"""

from knowledge_algorithms.contracts.runtime import KnowledgeCompositionArtifact
from knowledge_algorithms.contracts.wire import decode, encode


def public_composition(value):
    composition = decode(KnowledgeCompositionArtifact, value)
    evidence = composition.evidence
    return {
        "retrieval_ref": composition.retrieval_ref, "status": evidence.status,
        "reason": evidence.reason, "mode": evidence.mode.value, "continuation": evidence.continuation,
        "primary_hits": [{"key": hit.evidence_key.stable_key, "rank": hit.rank}
                         for hit in evidence.primary_hits],
        "items": [{"key": item.key.stable_key, "kind": item.kind.value, "mime_type": item.mime_type,
                   "byte_count": item.byte_count, "locators": encode(item.locators), "text": None}
                  for item in evidence.evidence],
        "linked": [{"anchor_key": group.anchor_key.stable_key,
                    "linked_keys": [key.stable_key for key in group.linked_keys],
                    "manifest_status": group.manifest_status, "continuation": group.continuation,
                    "incomplete_reason": group.incomplete_reason,
                    "relations": [{"relation_type": edge.relation_type.value,
                                   "source_key": edge.source_key.stable_key,
                                   "target_key": edge.target_key.stable_key} for edge in group.relations]}
                   for group in evidence.groups],
        "channels": encode(evidence.channel_status), "model_delivery": composition.delivery.status,
        "delivered_count": len(composition.delivery.prepared) if composition.delivery.status == "SENT" else 0,
        "coverage_status": composition.answer.coverage_status if composition.answer else "INSUFFICIENT",
        "excluded": [{"key": key.stable_key, "reason": reason} for key, reason in composition.excluded],
        "selection": {"selected_keys": [key.stable_key for key in composition.context_selection.selected_evidence_keys],
                      "cost": encode(composition.context_selection.cost)},
    }
