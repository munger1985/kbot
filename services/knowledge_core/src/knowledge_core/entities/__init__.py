"""Knowledge Core ORM entities.

These entities map only to KBOT_KC_* tables and must not import V1 KB/File/Chunk
entities. They are exposed through this subpackage to keep the two models isolated.
"""

from .collection import KcCollectionEntity, KcCollectionBindingEntity, KcIngestionReceiptEntity
from .ingestion import (
    KcBundleEntity, KcBundleRevisionDocumentEntity, KcBundleRevisionEntity,
    KcDocumentEntity, KcDocumentVersionEntity, KcEvidenceEntity,
    KcIngestionJobEntity, KcParseViewEntity, KcVisualAssetEntity,
)
from .discovery import KcDiscoveryObjectEntity
from .relation import KcRelationEntity

__all__ = [
    "KcCollectionEntity",
    "KcCollectionBindingEntity",
    "KcIngestionReceiptEntity",
    "KcBundleEntity",
    "KcBundleRevisionEntity",
    "KcBundleRevisionDocumentEntity",
    "KcDocumentEntity",
    "KcDocumentVersionEntity",
    "KcIngestionJobEntity",
    "KcParseViewEntity",
    "KcEvidenceEntity",
    "KcVisualAssetEntity",
    "KcDiscoveryObjectEntity",
    "KcRelationEntity",
]
