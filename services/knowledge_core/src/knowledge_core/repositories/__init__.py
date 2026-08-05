"""Persistence repositories for the isolated Knowledge Core model."""

from .collection_repo import CollectionBindingRepository, CollectionRepository, IngestionReceiptRepository
from .ingestion_repo import BundleRepository, BundleRevisionDocumentRepository, BundleRevisionRepository, DocumentRepository, DocumentVersionRepository, EvidenceRepository, IngestionJobRepository, ParseViewRepository
from .discovery_repo import DiscoveryRepository
from .relation_repo import RelationRepository
from .visual_asset_repo import VisualAssetRepository
from .model_reference_repo import ModelReferenceRepository
from .collection_purge_repo import CollectionPurgeRepository

__all__ = [
    "CollectionRepository",
    "CollectionBindingRepository",
    "IngestionReceiptRepository",
    "BundleRepository",
    "BundleRevisionRepository",
    "DocumentRepository",
    "DocumentVersionRepository",
    "BundleRevisionDocumentRepository",
    "IngestionJobRepository",
    "ParseViewRepository",
    "EvidenceRepository",
    "DiscoveryRepository",
    "RelationRepository",
    "VisualAssetRepository",
    "ModelReferenceRepository",
    "CollectionPurgeRepository",
]
