"""Knowledge Core 领域标识映射测试。"""

import unittest

from knowledge_core.entities import (
    KcBundleEntity,
    KcBundleRevisionDocumentEntity,
    KcBundleRevisionEntity,
    KcCollectionBindingEntity,
    KcCollectionEntity,
    KcDiscoveryObjectEntity,
    KcDocumentEntity,
    KcDocumentVersionEntity,
    KcEvidenceEntity,
    KcIngestionJobEntity,
    KcIngestionReceiptEntity,
    KcParseViewEntity,
    KcRelationEntity,
)
from platform_core.persistence import UUIDv7Type


class KnowledgeCoreUuidIdentityTest(unittest.TestCase):
    def test_all_resource_ids_use_uuidv7_database_type(self) -> None:
        expected_columns = {
            KcCollectionEntity: ("collection_id",),
            KcCollectionBindingEntity: (
                "binding_id",
                "collection_id",
                "consumer_id",
            ),
            KcIngestionReceiptEntity: (
                "ingestion_receipt_id",
                "collection_id",
                "bundle_id",
                "bundle_revision_id",
            ),
            KcBundleEntity: (
                "bundle_id",
                "collection_id",
                "current_revision_id",
            ),
            KcBundleRevisionEntity: (
                "bundle_revision_id",
                "collection_id",
                "bundle_id",
            ),
            KcDocumentEntity: (
                "document_id",
                "collection_id",
                "bundle_id",
            ),
            KcDocumentVersionEntity: (
                "document_version_id",
                "collection_id",
                "bundle_id",
                "document_id",
            ),
            KcBundleRevisionDocumentEntity: (
                "bundle_revision_document_id",
                "collection_id",
                "bundle_revision_id",
                "document_id",
                "document_version_id",
            ),
            KcParseViewEntity: (
                "parse_view_id",
                "collection_id",
                "document_version_id",
            ),
            KcIngestionJobEntity: (
                "ingestion_job_id",
                "collection_id",
                "bundle_revision_id",
                "document_version_id",
                "parse_view_id",
            ),
            KcEvidenceEntity: (
                "evidence_id",
                "collection_id",
                "bundle_revision_id",
                "bundle_revision_document_id",
                "document_id",
                "document_version_id",
                "parse_view_id",
            ),
            KcDiscoveryObjectEntity: (
                "discovery_object_id",
                "collection_id",
                "bundle_id",
                "bundle_revision_id",
                "bundle_revision_document_id",
                "document_id",
                "document_version_id",
            ),
            KcRelationEntity: (
                "relation_id",
                "collection_id",
                "bundle_id",
                "bundle_revision_id",
                "subject_id",
                "object_id",
            ),
        }
        for entity, names in expected_columns.items():
            for name in names:
                with self.subTest(entity=entity.__name__, column=name):
                    self.assertIsInstance(
                        entity.__table__.c[name].type,
                        UUIDv7Type,
                    )

    def test_resource_primary_keys_have_application_defaults(self) -> None:
        primary_keys = (
            (KcCollectionEntity, "collection_id"),
            (KcCollectionBindingEntity, "binding_id"),
            (KcIngestionReceiptEntity, "ingestion_receipt_id"),
            (KcBundleEntity, "bundle_id"),
            (KcBundleRevisionEntity, "bundle_revision_id"),
            (KcDocumentEntity, "document_id"),
            (KcDocumentVersionEntity, "document_version_id"),
            (
                KcBundleRevisionDocumentEntity,
                "bundle_revision_document_id",
            ),
            (KcParseViewEntity, "parse_view_id"),
            (KcIngestionJobEntity, "ingestion_job_id"),
            (KcEvidenceEntity, "evidence_id"),
            (KcDiscoveryObjectEntity, "discovery_object_id"),
            (KcRelationEntity, "relation_id"),
        )
        for entity, name in primary_keys:
            with self.subTest(entity=entity.__name__):
                column = entity.__table__.c[name]
                self.assertTrue(column.primary_key)
                self.assertIsNotNone(column.default)


if __name__ == "__main__":
    unittest.main()
