"""运维资料时间在 Oracle 查询边界恢复为 UTC。"""

from datetime import datetime, timezone
import unittest

from sqlalchemy import insert, select, update
from sqlalchemy.dialects import oracle, postgresql
from sqlalchemy.orm import aliased

from aiops_agent.entities import OperationsKnowledgeAssetEntity, OperationsKnowledgeVersionEntity


class OperationsKnowledgeTimestampTests(unittest.TestCase):
    def test_oracle_select_extracts_utc_without_wrapping_filters_or_ordering(self):
        asset = aliased(OperationsKnowledgeAssetEntity, name="asset")
        cutoff = datetime(2026, 11, 24, tzinfo=timezone.utc)
        statement = (select(asset.created_at, asset.updated_at)
                     .where(asset.created_at >= cutoff)
                     .order_by(asset.updated_at.desc()))
        sql = str(statement.compile(dialect=oracle.dialect()))
        self.assertIn("SYS_EXTRACT_UTC(asset.created_at) AS created_at", sql)
        self.assertIn("SYS_EXTRACT_UTC(asset.updated_at) AS updated_at", sql)
        self.assertIn("WHERE asset.created_at >=", sql)
        self.assertIn("ORDER BY asset.updated_at DESC", sql)
        self.assertNotIn("SYS_EXTRACT_UTC", str(statement.compile(dialect=postgresql.dialect())))

    def test_oracle_returning_retains_utc_conversion(self):
        asset = OperationsKnowledgeAssetEntity
        version = OperationsKnowledgeVersionEntity
        for statement, field in (
            (insert(asset).returning(asset.created_at), asset.created_at),
            (update(asset).returning(asset.updated_at), asset.updated_at),
            (insert(version).returning(version.created_at), version.created_at),
        ):
            with self.subTest(field=field):
                sql = str(statement.compile(dialect=oracle.dialect()))
                self.assertIn(f'RETURNING SYS_EXTRACT_UTC("{field.class_.__tablename__}".{field.key}) INTO', sql)
                normalized = field.type.process_result_value(datetime(2026, 11, 24, 1, 43, 10), oracle.dialect())
                self.assertEqual("2026-11-24T01:43:10+00:00", normalized.isoformat())
