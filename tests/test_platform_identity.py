"""UUIDv7 生成与数据库类型测试。"""

import unittest
from uuid import UUID

from sqlalchemy.dialects import oracle, postgresql, sqlite

from platform_core.identity import uuid7
from platform_core.persistence import UUIDv7Type


class UUIDv7Test(unittest.TestCase):
    def test_generated_values_are_version_7_and_monotonic(self) -> None:
        values = [uuid7() for _ in range(100)]
        self.assertTrue(all(value.version == 7 for value in values))
        self.assertTrue(all(value.variant == "specified in RFC 4122" for value in values))
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(values), len(set(values)))

    def test_database_type_round_trips_uuid(self) -> None:
        value = uuid7()
        field = UUIDv7Type()
        for dialect in (oracle.dialect(), sqlite.dialect()):
            stored = field.process_bind_param(value, dialect)
            self.assertEqual(value.bytes, stored)
            self.assertEqual(value, field.process_result_value(stored, dialect))

        pg_dialect = postgresql.dialect()
        self.assertEqual(value, field.process_bind_param(str(value), pg_dialect))
        self.assertEqual(value, field.process_result_value(value, pg_dialect))

    def test_generated_value_is_canonical_uuid(self) -> None:
        value = uuid7()
        self.assertEqual(value, UUID(str(value)))


if __name__ == "__main__":
    unittest.main()
