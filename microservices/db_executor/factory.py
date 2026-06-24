from typing import Any, Type
from .drivers.pg_driver import PgDriver
from .drivers.oracle_driver import OracleDriver
from .drivers.mysql_driver import MySqlDriver
from .schemas.db_config import PGConfig, MySQLConfig, OracleConfig
from core.dictionary import DbType

class DriverFactory:
    _driver_registry: dict[str, tuple[Type, Type]] = {
        DbType.POSTGRESQL.value: (PgDriver, PGConfig),
        DbType.MYSQL.value: (MySqlDriver, MySQLConfig),
        DbType.ORACLE.value: (OracleDriver, OracleConfig),
    }

    @classmethod
    def get_driver(cls, db_type: str, config_dict: dict[str, Any]):
        target = cls._driver_registry.get(db_type.lower())
        if not target:
            raise ValueError(f"不支持的数据库类型: {db_type}")
        
        driver_class, config_model = target
        validated_config = config_model(**config_dict)
        return driver_class(validated_config)