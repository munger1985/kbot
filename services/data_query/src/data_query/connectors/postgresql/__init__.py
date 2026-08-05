"""PostgreSQL-only compiler and execution adapter."""

from .compiler import CompiledPostgreSQLQuery, compile_postgresql_query
from .executor import (
    NormalizedQueryResult,
    PostgreSQLExecutionLimits,
    PostgreSQLReadOnlyExecutor,
    QueryResultNormalizationError,
    normalize_rows,
)

__all__ = [
    "CompiledPostgreSQLQuery",
    "compile_postgresql_query",
    "NormalizedQueryResult",
    "PostgreSQLExecutionLimits",
    "PostgreSQLReadOnlyExecutor",
    "QueryResultNormalizationError",
    "normalize_rows",
]
