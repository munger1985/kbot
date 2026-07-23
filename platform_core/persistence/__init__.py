"""Reusable persistence primitives shared by independently deployed services."""

from .orm import (
    BaseEntity,
    OracleJSON,
    UniversalTimestamp,
    UniversalVector,
    UUIDv7Type,
    VectorField,
)

__all__ = [
    "BaseEntity",
    "OracleJSON",
    "UniversalTimestamp",
    "UniversalVector",
    "UUIDv7Type",
    "VectorField",
]
