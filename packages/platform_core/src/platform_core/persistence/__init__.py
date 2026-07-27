"""Reusable persistence primitives shared by independently deployed services."""

from .orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UniversalVector,
    UUIDv7Type,
    VectorField,
)

__all__ = [
    "BaseEntity",
    "OracleNativeJSON",
    "UniversalTimestamp",
    "UniversalVector",
    "UUIDv7Type",
    "VectorField",
]
