"""Reusable persistence primitives shared by independently deployed services."""

from .orm import (
    BaseEntity,
    OracleJSON,
    UniversalVector,
    UUIDv7Type,
    VectorField,
)

__all__ = [
    "BaseEntity",
    "OracleJSON",
    "UniversalVector",
    "UUIDv7Type",
    "VectorField",
]
