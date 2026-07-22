"""Reusable persistence primitives shared by independently deployed services."""

from .orm import BaseEntity, OracleJSON, UniversalVector, VectorField

__all__ = ["BaseEntity", "OracleJSON", "UniversalVector", "VectorField"]
