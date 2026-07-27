"""Minimal repository base owned by the model service boundary."""
from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class ModelRepositoryBase(Generic[T]):
    def __init__(self, session: AsyncSession):
        self.session = session
