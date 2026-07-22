from sqlalchemy.ext.asyncio import AsyncSession
from typing import TypeVar, Generic

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """
    Repository Base Class
    """
    def __init__(self, session: AsyncSession):
        self.session = session