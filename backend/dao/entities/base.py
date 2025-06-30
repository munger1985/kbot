from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import declared_attr

class Base(DeclarativeBase):
    """Base class for all database models."""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Convert CamelCase to SNAKE_CASE"""
        name = cls.__name__
        return (
            ''.join(['_' + c.lower() if c.isupper() else c for c in name])
            .lstrip('_')
            .upper()
        )