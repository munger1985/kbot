from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import declared_attr

class Base(DeclarativeBase):
    """Base class for all database models."""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()