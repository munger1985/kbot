from typing import Sequence, Dict, Any
from sqlalchemy import select, delete
from backend.core.database.factory import create_session
from backend.dao.entities.kbot_biz_img_embedding import (
    KbotBizImgEmbeddingOracle,
    KbotBizImgEmbeddingPG,
    KbotBizImgEmbeddingMySQL
)

class KbotBizImgEmbeddingOracleRepo:
    """Repository for KBOT_BIZ_IMG_EMBEDDING table operations. (Oracle version)"""
    
    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.db_type = "oracle"
        self.model = KbotBizImgEmbeddingOracle

    async def create(self, embed: KbotBizImgEmbeddingOracle) -> KbotBizImgEmbeddingOracle:
        """Create new image vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
        return embed

    async def get_by_id(self, embed_id: str) -> KbotBizImgEmbeddingOracle | None:
        """Get image vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            result = await session.execute(
                select(KbotBizImgEmbeddingOracle).where(KbotBizImgEmbeddingOracle.embed_id == embed_id)
            )
        return result.scalars().first()

    async def update(self, embed: KbotBizImgEmbeddingOracle) -> KbotBizImgEmbeddingOracle:
        """Update image vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
            return embed

    async def delete(self, embed_id: str) -> bool:
        """Delete image vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            domain = await self.get_by_id(embed_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def delete_by_file_id(self, file_id: int) -> bool:
        """Delete image vector embeddings by file ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            stmt = delete(KbotBizImgEmbeddingOracle).where(KbotBizImgEmbeddingOracle.file_id == file_id)
            await session.execute(stmt)
            await session.commit()
            return True

class KbotBizImgEmbeddingPGRepo:
    """Repository for KBOT_BIZ_IMG_EMBEDDING table operations. (PostgreSQL version)"""
    
    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.db_type = "postgresql"
        self.model = KbotBizImgEmbeddingPG

    async def create(self, embed: KbotBizImgEmbeddingPG) -> KbotBizImgEmbeddingPG:
        """Create new image vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
        return embed

    async def get_by_id(self, embed_id: str) -> KbotBizImgEmbeddingPG | None:
        """Get image vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            result = await session.execute(
                select(KbotBizImgEmbeddingPG).where(KbotBizImgEmbeddingPG.embed_id == embed_id)
            )
        return result.scalars().first()

    async def update(self, embed: KbotBizImgEmbeddingPG) -> KbotBizImgEmbeddingPG:
        """Update image vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
            return embed

    async def delete(self, embed_id: str) -> bool:
        """Delete image vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            domain = await self.get_by_id(embed_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def delete_by_file_id(self, file_id: int) -> bool:
        """Delete image vector embeddings by file ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            stmt = delete(KbotBizImgEmbeddingPG).where(KbotBizImgEmbeddingPG.file_id == file_id)
            await session.execute(stmt)
            await session.commit()
            return True

class KbotBizImgEmbeddingMySQLRepo:
    """Repository for KBOT_BIZ_IMG_EMBEDDING table operations. (MySQL version)"""
    
    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.db_type = "mysql"
        self.model = KbotBizImgEmbeddingMySQL

    async def create(self, embed: KbotBizImgEmbeddingMySQL) -> KbotBizImgEmbeddingMySQL:
        """Create new image vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
        return embed

    async def get_by_id(self, embed_id: str) -> KbotBizImgEmbeddingMySQL | None:
        """Get image vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            result = await session.execute(
                select(KbotBizImgEmbeddingMySQL).where(KbotBizImgEmbeddingMySQL.embed_id == embed_id)
            )
        return result.scalars().first()

    async def update(self, embed: KbotBizImgEmbeddingMySQL) -> KbotBizImgEmbeddingMySQL:
        """Update image vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
            return embed

    async def delete(self, embed_id: str) -> bool:
        """Delete image vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            domain = await self.get_by_id(embed_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def delete_by_file_id(self, file_id: int) -> bool:
        """Delete image vector embeddings by file ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            stmt = delete(KbotBizImgEmbeddingMySQL).where(KbotBizImgEmbeddingMySQL.file_id == file_id)
            await session.execute(stmt)
            await session.commit()
            return True