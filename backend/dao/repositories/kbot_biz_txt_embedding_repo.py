from typing import Sequence, Dict, Any
from sqlalchemy import select, delete
from backend.core.database.factory import create_session
from backend.dao.entities.kbot_biz_txt_embedding import (
    KbotBizTxtEmbeddingOracle,
    KbotBizTxtEmbeddingPG,
    KbotBizTxtEmbeddingMySQL
)

class KbotBizTxtEmbeddingOracleRepo:
    """Repository for KBOT_BIZ_TXT_EMBEDDING table operations. (Oracle version)"""
    
    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.db_type = "oracle"
        self.model = KbotBizTxtEmbeddingOracle


    async def create(self, embed: KbotBizTxtEmbeddingOracle) -> KbotBizTxtEmbeddingOracle:
        """Create new vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
        return embed

    async def get_by_id(self, embed_id: str) -> KbotBizTxtEmbeddingOracle | None:
        """Get vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            result = await session.execute(
                select(KbotBizTxtEmbeddingOracle).where(KbotBizTxtEmbeddingOracle.embed_id == embed_id)
            )
        return result.scalars().first()

    async def update(self, embed: KbotBizTxtEmbeddingOracle) -> KbotBizTxtEmbeddingOracle:
        """Update vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
            return embed

    async def delete(self, embed_id: str) -> bool:
        """Delete vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            domain = await self.get_by_id(embed_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def delete_by_file_id(self, file_id: int) -> bool:
        """Delete vector embeddings by file ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            stmt = delete(KbotBizTxtEmbeddingOracle).where(KbotBizTxtEmbeddingOracle.file_id == file_id)
            await session.execute(stmt)
            await session.commit()
            return True

class KbotBizTxtEmbeddingPGRepo:
    """Repository for KBOT_BIZ_TXT_EMBEDDING table operations. (PostgreSQL version)"""
    
    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.db_type = "postgresql"
        self.model = KbotBizTxtEmbeddingPG

    async def create(self, embed: KbotBizTxtEmbeddingPG) -> KbotBizTxtEmbeddingPG:
        """Create new vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
        return embed

    async def get_by_id(self, embed_id: str) -> KbotBizTxtEmbeddingPG | None:
        """Get vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            result = await session.execute(
                select(KbotBizTxtEmbeddingPG).where(KbotBizTxtEmbeddingPG.embed_id == embed_id)
            )
        return result.scalars().first()

    async def update(self, embed: KbotBizTxtEmbeddingPG) -> KbotBizTxtEmbeddingPG:
        """Update vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
            return embed

    async def delete(self, embed_id: str) -> bool:
        """Delete vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            domain = await self.get_by_id(embed_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def delete_by_file_id(self, file_id: int) -> bool:
        """Delete vector embeddings by file ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            stmt = delete(KbotBizTxtEmbeddingPG).where(KbotBizTxtEmbeddingPG.file_id == file_id)
            await session.execute(stmt)
            await session.commit()
            return True

class KbotBizTxtEmbeddingMySQLRepo:
    """Repository for KBOT_BIZ_TXT_EMBEDDING table operations. (MySQL version)"""
    
    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.db_type = "mysql"
        self.model = KbotBizTxtEmbeddingMySQL

    async def create(self, embed: KbotBizTxtEmbeddingMySQL) -> KbotBizTxtEmbeddingMySQL:
        """Create new vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
        return embed

    async def get_by_id(self, embed_id: str) -> KbotBizTxtEmbeddingMySQL | None:
        """Get vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            result = await session.execute(
                select(KbotBizTxtEmbeddingMySQL).where(KbotBizTxtEmbeddingMySQL.embed_id == embed_id)
            )
        return result.scalars().first()

    async def update(self, embed: KbotBizTxtEmbeddingMySQL) -> KbotBizTxtEmbeddingMySQL:
        """Update vector embedding record"""
        async with create_session(self.db_type, self.connection_info) as session:
            session.add(embed)
            await session.commit()
            await session.refresh(embed)
            return embed

    async def delete(self, embed_id: str) -> bool:
        """Delete vector embedding record by ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            domain = await self.get_by_id(embed_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def delete_by_file_id(self, file_id: int) -> bool:
        """Delete vector embeddings by file ID"""
        async with create_session(self.db_type, self.connection_info) as session:
            stmt = delete(KbotBizTxtEmbeddingMySQL).where(KbotBizTxtEmbeddingMySQL.file_id == file_id)
            await session.execute(stmt)
            await session.commit()
            return True