from typing import Sequence, Optional
from sqlalchemy import select, delete
from backend.dao.entities.kbot_md_kb_chunks import KbotMdKbChunks, ChunkType
from backend.core.database.oracle import get_session

class KbotMdKbChunksRepository:
    """Repository for KBOT_MD_KB_CHUNKS table operations."""
    
    async def create(self, chunk: KbotMdKbChunks) -> KbotMdKbChunks:
        """Create a new knowledge base chunk record."""
        async with get_session() as session:
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            return chunk
    
    async def get_by_id(self, chunk_id: str) -> Optional[KbotMdKbChunks]:
        """Get knowledge base chunk by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbChunks).where(KbotMdKbChunks.chunk_id == chunk_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdKbChunks]:
        """Get all knowledge base chunk records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdKbChunks))
            return result.scalars().all()
    
    async def update(self, chunk: KbotMdKbChunks) -> KbotMdKbChunks:
        """Update a knowledge base chunk record."""
        async with get_session() as session:
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            return chunk
    
    async def delete(self, chunk_id: str) -> bool:
        """Delete a knowledge base chunk record by ID."""
        async with get_session() as session:
            chunk = await self.get_by_id(chunk_id)
            if not chunk:
                return False
            await session.delete(chunk)
            await session.commit()
            return True
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdKbChunks]:
        """Get knowledge base chunks by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbChunks).where(KbotMdKbChunks.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_kb_id(self, kb_id: int) -> Sequence[KbotMdKbChunks]:
        """Get knowledge base chunks by knowledge base ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbChunks).where(KbotMdKbChunks.kb_id == kb_id)
            )
            return result.scalars().all()
    
    async def get_by_batch_id(self, batch_id: int) -> Sequence[KbotMdKbChunks]:
        """Get knowledge base chunks by batch ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbChunks).where(KbotMdKbChunks.batch_id == batch_id)
            )
            return result.scalars().all()
    
    async def get_by_file_id(self, file_id: int) -> Sequence[KbotMdKbChunks]:
        """Get knowledge base chunks by file ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbChunks).where(KbotMdKbChunks.file_id == file_id)
            )
            return result.scalars().all()
    
    async def get_by_chunk_type(self, chunk_type: ChunkType) -> Sequence[KbotMdKbChunks]:
        """Get knowledge base chunks by chunk type."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbChunks).where(KbotMdKbChunks.chunk_type == chunk_type.value)
            )
            return result.scalars().all()