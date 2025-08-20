from sqlalchemy import select
from dao.entities.kbot_md_api_security import KbotMdApiSecurity
from core.database.meta_oracle import get_session


class KbotMdApiSecurityRepository:
    """Repository for KBOT_MD_API_SECURITY table operations."""
        
    async def get_hashed_secret(self, accessor: str, hashed_secret: str) -> dict[str, str] | None:
        """Verify the access pass."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdApiSecurity.hashed_secret,
                        KbotMdApiSecurity.accessor_type)
                .where(KbotMdApiSecurity.accessor == accessor)
            )
            row = result.first()
            if row is None:
                return None
            return {"hashed_secret": row[0], "accessor_type": row[1]}
        
    async def create(self, security: KbotMdApiSecurity) -> bool:
        """Create a new security."""
        async with get_session() as session:
            session.add(security)
            await session.commit()
            return True
