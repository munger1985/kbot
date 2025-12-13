from datetime import datetime, timezone
from sqlalchemy import select, update, and_
from core.database.meta_oracle import get_session
from core.dictionary import UserTokenStatus
from ..entities import UserToken


class UserTokenRepository:
    """用户Token仓储"""
    
    @staticmethod
    async def get_by_jti(jti: str) -> UserToken | None:
        async with get_session() as session:
            result = await session.execute(
                select(UserToken).where(UserToken.jti == jti)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def create(
        jti: str,
        user_id: int,
        expires_at: datetime,
        device_info: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None
    ) -> UserToken:
        async with get_session() as session:
            token = UserToken(
                jti=jti,
                user_id=user_id,
                device_info=device_info,
                ip_address=ip_address,
                user_agent=user_agent,
                status=UserTokenStatus.ACTIVE,
                expires_at=expires_at
            )
            session.add(token)
            await session.commit()
            await session.refresh(token)
            return token
    
    @staticmethod
    async def revoke(
        jti: str, 
        reason: str | None = None,
        user_id: int | None = None
    ) -> bool:
        conditions = [
            UserToken.jti == jti,
            UserToken.status == UserTokenStatus.ACTIVE
        ]
        
        if user_id:
            conditions.append(UserToken.user_id == user_id)
        
        async with get_session() as session:
            result = await session.execute(
                update(UserToken)
                .where(and_(*conditions))
                .values(
                    status=UserTokenStatus.REVOKED,
                    revoked_reason=reason,
                    revoked_at=datetime.now(timezone.utc)
                )
            )
            await session.commit()
            return result.rowcount > 0
    
    @staticmethod
    async def revoke_all_user_tokens(
        user_id: int, 
        reason: str = "logout_all"
    ) -> int:
        async with get_session() as session:
            result = await session.execute(
                update(UserToken)
                .where(
                    UserToken.user_id == user_id,
                    UserToken.status == UserTokenStatus.ACTIVE
                )
                .values(
                    status=UserTokenStatus.REVOKED,
                    revoked_reason=reason,
                    revoked_at=datetime.now(timezone.utc)
                )
            )
            await session.commit()
            return result.rowcount or 0
    
    @staticmethod
    async def is_valid(jti: str, user_id: int | None = None) -> bool:
        conditions = [
            UserToken.jti == jti,
            UserToken.status == UserTokenStatus.ACTIVE,
            UserToken.expires_at > datetime.now(timezone.utc)
        ]
        
        if user_id:
            conditions.append(UserToken.user_id == user_id)
        
        async with get_session() as session:
            result = await session.execute(
                select(UserToken.id).where(and_(*conditions)).limit(1)
            )
            return result.scalar_one_or_none() is not None