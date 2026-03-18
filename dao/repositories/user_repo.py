from datetime import datetime, timezone
from sqlalchemy import select, update, delete
from core.database.oracle import get_session
from ..entities import User


class UserRepository:
    """用户仓储"""
    
    @staticmethod
    async def get_by_id(user_id: int) -> User | None:
        async with get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_username(username: str) -> User | None:
        async with get_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_email(email: str) -> User | None:
        async with get_session() as session:
            result = await session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def create(
        username: str,
        email: str,
        hashed_password: str
    ) -> User:
        async with get_session() as session:
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                is_active=True
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    @staticmethod
    async def update_last_login(user_id: int) -> bool:
        async with get_session() as session:
            result = await session.execute(
                update(User)
                .where(User.id == user_id)
                .values(last_login_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return True
        
    @staticmethod
    async def update_password_by_name(username: str, hashed_password: str) -> bool:
        async with get_session() as session:
            result = await session.execute(
                update(User)
                .where(User.username == username)
                .values(hashed_password=hashed_password)
            )
            await session.commit()
            return True
        
    @staticmethod
    async def delete_user_by_name(username: str) -> bool:
        async with get_session() as session:
            result = await session.execute(
                delete(User)
                .where(User.username == username)
            )
            await session.commit()
            return True