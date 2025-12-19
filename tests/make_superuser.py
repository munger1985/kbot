#!/usr/bin/env python3
import asyncio
from sqlalchemy import update
from core.database.meta_oracle import get_session
from core.auth.entities.user import User

async def make_user_superuser(username: str):
    """将用户提升为超级用户"""
    async with get_session() as session:
        # 查找用户
        from sqlalchemy import select
        result = await session.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            print(f"用户 {username} 不存在")
            return False
        
        # 更新为超级用户
        result = await session.execute(
            update(User)
            .where(User.id == user.id)
            .values(is_superuser=True)
        )
        await session.commit()
        
        if result.rowcount > 0:
            print(f"用户 {username} 已成功提升为超级用户")
            return True
        else:
            print(f"提升用户 {username} 为超级用户失败")
            return False

if __name__ == "__main__":
    # 将您的用户名提升为超级用户
    username = "chris"  # 替换为您的用户名
    asyncio.run(make_user_superuser(username))