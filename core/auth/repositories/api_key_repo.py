import json
from datetime import datetime, timezone
from sqlalchemy import select, update, and_
from sqlalchemy.orm import joinedload
from core.database.meta_oracle import get_session
from core.dictionary import APIKeyStatus
from ..entities import APIKey



class APIKeyRepository:
    """API Key仓储"""
    
    @staticmethod
    async def get_by_key_id(key_id: str) -> APIKey | None:
        async with get_session() as session:
            result = await session.execute(
                select(APIKey)
                .where(APIKey.key_id == key_id)
                .options(joinedload(APIKey.service))  # 预先加载关联的服务信息
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def create(
        key_id: str,
        hashed_key: str,
        key_prefix: str,
        name: str,
        service_id: int,
        scopes: list[str] | None = None,
        expires_at: datetime | None = None,
        allowed_ips: list[str] | None = None,
        rate_limit: int = 0,
        created_by: str | None = None
    ) -> APIKey:
        async with get_session() as session:
            api_key = APIKey(
                key_id=key_id,
                hashed_key=hashed_key,
                key_prefix=key_prefix,
                name=name,
                service_id=service_id,
                scopes=json.dumps(scopes or []),
                status=APIKeyStatus.ACTIVE,
                expires_at=expires_at,
                allowed_ips=json.dumps(allowed_ips or []),
                rate_limit=rate_limit,
                created_by=created_by
            )
            session.add(api_key)
            await session.commit()
            await session.refresh(api_key)
            return api_key
    
    @staticmethod
    async def update_usage(key_id: str, client_ip: str | None = None) -> bool:
        async with get_session() as session:
            update_values = {
                "last_used_at": datetime.now(timezone.utc),
                "usage_count": APIKey.usage_count + 1
            }
            
            result = await session.execute(
                update(APIKey)
                .where(APIKey.key_id == key_id)
                .values(**update_values)
            )
            await session.commit()
            return result.rowcount > 0
    
    @staticmethod
    async def revoke(
        key_id: str, 
        reason: str | None = None,
        service_id: int | None = None
    ) -> bool:
        conditions = [
            APIKey.key_id == key_id,
            APIKey.status == APIKeyStatus.ACTIVE
        ]
        
        if service_id:
            conditions.append(APIKey.service_id == service_id)
        
        async with get_session() as session:
            result = await session.execute(
                update(APIKey)
                .where(and_(*conditions))
                .values(
                    status=APIKeyStatus.REVOKED,
                    revoked_reason=reason,
                    revoked_at=datetime.now(timezone.utc)
                )
            )
            await session.commit()
            return result.rowcount > 0
    
    @staticmethod
    async def list_by_service(
        service_id: int, 
        active_only: bool = True
    ) -> list[APIKey]:
        async with get_session() as session:
            query = select(APIKey).where(
                APIKey.service_id == service_id
            )
            
            if active_only:
                query = query.where(
                    APIKey.status == APIKeyStatus.ACTIVE
                )
            
            query = query.order_by(APIKey.created_at.desc())
            query = query.options(joinedload(APIKey.service))  # 预先加载关联的服务信息
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    @staticmethod
    async def mark_expired(key_id: str) -> bool:
        async with get_session() as session:
            result = await session.execute(
                update(APIKey)
                .where(
                    APIKey.key_id == key_id,
                    APIKey.status == APIKeyStatus.ACTIVE
                )
                .values(status=APIKeyStatus.EXPIRED)
            )
            await session.commit()
            return result.rowcount > 0