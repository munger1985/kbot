from sqlalchemy import select, update, and_
from platform_core.database.oracle import get_session
from ..entities import Service


class ServiceRepository:
    """服务仓储"""
    
    @staticmethod
    async def get_by_id(service_id: int) -> Service | None:
        async with get_session() as session:
            result = await session.execute(
                select(Service).where(Service.id == service_id)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_code(service_code: str) -> Service | None:
        async with get_session() as session:
            result = await session.execute(
                select(Service).where(Service.service_code == service_code)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def create(
        service_code: str,
        name: str,
        service_type: str = "internal",
        description: str | None = None,
        owner: str | None = None,
        contact_email: str | None = None
    ) -> Service:
        async with get_session() as session:
            service = Service(
                service_code=service_code,
                name=name,
                service_type=service_type,
                description=description,
                owner=owner,
                contact_email=contact_email,
                is_active=True
            )
            session.add(service)
            await session.commit()
            await session.refresh(service)
            return service
    
    @staticmethod
    async def list_active() -> list[Service]:
        async with get_session() as session:
            result = await session.execute(
                select(Service)
                .where(Service.is_active == True)
                .order_by(Service.created_at.desc())
            )
            return list(result.scalars().all())