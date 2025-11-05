from typing import AsyncIterator, Sequence
from sqlalchemy import select, update, delete
from core.database.meta_oracle import get_session
from dao.entities.kbot_md_agent_conf import KbotMdAgentConf


class KbotMdAgentConfRepository:
    """KBOT_MD_AGENT_CONF表的Repository类"""

    async def get_by_id(self, conf_id: int) -> KbotMdAgentConf | None:
        """get agent config by id. """
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdAgentConf).where(KbotMdAgentConf.conf_id == conf_id)
            )
            return result.scalar_one_or_none()

    async def get_by_agent_id(self, agent_id: int) -> Sequence[KbotMdAgentConf]:
        """get agent config by agent id. """
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdAgentConf).where(KbotMdAgentConf.agent_id == agent_id)
            )
            return result.scalars().all()
        
    async def get_unique_kb_id(self, agent_id: int) -> Sequence[int]:
        """get unique kb id by agent id. """
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdAgentConf.tool_id).where(KbotMdAgentConf.agent_id == agent_id).distinct()
            )
            return result.scalars().all()
        
    async def get_by_agnet_and_kb(self, agent_id: int, kb_id: int) -> KbotMdAgentConf | None:
        """get agent config by agent id and kb id. """
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdAgentConf).where(KbotMdAgentConf.agent_id == agent_id,
                                              KbotMdAgentConf.tool_id == kb_id,
                                              KbotMdAgentConf.search_type == 1)
            )
            return result.scalar_one_or_none()

    async def create(self, conf_data: dict) -> KbotMdAgentConf:
        """create agent config. """
        new_conf = KbotMdAgentConf(**conf_data)
        async with get_session() as session:
            session.add(new_conf)
            await session.commit()
            await session.refresh(new_conf)
        return new_conf
  
    async def update(self, conf_id: int, update_data: dict) -> KbotMdAgentConf | None:
        """update agent config by id. """
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdAgentConf).where(KbotMdAgentConf.conf_id == conf_id)
            )
            conf = result.scalars().first()
            if not conf:
                return None

            for key, value in update_data.items():
                setattr(conf, key, value)

            await session.commit()
            await session.refresh(conf)
            return conf

    async def delete(self, conf_id: int) -> bool:
        """delete agent config by id. """
        async with get_session() as session:
            result = await session.execute(
                delete(KbotMdAgentConf).where(KbotMdAgentConf.conf_id == conf_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def delete_by_agent_id(self, agent_id: int) -> int:
        """delete agent config by agent id. """
        async with get_session() as session:
            result = await session.execute(
                delete(KbotMdAgentConf).where(KbotMdAgentConf.agent_id == agent_id)
            )
            await session.commit()
            return result.rowcount

    async def get_all(self) -> AsyncIterator[KbotMdAgentConf]:
        """get all agent config. """
        async with get_session() as session:
            result = await session.stream(
                select(KbotMdAgentConf).order_by(KbotMdAgentConf.conf_id)
            )
            async for row in result:
                yield row.KbotMdAgentConf