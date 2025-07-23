from typing import Optional, List, AsyncIterator, Sequence
from sqlalchemy import select, update, delete
from core.database.meta_oracle import get_session
from dao.entities.kbot_md_agent_conf import KBotMdAgentConf


class KBotMdAgentConfRepository:
    """KBOT_MD_AGENT_CONF表的Repository类"""

    async def get_by_id(self, conf_id: int) -> Optional[KBotMdAgentConf]:
        """get agent config by id. """
        async with get_session() as session:
            result = await session.execute(
                select(KBotMdAgentConf).where(KBotMdAgentConf.conf_id == conf_id)
            )
            return result.scalars().first()

    async def get_by_agent_id(self, agent_id: int) -> Sequence[KBotMdAgentConf]:
        """get agent config by agent id. """
        async with get_session() as session:
            result = await session.execute(
                select(KBotMdAgentConf).where(KBotMdAgentConf.agent_id == agent_id)
            )
            return result.scalars().all()

    async def create(self, conf_data: dict) -> KBotMdAgentConf:
        """create agent config. """
        new_conf = KBotMdAgentConf(**conf_data)
        async with get_session() as session:
            session.add(new_conf)
            await session.commit()
            await session.refresh(new_conf)
        return new_conf
  
    async def update(self, conf_id: int, update_data: dict) -> Optional[KBotMdAgentConf]:
        """update agent config by id. """
        async with get_session() as session:
            result = await session.execute(
                select(KBotMdAgentConf).where(KBotMdAgentConf.conf_id == conf_id)
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
                delete(KBotMdAgentConf).where(KBotMdAgentConf.conf_id == conf_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def get_all(self) -> AsyncIterator[KBotMdAgentConf]:
        """get all agent config. """
        async with get_session() as session:
            result = await session.stream(
                select(KBotMdAgentConf).order_by(KBotMdAgentConf.conf_id)
            )
            async for row in result:
                yield row.KBotMdAgentConf