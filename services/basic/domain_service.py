from loguru import logger
from dao.repositories import DomainRepository
from core.exceptions import *
from core.database.oracle import get_session


class DomainService:
    def __init__(self):
        pass

    @property
    def db_session(self):
        return get_session()
    
    async def get_name_and_desc_by_kb(self, kb_id: int) -> tuple[str, str]:
        """根据知识库ID获取业务域名称与描述"""
        async with self.db_session as session:
            repo = DomainRepository(session)
            try:
                domain_name, domain_descs = await repo.get_name_and_desc_by_kb(kb_id)
                return (domain_name, domain_descs)
            except Exception as e:
                handle_exception(e, f"根据知识库ID {kb_id} 获取业务域名称与描述出错")