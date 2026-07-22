from loguru import logger
from dao.entities import ParserConfEntity
from dao.repositories import ParserConfRepository
from platform_core.exceptions import *
from platform_core.database.oracle import get_session

class ParserConfService:
    def __init__(self):
        pass

    @property
    def db_session(self):
        return get_session()

    async def add(self, user_name: str, domain_id: int, engine: str, parser_params: dict):
        """添加解析配置"""
        async with self.db_session as session:
            repo = ParserConfRepository(session)
            try:
                new_conf = ParserConfEntity(
                    domain_id=domain_id,
                    engine=engine,
                    parser_params=parser_params,
                    created_by=user_name
                )
                await repo.create(new_conf)
                logger.info(f"添加解析引擎配置 {engine} 成功")
            except Exception as e:
                handle_exception(e, f"添加解析引擎配置 {engine} 出错")

    async def modify(
        self, 
        parser_conf_id: int,
        user_name: str, 
        engine: str | None, 
        parser_params: dict | None
    ):
        """修改解析配置"""
        async with self.db_session as session:
            repo = ParserConfRepository(session)
            try:
                kwargs = {
                    "engine": engine,
                    "parser_params": parser_params if parser_params else None, 
                    "updated_by": user_name
                }
                # 过滤掉 None 值，避免覆盖原有数据
                update_data = {k: v for k, v in kwargs.items() if v is not None}
                
                await repo.update(parser_conf_id, **update_data)
                logger.info(f"修改解析引擎配置 {parser_conf_id} 成功")
            except Exception as e:
                handle_exception(e, f"修改解析引擎配置 {parser_conf_id} 出错")

    async def remove(self, parser_conf_id: int):
        """删除解析配置"""
        async with self.db_session as session:
            repo = ParserConfRepository(session)
            try:
                await repo.delete(parser_conf_id)
                logger.info(f"删除解析配置 {parser_conf_id} 成功")
            except Exception as e:
                handle_exception(e, f"删除解析配置 {parser_conf_id} 出错")

    async def get_parser(self, parser_conf_id: int) -> dict:
        """根据配置ID获取解析配置详情"""
        async with self.db_session as session:
            repo = ParserConfRepository(session)
            try:
                parser_conf = await repo.get(parser_conf_id)
                # 使用 Entity 的 to_dict 方法处理字段映射和类型转换
                return parser_conf.to_dict()
            except Exception as e:
                handle_exception(e, "获取解析配置详情出错")

    async def get_all(self, domain_id: int) -> list[dict]:
        """获取指定业务域下的所有解析配置"""
        async with self.db_session as session:
            repo = ParserConfRepository(session)
            try:
                parser_confs = await repo.get_all(domain_id)
                return [conf.to_dict() for conf in parser_confs]
            except Exception as e:
                handle_exception(e, "获取解析配置列表出错")

    async def get_parser_params_by_engine(self, domain_id: int, engine: str) -> dict:
        """根据解析引擎快速获取解析参数"""
        async with self.db_session as session:
            repo = ParserConfRepository(session)
            try:
                # 直接获取解析参数字典
                raw_params = await repo.get_parser_params_by_engine(domain_id, engine)
                return raw_params
            except Exception as e:
                handle_exception(e, f"获取 {engine} 解析参数失败")
