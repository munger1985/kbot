from sqlalchemy import select, update, delete, insert
from typing import Sequence
from dao.entities import ParserConfEntity
from platform_core.exceptions import DatabaseException, DataNotFoundException, DataConflictException
from .base_repo import BaseRepository

class ParserConfRepository(BaseRepository[ParserConfEntity]):
    """
    解析配置仓库
    """

    async def create(self, parser_conf: ParserConfEntity) -> None:
        """新建解析配置"""
        try:
            # 1. 检查冲突（在同一个注入的 session 中操作）
            result = await self.session.execute(
                select(ParserConfEntity)
                .where(ParserConfEntity.domain_id == parser_conf.domain_id)
                .where(ParserConfEntity.engine == parser_conf.engine)
            )
            if result.scalar_one_or_none():
                raise DataConflictException(
                    f"解析配置已存在: domain_id={parser_conf.domain_id}, engine={parser_conf.engine}"
                )
            
            # 2. 添加记录
            self.session.add(parser_conf)
            # 事务提交由 Service 层统一控制
        except DataConflictException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"创建解析配置失败", original_error=e)

    async def update(self, parser_conf_id: int, **kwargs) -> None:
        """根据ID更新解析配置"""
        try:
            result = await self.session.execute(
                update(ParserConfEntity)
                .where(ParserConfEntity.parser_conf_id == parser_conf_id)
                .values(**kwargs)
                .returning(ParserConfEntity.parser_conf_id)
            )
            if result.scalar() is None:
                raise DataNotFoundException(f"解析配置 {parser_conf_id} 不存在")
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"根据ID更新解析配置失败", original_error=e)

    async def delete(self, parser_conf_id: int) -> None:
        """根据ID删除解析配置"""
        try:
            await self.session.execute(
                delete(ParserConfEntity)
                .where(ParserConfEntity.parser_conf_id == parser_conf_id)
            )
        except Exception as e:
            raise DatabaseException(f"根据ID删除解析配置失败", original_error=e)

    async def get(self, parser_conf_id: int) -> ParserConfEntity:
        """根据ID获取解析配置"""
        try:
            result = await self.session.execute(
                select(ParserConfEntity)
                .where(ParserConfEntity.parser_conf_id == parser_conf_id)
            )
            parser_conf = result.scalar_one_or_none()
            if not parser_conf:
                raise DataNotFoundException(f"解析配置 {parser_conf_id} 不存在")
            return parser_conf
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"根据ID获取解析配置失败", original_error=e)

    async def get_all(self, domain_id: int) -> Sequence[ParserConfEntity]:
        """获取所有解析配置"""
        try:
            result = await self.session.execute(
                select(ParserConfEntity)
                .where(ParserConfEntity.domain_id == domain_id)
            )
            parser_confs = result.scalars().all()
            if not parser_confs:
                raise DataNotFoundException(f"不存在解析配置")
            return parser_confs
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取所有解析配置失败", original_error=e)

    async def get_parser_params_by_engine(self, domain_id: int, engine: str) -> dict:
        """根据解析引擎获取解析参数"""
        try:
            result = await self.session.execute(
                select(ParserConfEntity.parser_params)
                .where(ParserConfEntity.domain_id == domain_id)
                .where(ParserConfEntity.engine == engine.lower())
            )
            parser_params_dict = result.scalar_one_or_none()
            if not parser_params_dict:
                raise DataNotFoundException(f"解析引擎 {engine} 不存在解析配置")
            
            return parser_params_dict

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取解析引擎 {engine} 的解析参数失败", original_error=e)
