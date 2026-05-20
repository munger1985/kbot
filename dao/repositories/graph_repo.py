import json
import asyncio
import random
from typing import Any
from sqlalchemy import select, func, text
from loguru import logger
from core.exceptions import DatabaseException
from dao.entities import GraphVertexEntity, GraphEdgeEntity, GraphEdgeChunkMapEntity # 确认你的实体类名

class GraphRepository:
    """Repository for managing Oracle 26ai Property Graph data structures using pure SQLAlchemy 2.0 ORM"""

    def __init__(self, session):
        self.session = session

    async def upsert_vertex(
        self, 
        vertex_id: str, 
        vertex_name: str, 
        vertex_type: str, 
        description: str | None = None, 
        attributes: dict[str, Any] | None = None, 
        name_vector: list[float] | None = None
    ) -> None:
        """利用 SQLAlchemy 2.0 官方 ORM 机制进行单条 Upsert 顶点"""
        try:
            stmt = select(GraphVertexEntity).where(GraphVertexEntity.vertex_id == vertex_id)
            result = await self.session.execute(stmt)
            db_vertex = result.scalar_one_or_none()

            if db_vertex:
                db_vertex.description = description
                db_vertex.name_vector = name_vector
                db_vertex.attributes = attributes
                db_vertex.updated_at = func.current_timestamp()
            else:
                new_vertex = GraphVertexEntity(
                    vertex_id=vertex_id,
                    vertex_name=vertex_name,
                    vertex_type=vertex_type,
                    description=description,
                    attributes=attributes,
                    name_vector=name_vector
                )
                self.session.add(new_vertex)
            
            await self.session.flush()
            logger.debug(f"[GraphRepo] Successfully upserted vertex: {vertex_id} ({vertex_name})")
            
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to upsert vertex {vertex_name}. Internal Error: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to upsert graph vertex", original_error=e)

    async def upsert_edge_with_map(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        chunk_id: str,
        file_id: str,
        attributes: dict
    ) -> None:
        """采用 Oracle 原生 MERGE 语句直接落库。"""
        # 1. 严格确保属性字典转为了纯净的 JSON 字符串，不给 ORM 留任何转译空间
        # 即使 Oracle 26ai 支持原生的 JSON 类型，在绑定变量时传标准字符串也是最安全的
        attributes_json = json.dumps(attributes, ensure_ascii=False)

        # 2. 编写 Oracle 标准的 MERGE INTO 语句
        # 请根据你具体的表名（如 graph_edges）以及字段名进行微调
        statement = text("""
            MERGE INTO graph_edges t
            USING (
                SELECT 
                    :edge_id AS edge_id, 
                    :source_id AS source_id, 
                    :target_id AS target_id, 
                    :relation_type AS relation_type,
                    :chunk_id AS chunk_id,
                    :file_id AS file_id,
                    :attributes AS attributes
                FROM DUAL
            ) s
            ON (t.edge_id = s.edge_id)
            WHEN MATCHED THEN
                UPDATE SET 
                    t.chunk_id = s.chunk_id,
                    t.attributes = s.attributes
            WHEN NOT MATCHED THEN
                INSERT (edge_id, source_id, target_id, relation_type, chunk_id, file_id, attributes)
                VALUES (s.edge_id, s.source_id, s.target_id, s.relation_type, s.chunk_id, s.file_id, s.attributes)
        """)

        # 3. 使用异步 session 直接 execute 执行
        # 这会绕过所有的 ORM 属性生命周期，直接把纯净的数据砸进驱动里
        await self.session.execute(
            statement,
            {
                "edge_id": str(edge_id),
                "source_id": str(source_id),
                "target_id": str(target_id),
                "relation_type": str(relation_type),
                "chunk_id": str(chunk_id),
                "file_id": str(file_id),
                "attributes": attributes_json  # 传入序列化后的纯字符串
            }
        )
        try:
            logger.info("[诊断拦截] 已经完成所有节点和边的处理，准备执行最后一步：session.commit()...")
            await self.session.commit()
            logger.info(f"Successfully processed and committed graph network for chunk {chunk_id}")
        except Exception as commit_exc:
            # 🚨 核心武器：利用 traceback 还原真正的犯罪现场，不经任何二次过滤
            import traceback
            error_stack = traceback.format_exc()
            
            logger.error("=" * 80)
            logger.error("🚨🚨🚨 [终极拦截] 抓到了！commit 阶段爆发了致命异常 🚨🚨🚨")
            logger.error(f"异常原始类型: {type(commit_exc)}")
            logger.error(f"异常原始信息: {str(commit_exc)}")
            logger.error("⬇️⬇️⬇️ 以下为未受污染的完整调用栈（请仔细观察最后一行的报错文件名和行号） ⬇️⬇️⬇️")
            logger.error(f"\n{error_stack}")
            logger.error("=" * 80)
            
            # 重新抛出，保证原有的回滚逻辑不受影响
            raise commit_exc

    async def get_vertex_by_id(self, vertex_id: str) -> GraphVertexEntity | None:
        """使用 ORM 查找顶点，直接返回模型实体，自带属性解析与小写列对齐防御"""
        try:
            stmt = select(GraphVertexEntity).where(GraphVertexEntity.vertex_id == vertex_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to fetch vertex by id {vertex_id}: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to fetch graph vertex by id", original_error=e)

    async def get_edge_by_id(self, edge_id: str) -> GraphEdgeEntity | None:
        """使用 ORM 查找关系边，直接返回模型实体"""
        try:
            stmt = select(GraphEdgeEntity).where(GraphEdgeEntity.edge_id == edge_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to fetch edge by id {edge_id}: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to fetch graph edge by id", original_error=e)