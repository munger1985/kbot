import json
import asyncio
import random
from typing import Any
from sqlalchemy import select, func, text, insert, update, and_
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
        """
        通过纯原生 Oracle MERGE SQL 彻底终结 ORA-00942 与回滚 KeyError。
        直接对齐大写表名与列名，清晰透明。
        """
        try:
            # 1. 严格控制 JSON 序列化格式
            attributes_json = json.dumps(attributes, ensure_ascii=False) if isinstance(attributes, dict) else "{}"

            # ========================================================
            # 第一步：原生 MERGE 写入边表 (KBOT_GRAPH_KNOWLEDGE_EDGES)
            # ========================================================
            # 注意：Oracle 26ai 原生支持 JSON 类型，直接传入序列化后的 JSON 字符串即可
            edge_sql = text("""
                MERGE INTO KBOT_GRAPH_KNOWLEDGE_EDGES t
                USING DUAL
                ON (t.EDGE_ID = :edge_id)
                WHEN MATCHED THEN
                    UPDATE SET 
                        t.WEIGHT = t.WEIGHT + 1,
                        t.ATTRIBUTES = :attributes,
                        t.UPDATED_AT = CURRENT_TIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (EDGE_ID, SOURCE_ID, TARGET_ID, RELATION_TYPE, WEIGHT, ATTRIBUTES, CREATED_AT, UPDATED_AT)
                    VALUES (:edge_id, :source_id, :target_id, :relation_type, 1, :attributes, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)

            await self.session.execute(
                edge_sql,
                {
                    "edge_id": str(edge_id),
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "relation_type": str(relation_type),
                    "attributes": attributes_json
                }
            )
            logger.debug(f"[GraphRepo] Edge {edge_id} processed via native SQL.")

            # ========================================================
            # 第二步：原生 MERGE 写入映射表 (KBOT_GRAPH_EDGE_CHUNK_MAP)
            # ========================================================
            map_sql = text("""
                MERGE INTO KBOT_GRAPH_EDGE_CHUNK_MAP m
                USING DUAL
                ON (m.EDGE_ID = :edge_id AND m.CHUNK_ID = :chunk_id)
                WHEN MATCHED THEN
                    UPDATE SET 
                        m.FILE_ID = :file_id,
                        m.UPDATED_AT = CURRENT_TIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (EDGE_ID, CHUNK_ID, FILE_ID, CREATED_AT, UPDATED_AT)
                    VALUES (:edge_id, :chunk_id, :file_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)

            await self.session.execute(
                map_sql,
                {
                    "edge_id": str(edge_id),
                    "chunk_id": str(chunk_id),
                    "file_id": str(file_id) if file_id else None
                }
            )
            logger.debug(f"[GraphRepo] Edge-Chunk map linked for chunk: {chunk_id}")

        except Exception as e:
            logger.error(f"🚨 [GraphRepo Native 崩溃] 核心原生 SQL 执行失败。错误详情: {str(e)}")
            raise DatabaseException(f"Failed to upsert graph edge with map", original_error=e)

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