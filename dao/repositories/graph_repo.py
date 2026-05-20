import json
import asyncio
import random
from typing import Any
from sqlalchemy import select, func
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
        file_id: str | None = None,
        attributes: dict[str, Any] | None = None
    ) -> None:
        """
        利用纯 ORM 语法进行双向合并：
        1. 增量 Upsert 边表（若存在则 weight + 1）
        2. 幂等插入边-切片关联表
        完全规避了原生 SQL 参数绑定中对 dict 转换导致的异常。
        """
        try:
            # -------- 1. 处理 GraphEdgeEntity (Upsert) --------
            edge_stmt = select(GraphEdgeEntity).where(GraphEdgeEntity.edge_id == edge_id)
            edge_result = await self.session.execute(edge_stmt)
            db_edge = edge_result.scalar_one_or_none()

            if db_edge:
                db_edge.weight = (db_edge.weight or 1) + 1
                db_edge.attributes = attributes
                db_edge.updated_at = func.current_timestamp()
            else:
                new_edge = GraphEdgeEntity(
                    edge_id=edge_id,
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=1,
                    attributes=attributes
                )
                self.session.add(new_edge)

            # -------- 2. 处理 GraphEdgeChunkMapEntity (Merge 语义) --------
            map_stmt = select(GraphEdgeChunkMapEntity).where(
                GraphEdgeChunkMapEntity.edge_id == edge_id,
                GraphEdgeChunkMapEntity.chunk_id == chunk_id
            )
            map_result = await self.session.execute(map_stmt)
            db_map = map_result.scalar_one_or_none()

            if not db_map:
                new_map = GraphEdgeChunkMapEntity(
                    edge_id=edge_id,
                    chunk_id=chunk_id,
                    file_id=file_id
                )
                self.session.add(new_map)

            await self.session.flush()
            logger.debug(f"[GraphRepo] Successfully upserted edge: {edge_id} and mapped to chunk: {chunk_id}")

        except Exception as e:
            logger.error(f"[GraphRepo] Failed to upsert edge {edge_id} with map: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to upsert graph edge or map", original_error=e)

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