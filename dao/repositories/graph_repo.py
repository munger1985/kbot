import json
from typing import Any
from sqlalchemy import text
from loguru import logger
from core.exceptions import DatabaseException
from dao.entities import GraphVertexEntity
from .base_repo import BaseRepository

class GraphRepository(BaseRepository[GraphVertexEntity]):
    """Repository for managing Oracle 26ai Property Graph data structures"""

    async def upsert_vertex(
        self, 
        vertex_id: str, 
        vertex_name: str, 
        vertex_type: str, 
        description: str | None = None, 
        attributes: dict[str, Any] | None = None, 
        name_vector: list[float] | None = None
    ) -> None:
        """
        利用 Oracle 26ai MERGE 语法单条高性能 Upsert 顶点
        """
        sql = """
        MERGE INTO kbot_graph_knowledge_vertices t
        USING (
            SELECT 
                :vertex_id AS vertex_id, 
                :vertex_name AS vertex_name, 
                :vertex_type AS vertex_type, 
                :description AS description, 
                :attributes AS attributes, 
                :name_vector AS name_vector 
            FROM dual
        ) s
        ON (t.vertex_id = s.vertex_id)
        WHEN MATCHED THEN
            UPDATE SET 
                t.description = s.description,
                t.name_vector = s.name_vector,
                t.attributes = s.attributes,
                t.updated_at = CURRENT_TIMESTAMP
        WHEN NOT MATCHED THEN
            INSERT (vertex_id, vertex_name, vertex_type, description, attributes, name_vector, created_at, updated_at)
            VALUES (s.vertex_id, s.vertex_name, s.vertex_type, s.description, s.attributes, s.name_vector, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        try:
            # 序列化 JSON 结构适应 Oracle JSON 字段要求
            attr_json = json.dumps(attributes) if attributes else None
            
            # Oracle 26ai 原生驱动支持传入 list/array 绑定给 VECTOR 字段类型
            await self.session.execute(
                text(sql),
                {
                    "vertex_id": vertex_id,
                    "vertex_name": vertex_name,
                    "vertex_type": vertex_type,
                    "description": description,
                    "attributes": attr_json,
                    "name_vector": name_vector
                }
            )
            await self.session.flush()
            logger.debug(f"[GraphRepo] Successfully upserted vertex: {vertex_id} ({vertex_name})")
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to upsert vertex {vertex_name}: {str(e)}", exc_info=True)
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
        高性能双向合并：
        1. 增量 Upsert 边表（若存在则 weight + 1）
        2. 幂等插入边-切片关联表（避免单 Chunk 内或跨重复文档重复绑定）
        """
        edge_sql = """
        MERGE INTO kbot_graph_knowledge_edges t
        USING (
            SELECT 
                :edge_id AS edge_id, 
                :source_id AS source_id, 
                :target_id AS target_id, 
                :relation_type AS relation_type, 
                :attributes AS attributes 
            FROM dual
        ) s
        ON (t.edge_id = s.edge_id)
        WHEN MATCHED THEN
            UPDATE SET 
                t.weight = t.weight + 1,
                t.updated_at = CURRENT_TIMESTAMP
        WHEN NOT MATCHED THEN
            INSERT (edge_id, source_id, target_id, relation_type, weight, attributes)
            VALUES (s.edge_id, s.source_id, s.target_id, s.relation_type, 1, s.attributes)
        """
        
        map_sql = """
        MERGE INTO kbot_graph_edge_chunk_map t
        USING (
            SELECT 
                :edge_id AS edge_id, 
                :chunk_id AS chunk_id, 
                :file_id AS file_id 
            FROM dual
        ) s
        ON (t.edge_id = s.edge_id AND t.chunk_id = s.chunk_id)
        WHEN NOT MATCHED THEN
            INSERT (edge_id, chunk_id, file_id)
            VALUES (s.edge_id, s.chunk_id, s.file_id)
        """
        try:
            attr_json = json.dumps(attributes) if attributes else None
            
            # 1. 更新边元数据与其权重
            await self.session.execute(
                text(edge_sql),
                {
                    "edge_id": edge_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                    "attributes": attr_json
                }
            )
            
            # 2. 绑定当前 Chunk ID 用于图搜索 Skill 到原文的精准溯源
            await self.session.execute(
                text(map_sql),
                {
                    "edge_id": edge_id,
                    "chunk_id": chunk_id,
                    "file_id": file_id
                }
            )
            await self.session.flush()
            logger.debug(f"[GraphRepo] Successfully upserted edge: {edge_id} and mapped to chunk: {chunk_id}")
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to upsert edge {edge_id} with map: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to upsert graph edge or map", original_error=e)