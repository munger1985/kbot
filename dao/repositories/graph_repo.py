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
        

    async def search_graph_context(
        self,
        vertex_names: list[str],
        max_depth: int = 2,
        limit: int = 30
    ) -> dict[str, Any]:
        """
        通过原生 SQL 检索知识图谱，查找与目标实体关联的 1~2 度子图，并召回关联的 Chunk ID。
        
        :param vertex_names: 检索词列表（比如从用户问题中提取出的实体）
        :param max_depth: 图游走深度，默认 2 度关联
        :param limit: 限制返回的边数量，防止子图爆炸
        :return: 包含实体、关系、以及溯源 Chunk ID 的结构化字典
        """
        if not vertex_names:
            return {"vertices": [], "edges": [], "chunk_ids": []}

        # Oracle SQL IN 绑定的安全处理
        # 构造形如 :name_0, :name_1 的动态占位符
        bind_params: dict[str, Any] = {f"name_{i}": name for i, name in enumerate(vertex_names)}
        in_clause = ", ".join(f":name_{i}" for i in range(len(vertex_names)))

        # 1. 深度整合图检索 SQL
        # 利用经典的 CONNECT BY 树状图游走或标准关联，顺便把关联的 chunk_id 聚合上来
        search_sql = text(f"""
            WITH sub_edges AS (
                -- 第一步：基于初始节点，在边表里向下游走检索关联的边
                SELECT DISTINCT
                    e.EDGE_ID,
                    e.SOURCE_ID,
                    e.TARGET_ID,
                    e.RELATION_TYPE,
                    e.WEIGHT,
                    e.ATTRIBUTES
                FROM KBOT_GRAPH_KNOWLEDGE_EDGES e
                START WITH e.SOURCE_ID IN (
                    SELECT v.VERTEX_ID 
                    FROM KBOT_GRAPH_KNOWLEDGE_VERTICES v 
                    WHERE v.VERTEX_NAME IN ({in_clause})
                )
                CONNECT BY PRIOR e.TARGET_ID = e.SOURCE_ID 
                AND LEVEL <= :max_depth
            )
            -- 第二步：拉出这些边对应的所有 Chunk 映射，并按边做合并
            SELECT 
                se.EDGE_ID,
                se.SOURCE_ID,
                se.TARGET_ID,
                se.RELATION_TYPE,
                se.WEIGHT,
                se.ATTRIBUTES,
                -- 将关联的 chunk_id 聚合为逗号分隔的字符串，方便应用层解析
                (
                    SELECT LISTAGG(m.CHUNK_ID, ',') WITHIN GROUP (ORDER BY m.CREATED_AT DESC)
                    FROM KBOT_GRAPH_EDGE_CHUNK_MAP m
                    WHERE m.EDGE_ID = se.EDGE_ID
                ) as AS_CHUNK_IDS,
                v_src.VERTEX_NAME as SOURCE_NAME,
                v_src.VERTEX_TYPE as SOURCE_TYPE,
                v_dst.VERTEX_NAME as TARGET_NAME,
                v_dst.VERTEX_TYPE as TARGET_TYPE
            FROM sub_edges se
            JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v_src ON se.SOURCE_ID = v_src.VERTEX_ID
            JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v_dst ON se.TARGET_ID = v_dst.VERTEX_ID
            WHERE ROWNUM <= :limit
        """)

        # 注入额外的控制参数
        bind_params["max_depth"] = max_depth
        bind_params["limit"] = limit

        # 2. 结构化组装返回对象
        try:
            result = await self.session.execute(search_sql, bind_params)
            rows = result.fetchall()

            vertices_set = set()
            edges_list = []
            chunk_ids_set = set()

            for row in rows:
                # 不管驱动吐出大写还是小写，强行转换为标准纯文本全小写字典
                # 这能完美免疫所有 Oracle 驱动层的大小写扯皮问题
                r = {str(k).lower(): v for k, v in row._mapping.items()}

                # 提取核心字段（全部使用小写安全的 key 提取）
                source_id = r.get("source_id")
                source_name = r.get("source_name")
                source_type = r.get("source_type")
                
                target_id = r.get("target_id")
                target_name = r.get("target_name")
                target_type = r.get("target_type")
                
                edge_id = r.get("edge_id")
                relation_type = r.get("relation_type")
                weight = r.get("weight")
                attributes = r.get("attributes")
                as_chunk_ids = r.get("as_chunk_ids")

                # 收集涉及到的实体（消歧后的双向节点）
                if source_id and source_name:
                    vertices_set.add((source_id, source_name, source_type))
                if target_id and target_name:
                    vertices_set.add((target_id, target_name, target_type))

                # 收集边
                if edge_id:
                    edges_list.append({
                        "edge_id": edge_id,
                        "source": source_name,
                        "target": target_name,
                        "relation": relation_type,
                        "weight": weight,
                        "attributes": attributes  # 如果是字符串，可以在上层进行 json.loads
                    })

                # 收集用来做 RAG 溯源召回的 Chunk ID
                if as_chunk_ids:
                    for cid in str(as_chunk_ids).split(','):
                        if cid.strip():
                            chunk_ids_set.add(cid.strip())

            # 格式化输出
            formatted_vertices = [
                {"id": v[0], "name": v[1], "type": v[2]} for v in vertices_set
            ]

            logger.info(f"🔮 [GraphSearch] 成功破冰！召回图谱实体 {len(formatted_vertices)} 个, 关系边 {len(edges_list)} 条, 溯源 Chunk {len(chunk_ids_set)} 个。")

            return {
                "vertices": formatted_vertices,
                "edges": edges_list,
                "chunk_ids": list(chunk_ids_set)
            }

        except Exception as e:
            # 打印完整的错误堆栈，拒绝静默失败
            logger.error(f"🚨 [GraphSearch 失败] 图检索运行时崩溃，错误详情: {str(e)}", exc_info=True)
            return {"vertices": [], "edges": [], "chunk_ids": []}