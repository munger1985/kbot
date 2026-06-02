import json
from typing import Any
from sqlalchemy import select, func, text, bindparam, Integer
from loguru import logger
from core.exceptions import DatabaseException
from dao.entities import GraphVertexEntity, GraphEdgeEntity
from utils.oracle_vec_handler import OracleVecHandler

class GraphRepository:
    """Repository for managing Oracle 26ai Property Graph data structures using SQLAlchemy 2.0 ORM & Native SQL"""

    def __init__(self, session):
        self.session = session

    async def upsert_vertex(
        self, 
        kb_id: int,
        vertex_id: str, 
        vertex_name: str, 
        vertex_type: str, 
        description: str | None = None, 
        attributes: dict[str, Any] | None = None, 
        name_vector: list[float] | None = None
    ) -> None:
        """
        Upserts a single graph vertex using SQLAlchemy 2.0 ORM mechanism.
        Queries against a composite key (kb_id, vertex_id) to hit performance indexes.
        """
        try:
            stmt = select(GraphVertexEntity).where(
                GraphVertexEntity.kb_id == kb_id,
                GraphVertexEntity.vertex_id == vertex_id
            )
            result = await self.session.execute(stmt)
            db_vertex = result.scalar_one_or_none()

            if db_vertex:
                db_vertex.description = description
                db_vertex.name_vector = name_vector
                db_vertex.attributes = attributes
                db_vertex.updated_at = func.current_timestamp()
            else:
                new_vertex = GraphVertexEntity(
                    kb_id=kb_id,
                    vertex_id=vertex_id,
                    vertex_name=vertex_name,
                    vertex_type=vertex_type,
                    description=description,
                    attributes=attributes,
                    name_vector=name_vector
                )
                self.session.add(new_vertex)
            
            await self.session.flush()
            logger.debug(f"[GraphRepo] Successfully upserted vertex: {vertex_id} ({vertex_name}) in KB: {kb_id}")
            
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to upsert vertex {vertex_name}. Internal Error: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to upsert graph vertex", original_error=e)

    async def upsert_edge_with_map(
        self,
        kb_id: int,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        chunk_id: str,
        file_id: str,
        attributes: dict
    ) -> None:
        """
        Upserts an edge and its chunk mapping via pure native Oracle MERGE SQL.
        Locks down matching scopes on composite keys with uppercase table/column alignment.
        """
        try:
            attributes_json = json.dumps(attributes, ensure_ascii=False) if isinstance(attributes, dict) else "{}"

            # ========================================================
            # Step 1: Native MERGE into Edge Table
            # ========================================================
            edge_sql = text("""
                MERGE INTO KBOT_GRAPH_KNOWLEDGE_EDGES t
                USING DUAL
                ON (t.KB_ID = :kb_id AND t.EDGE_ID = :edge_id)
                WHEN MATCHED THEN
                    UPDATE SET 
                        t.WEIGHT = t.WEIGHT + 1,
                        t.ATTRIBUTES = JSON_MERGEPATCH(t.ATTRIBUTES, :attributes),
                        t.UPDATED_AT = CURRENT_TIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (KB_ID, EDGE_ID, SOURCE_ID, TARGET_ID, RELATION_TYPE, WEIGHT, ATTRIBUTES, CREATED_AT, UPDATED_AT)
                    VALUES (:kb_id, :edge_id, :source_id, :target_id, :relation_type, 1, :attributes, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)

            await self.session.execute(
                edge_sql,
                {
                    "kb_id": int(kb_id),
                    "edge_id": str(edge_id),
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "relation_type": str(relation_type),
                    "attributes": attributes_json
                }
            )

            # ========================================================
            # Step 2: Native MERGE into Edge-Chunk Map Table
            # ========================================================
            map_sql = text("""
                MERGE INTO KBOT_GRAPH_EDGE_CHUNK_MAP m
                USING DUAL
                ON (m.KB_ID = :kb_id AND m.EDGE_ID = :edge_id AND m.CHUNK_ID = :chunk_id)
                WHEN MATCHED THEN
                    UPDATE SET 
                        m.FILE_ID = :file_id,
                        m.UPDATED_AT = CURRENT_TIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (KB_ID, EDGE_ID, CHUNK_ID, FILE_ID, CREATED_AT, UPDATED_AT)
                    VALUES (:kb_id, :edge_id, :chunk_id, :file_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)

            await self.session.execute(
                map_sql,
                {
                    "kb_id": int(kb_id),
                    "edge_id": str(edge_id),
                    "chunk_id": str(chunk_id),
                    "file_id": str(file_id) if file_id else None
                }
            )
            logger.debug(f"[GraphRepo] Edge-Chunk map linked. KB: {kb_id}, Chunk: {chunk_id}")

        except Exception as e:
            logger.error(f"[GraphRepo Native Crash] Core native SQL execution failed. Details: {str(e)}")
            raise DatabaseException(f"Failed to upsert graph edge with map", original_error=e)

    async def get_vertex_by_id(self, kb_id: int, vertex_id: str) -> GraphVertexEntity | None:
        """Fetches a single vertex using the composite key (kb_id, vertex_id)"""
        try:
            stmt = select(GraphVertexEntity).where(
                GraphVertexEntity.kb_id == kb_id,
                GraphVertexEntity.vertex_id == vertex_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to fetch vertex by id {vertex_id} in KB {kb_id}: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to fetch graph vertex by id", original_error=e)

    async def get_edge_by_id(self, kb_id: int, edge_id: str) -> GraphEdgeEntity | None:
        """Fetches a single edge using the composite key (kb_id, edge_id)"""
        try:
            stmt = select(GraphEdgeEntity).where(
                GraphEdgeEntity.kb_id == kb_id,
                GraphEdgeEntity.edge_id == edge_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to fetch edge by id {edge_id} in KB {kb_id}: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to fetch graph edge by id", original_error=e)

    # async def search_graph_context(
    #     self,
    #     kb_id: int,
    #     vertex_names: list[str],
    #     max_depth: int = 2,
    #     limit: int = 30,
    #     min_weight: int = 2
    # ) -> dict[str, Any]:
    #     """
    #     Performs graph traversal retrieval via native Oracle CONNECT BY.
    #     Locates 1 to max_depth subgraphs based on extracted entities and gathers chunk references.
    #     """
    #     if not vertex_names:
    #         return {"vertices": [], "edges": [], "chunk_ids": []}

    #     bind_params: dict[str, Any] = {f"name_{i}": name for i, name in enumerate(vertex_names)}
    #     in_clause = ", ".join(f":name_{i}" for i in range(len(vertex_names)))

    #     search_sql = text(f"""
    #         WITH sub_edges AS (
    #             SELECT DISTINCT
    #                 e.EDGE_ID, e.SOURCE_ID, e.TARGET_ID, e.RELATION_TYPE, e.WEIGHT, e.ATTRIBUTES, e.KB_ID
    #             FROM KBOT_GRAPH_KNOWLEDGE_EDGES e
    #             WHERE e.KB_ID = :kb_id AND e.WEIGHT >= :min_weight
    #             START WITH e.KB_ID = :kb_id AND e.SOURCE_ID IN (
    #                 SELECT v.VERTEX_ID 
    #                 FROM KBOT_GRAPH_KNOWLEDGE_VERTICES v 
    #                 WHERE v.KB_ID = :kb_id AND v.VERTEX_NAME IN ({in_clause})
    #             )
    #             CONNECT BY PRIOR e.TARGET_ID = e.SOURCE_ID 
    #             AND PRIOR e.KB_ID = e.KB_ID
    #             AND e.WEIGHT >= :min_weight    
    #             AND LEVEL <= :max_depth
    #         ),
    #         ranked_edges AS (
    #             SELECT 
    #                 se.EDGE_ID, se.SOURCE_ID, se.TARGET_ID, se.RELATION_TYPE, se.WEIGHT, se.ATTRIBUTES,
    #                 (
    #                     SELECT LISTAGG(TO_CHAR(m.CHUNK_ID), ',') WITHIN GROUP (ORDER BY m.CREATED_AT DESC)
    #                     FROM KBOT_GRAPH_EDGE_CHUNK_MAP m
    #                     WHERE m.KB_ID = se.KB_ID AND m.EDGE_ID = se.EDGE_ID
    #                 ) as AS_CHUNK_IDS,
    #                 v_src.VERTEX_NAME as SOURCE_NAME,
    #                 v_src.VERTEX_TYPE as SOURCE_TYPE,
    #                 v_dst.VERTEX_NAME as TARGET_NAME,
    #                 v_dst.VERTEX_TYPE as TARGET_TYPE
    #             FROM sub_edges se
    #             JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v_src ON se.KB_ID = v_src.KB_ID AND se.SOURCE_ID = v_src.VERTEX_ID
    #             JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v_dst ON se.KB_ID = v_dst.KB_ID AND se.TARGET_ID = v_dst.VERTEX_ID
    #             ORDER BY se.WEIGHT DESC
    #         )
    #         SELECT * FROM ranked_edges 
    #         WHERE ROWNUM <= :limit
    #     """)

    #     bind_params["kb_id"] = int(kb_id)
    #     bind_params["max_depth"] = max_depth
    #     bind_params["limit"] = limit
    #     bind_params["min_weight"] = min_weight

    #     try:
    #         result = await self.session.execute(search_sql, bind_params)
    #         rows = result.fetchall()

    #         vertices_set = set()
    #         edges_list = []
    #         chunk_ids_set = set()

    #         for row in rows:
    #             r = {str(k).lower(): v for k, v in row._mapping.items()}
                
    #             # 🔍 诊断日志：记录第一行的所有键名，便于排查 Oracle 返回的列名是否异常
    #             if rows and row is rows[0]:
    #                 logger.debug(f"[GraphRepo] search_graph_context 返回第一行映射键名 (KB_ID {kb_id}): {list(r.keys())!r}")

    #             source_id = r.get("source_id")
    #             source_name = r.get("source_name")
    #             source_type = r.get("source_type")
                
    #             target_id = r.get("target_id")
    #             target_name = r.get("target_name")
    #             target_type = r.get("target_type")
                
    #             edge_id = r.get("edge_id")
    #             relation_type = r.get("relation_type")
    #             weight = r.get("weight")
    #             attributes = r.get("attributes")
    #             as_chunk_ids = r.get("as_chunk_ids")

    #             if source_id and source_name:
    #                 vertices_set.add((str(source_id), source_name, source_type))
    #             if target_id and target_name:
    #                 vertices_set.add((str(target_id), target_name, target_type))

    #             if edge_id:
    #                 edges_list.append({
    #                     "edge_id": str(edge_id),
    #                     "source": source_name,
    #                     "target": target_name,
    #                     "relation": relation_type,
    #                     "weight": weight,
    #                     "attributes": attributes
    #                 })

    #             if as_chunk_ids:
    #                 for cid in str(as_chunk_ids).split(','):
    #                     if cid.strip():
    #                         chunk_ids_set.add(cid.strip())

    #         formatted_vertices = [
    #             {"id": v[0], "name": v[1], "type": v[2]} for v in vertices_set
    #         ]

    #         logger.info(f"[GraphSearch] Traversal successful. Recalled {len(formatted_vertices)} vertices, {len(edges_list)} edges, and {len(chunk_ids_set)} chunks for KB: {kb_id}")

    #         return {
    #             "vertices": formatted_vertices,
    #             "edges": edges_list,
    #             "chunk_ids": list(chunk_ids_set)
    #         }

    #     except Exception as e:
    #         logger.error(f"[GraphSearch Failed] Graph retrieval execution crashed: {str(e)}", exc_info=True)
    #         return {"vertices": [], "edges": [], "chunk_ids": []}

    async def search_graph_context(
        self, 
        kb_id: int, 
        vertex_names: list[str], 
        limit: int = 30, 
        min_weight: int = 2,
        **kwargs
    ) -> dict:
        """
        实体驱动的知识图谱空间网络检索 (Graph-RAG)。
        
        通过给定的核心实体名称列表，召回关联的图谱拓扑网络边及映射的底层文本切片 ID 集合。

        Args:
            kb_id: 知识库唯一标识ID
            vertex_names: 待检索的核心实体名称列表
            limit: 最大边召回上限
            min_weight: 关系的最小置信度权重过滤阈值
            *args: 向上兼容的匿名位置参数
            **kwargs: 向上兼容的动态关键字参数（自动吸纳 max_depth 等参数）

        Returns:
            dict: 符合上层契约要求的结构化字典：
                {
                    "edges": list[dict],    # 拓扑边关系列表
                    "chunk_ids": list[str]  # 关联的底层非结构化文本块切片ID集合
                }
        """
        if not vertex_names:
            return {"edges": [], "chunk_ids": []}

        max_depth = kwargs.get("max_depth", 1)

        logger.info(
            f"[GraphRepo] Starting graph context retrieval. "
            f"KB_ID: {kb_id}, Vertices Count: {len(vertex_names)}, Limit: {limit}, "
            f"MinWeight: {min_weight}, MaxDepth(Ext): {max_depth}"
        )
        
        try:
            # 1. 动态构建 SQL 中的 IN 集合占位符 (如 :name_0, :name_1)
            in_clauses = [f":name_{i}" for i in range(len(vertex_names))]
            in_expr = ", ".join(in_clauses)

            # 2. 标量硬编码注入防御 + 多表连接提取 CHUNK_ID：
            # 连接关系边切片映射表 (KBOT_GRAPH_EDGE_CHUNK_MAP) 以便为上层提供批量回表所需的 chunk_ids。
            # 整数标量通过 f-string 物理嵌入，阻断底层组件在映射生命周期内的参数字典冲突。
            graph_sql_text = f"""
                SELECT 
                    v1.VERTEX_NAME as source_name, 
                    v1.VERTEX_TYPE as source_type,
                    e.RELATION_TYPE as relation_type, 
                    e.WEIGHT as weight,
                    v2.VERTEX_NAME as target_name, 
                    v2.VERTEX_TYPE as target_type,
                    m.CHUNK_ID as chunk_id
                FROM KBOT_GRAPH_KNOWLEDGE_EDGES e
                JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v1 
                    ON e.KB_ID = v1.KB_ID AND e.SOURCE_ID = v1.VERTEX_ID
                JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v2 
                    ON e.KB_ID = v2.KB_ID AND e.TARGET_ID = v2.VERTEX_ID
                LEFT JOIN KBOT_GRAPH_EDGE_CHUNK_MAP m
                    ON e.KB_ID = m.KB_ID AND e.EDGE_ID = m.EDGE_ID
                WHERE e.KB_ID = {int(kb_id)}
                AND e.WEIGHT >= {int(min_weight)}
                AND (v1.VERTEX_NAME IN ({in_expr}) OR v2.VERTEX_NAME IN ({in_expr}))
                ORDER BY e.WEIGHT DESC
                FETCH FIRST {int(limit)} ROWS ONLY
            """

            # 3. 构建安全的纯净绑定参数字典
            bind_params = {}
            for i, name in enumerate(vertex_names):
                key = f"name_{i}"
                val = str(name)
                bind_params[key] = val
                bind_params[f"'{key}'"] = val  # 铁壁防线：防备隐式转义比对冲突

            # 4. 执行异步数据库检索
            result = await self.session.execute(text(graph_sql_text), bind_params)
            rows = result.fetchall()
            
            # 5. 严格对齐上层服务的 Dict 契约结构进行解析封装
            network_edges = []
            chunk_ids_set = set()
            
            # 提取去重，保证拓扑边和 chunk_id 的平铺归集
            for row in rows:
                # 封装拓扑关系边
                edge_item = {
                    "source": str(row[0]),
                    "source_type": str(row[1]),
                    "relation": str(row[2]),
                    "weight": int(row[3]),
                    "target": str(row[4]),
                    "target_type": str(row[5])
                }
                # 避免将重复的边放入列表
                if edge_item not in network_edges:
                    network_edges.append(edge_item)
                    
                # 归集底层关联文本块 ID
                if row[6]:
                    chunk_ids_set.add(str(row[6]))
                
            ret_dict = {
                "edges": network_edges,
                "chunk_ids": list(chunk_ids_set)
            }
                
            logger.info(
                f"[GraphRepo] Graph context retrieval completed successfully. "
                f"KB_ID: {kb_id}, Unique Edges Count: {len(network_edges)}, Unique Chunk IDs Count: {len(ret_dict['chunk_ids'])}"
            )
            return ret_dict

        except Exception as e:
            logger.error(
                f"[GraphRepo] Failed to execute graph connection query. "
                f"KB_ID: {kb_id}, Error Type: {type(e).__name__}, Message: {str(e)}", 
                exc_info=True
            )
            raise DatabaseException(message="Failed to execute graph connection query.", original_error=e)
        
    async def delete_graph_by_file(self, kb_id: int, file_ids: list[str]) -> int:
        """
        Removes file graph components from Oracle based on KB ID and an incoming string list of File IDs.
        Passes a tuple of strings safely matching the SQL 'IN' binder requirements.
        """
        if not file_ids:
            logger.warning("[GraphRepo] Received empty file ID list. Skipping graph context cleanup.")
            return 0

        try:
            formatted_file_ids = tuple(file_ids)

            # ========================================================
            # Step 1: Clean up the mappings for targeted file strings
            # ========================================================
            delete_map_sql = text("""
                DELETE FROM KBOT_GRAPH_EDGE_CHUNK_MAP 
                WHERE kb_id = :kb_id AND file_id IN :file_ids;
            """).bindparams(bindparam('file_ids', expanding=True))
            await self.session.execute(
                delete_map_sql, 
                {
                    "kb_id": int(kb_id), 
                    "file_ids": formatted_file_ids
                }
            )

            # ========================================================
            # Step 2: Clean up dead edges with no existing chunk maps
            # ========================================================
            delete_dead_edges_sql = text("""
                DELETE FROM KBOT_GRAPH_KNOWLEDGE_EDGES e
                WHERE e.KB_ID = :kb_id 
                  AND NOT EXISTS (
                      SELECT 1 FROM KBOT_GRAPH_EDGE_CHUNK_MAP m 
                      WHERE m.KB_ID = e.KB_ID AND m.EDGE_ID = e.EDGE_ID
                  )
            """)
            edge_res = await self.session.execute(delete_dead_edges_sql, {"kb_id": int(kb_id)})
            deleted_edges_count = edge_res.rowcount

            # ========================================================
            # Step 3: Remove isolated vertices with no connecting paths
            # ========================================================
            delete_orphan_vertices_sql = text("""
                DELETE FROM KBOT_GRAPH_KNOWLEDGE_VERTICES v
                WHERE v.KB_ID = :kb_id
                  AND NOT EXISTS (SELECT 1 FROM KBOT_GRAPH_KNOWLEDGE_EDGES e WHERE e.KB_ID = v.KB_ID AND e.SOURCE_ID = v.VERTEX_ID)
                  AND NOT EXISTS (SELECT 1 FROM KBOT_GRAPH_KNOWLEDGE_EDGES e WHERE e.KB_ID = v.KB_ID AND e.TARGET_ID = v.VERTEX_ID)
            """)
            await self.session.execute(delete_orphan_vertices_sql, {"kb_id": int(kb_id)})

            logger.info(f"[GraphRepo] Batch file graph erasure finished. KB: {kb_id}, Removed Dead Edges: {deleted_edges_count}")
            return deleted_edges_count

        except Exception as e:
            logger.error(f"[GraphRepo] Failed to delete graph by file list for kb_id: {kb_id}, Error: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to delete graph by file list", original_error=e)

    async def delete_graph_by_knowledge_base(self, kb_id: int) -> None:
        """
        Drops all graph objects bound to a specific Knowledge Base ID.
        Relies on downstream ON DELETE CASCADE rules to wipe maps and edges automatically.
        """
        try:
            delete_all_sql = text("""
                DELETE FROM KBOT_GRAPH_KNOWLEDGE_VERTICES 
                WHERE KB_ID = :kb_id
            """)
            result = await self.session.execute(delete_all_sql, {"kb_id": int(kb_id)})
            
            logger.warning(f"[GraphRepo] Executed complete graph purge for KB: {kb_id}. Removed Vertices: {result.rowcount}")
            
        except Exception as e:
            logger.error(f"[GraphRepo] Failed to wipe graph for kb_id: {kb_id}, Error: {str(e)}")
            raise DatabaseException(f"Failed to delete graph by knowledge base", original_error=e)
        
    async def get_vertex_names_by_embedding(self, kb_id: int, keyword_embedding: list[float], top_k: int = 10) -> list[str]:
        """
        利用 Oracle 23ai/26ai 原生向量检索，单次通过单个 Embedding 向量召回最接近的真实节点名称。
        """
        if not keyword_embedding:
            return []

        logger.info(f"[BUG_TRACK_VECTOR] === 物理净化向量检索开始 ===")

        try:
            # 1. 强制在 Python 内存中直接完成向量文本组装，变成纯文本字面量
            vec_literal = '[' + ','.join(map(str, keyword_embedding)) + ']'

            # 2. 🚨 物理核平：SQL 语句中绝对不留任何一个冒号参数！列名精确定位为 "NAME_VECTOR"
            vertices_sql = text(f"""
                SELECT VERTEX_NAME
                FROM KBOT_GRAPH_KNOWLEDGE_VERTICES
                WHERE KB_ID = {int(kb_id)}
                AND NAME_VECTOR IS NOT NULL
                ORDER BY VECTOR_DISTANCE(NAME_VECTOR, to_vector('{vec_literal}'), COSINE) ASC
                FETCH FIRST {int(top_k)} ROWS ONLY
            """)

            # 3. 保持空的绑定字典，彻底断绝任何内置/外置组件对 bind_params 字典做手脚、加单引号的可能
            bind_params = {}
            
            result = await self.session.execute(vertices_sql, bind_params)
            rows = result.fetchall()
            
            clean_names = []
            for row in rows:
                if row and row[0]:
                    clean_name = str(row[0]).strip("'\" ")
                    if clean_name:
                        clean_names.append(clean_name)
            
            logger.info(f"[BUG_TRACK_VECTOR] 物理净化检索大获成功，召回行数: {len(clean_names)}")
            return clean_names

        except Exception as e:
            logger.error(f"[BUG_TRACK_VECTOR] 物理净化模式遭遇异常: {str(e)}", exc_info=True)
            raise e