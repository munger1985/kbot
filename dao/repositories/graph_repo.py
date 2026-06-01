import json
from typing import Any
from sqlalchemy import select, func, text, bindparam
from loguru import logger
from core.exceptions import DatabaseException
from dao.entities import GraphVertexEntity, GraphEdgeEntity

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

    async def search_graph_context(
        self,
        kb_id: int,
        vertex_names: list[str],
        max_depth: int = 2,
        limit: int = 30,
        min_weight: int = 2
    ) -> dict[str, Any]:
        """
        Performs graph traversal retrieval via native Oracle CONNECT BY.
        Locates 1 to max_depth subgraphs based on extracted entities and gathers chunk references.
        """
        if not vertex_names:
            return {"vertices": [], "edges": [], "chunk_ids": []}

        bind_params: dict[str, Any] = {f"name_{i}": name for i, name in enumerate(vertex_names)}
        in_clause = ", ".join(f":name_{i}" for i in range(len(vertex_names)))

        search_sql = text(f"""
            WITH sub_edges AS (
                SELECT DISTINCT
                    e.EDGE_ID, e.SOURCE_ID, e.TARGET_ID, e.RELATION_TYPE, e.WEIGHT, e.ATTRIBUTES, e.KB_ID
                FROM KBOT_GRAPH_KNOWLEDGE_EDGES e
                WHERE e.KB_ID = :kb_id AND e.WEIGHT >= :min_weight
                START WITH e.KB_ID = :kb_id AND e.SOURCE_ID IN (
                    SELECT v.VERTEX_ID 
                    FROM KBOT_GRAPH_KNOWLEDGE_VERTICES v 
                    WHERE v.KB_ID = :kb_id AND v.VERTEX_NAME IN ({in_clause})
                )
                CONNECT BY PRIOR e.TARGET_ID = e.SOURCE_ID 
                AND PRIOR e.KB_ID = e.KB_ID
                AND e.WEIGHT >= :min_weight    
                AND LEVEL <= :max_depth
            ),
            ranked_edges AS (
                SELECT 
                    se.EDGE_ID, se.SOURCE_ID, se.TARGET_ID, se.RELATION_TYPE, se.WEIGHT, se.ATTRIBUTES,
                    (
                        SELECT LISTAGG(TO_CHAR(m.CHUNK_ID), ',') WITHIN GROUP (ORDER BY m.CREATED_AT DESC)
                        FROM KBOT_GRAPH_EDGE_CHUNK_MAP m
                        WHERE m.KB_ID = se.KB_ID AND m.EDGE_ID = se.EDGE_ID
                    ) as AS_CHUNK_IDS,
                    v_src.VERTEX_NAME as SOURCE_NAME,
                    v_src.VERTEX_TYPE as SOURCE_TYPE,
                    v_dst.VERTEX_NAME as TARGET_NAME,
                    v_dst.VERTEX_TYPE as TARGET_TYPE
                FROM sub_edges se
                JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v_src ON se.KB_ID = v_src.KB_ID AND se.SOURCE_ID = v_src.VERTEX_ID
                JOIN KBOT_GRAPH_KNOWLEDGE_VERTICES v_dst ON se.KB_ID = v_dst.KB_ID AND se.TARGET_ID = v_dst.VERTEX_ID
                ORDER BY se.WEIGHT DESC
            )
            SELECT * FROM ranked_edges 
            WHERE ROWNUM <= :limit
        """)

        bind_params["kb_id"] = int(kb_id)
        bind_params["max_depth"] = max_depth
        bind_params["limit"] = limit
        bind_params["min_weight"] = min_weight

        try:
            result = await self.session.execute(search_sql, bind_params)
            rows = result.fetchall()

            vertices_set = set()
            edges_list = []
            chunk_ids_set = set()

            for row in rows:
                r = {str(k).lower(): v for k, v in row._mapping.items()}
                
                # 🔍 诊断日志：记录第一行的所有键名，便于排查 Oracle 返回的列名是否异常
                if rows and row is rows[0]:
                    logger.debug(f"[GraphRepo] search_graph_context 返回第一行映射键名 (KB_ID {kb_id}): {list(r.keys())!r}")

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

                if source_id and source_name:
                    vertices_set.add((str(source_id), source_name, source_type))
                if target_id and target_name:
                    vertices_set.add((str(target_id), target_name, target_type))

                if edge_id:
                    edges_list.append({
                        "edge_id": str(edge_id),
                        "source": source_name,
                        "target": target_name,
                        "relation": relation_type,
                        "weight": weight,
                        "attributes": attributes
                    })

                if as_chunk_ids:
                    for cid in str(as_chunk_ids).split(','):
                        if cid.strip():
                            chunk_ids_set.add(cid.strip())

            formatted_vertices = [
                {"id": v[0], "name": v[1], "type": v[2]} for v in vertices_set
            ]

            logger.info(f"[GraphSearch] Traversal successful. Recalled {len(formatted_vertices)} vertices, {len(edges_list)} edges, and {len(chunk_ids_set)} chunks for KB: {kb_id}")

            return {
                "vertices": formatted_vertices,
                "edges": edges_list,
                "chunk_ids": list(chunk_ids_set)
            }

        except Exception as e:
            logger.error(f"[GraphSearch Failed] Graph retrieval execution crashed: {str(e)}", exc_info=True)
            return {"vertices": [], "edges": [], "chunk_ids": []}

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