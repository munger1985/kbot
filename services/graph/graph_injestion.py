import asyncio
import hashlib
from typing import Any
from loguru import logger

from core.database.oracle import get_session
from core.exceptions import *  # 保持与基础架构异常捕获一致
from utils.sanitize import sanitize_dict_for_oracle_json


class GraphIngestionService:
    """Graph Ingestion service for managing knowledge graph entity fusion and relation sync."""

    def __init__(
        self, 
        llm_client: Any,            # 你的 LLM 驱动实例
        embedding_client: Any       # 你的 Embedding 驱动实例
    ) -> None:
        self.llm_client = llm_client
        self.embedding_client = embedding_client

    @property
    def oracle_session(self):
        """Provides a database session instance following the core architecture pattern."""
        return get_session()

    def _generate_md5_id(self, *args: str) -> str:
        """全局一致的 ID 哈希生成器，确保相同实体的 ID 天然对齐"""
        content = "_".join([str(arg).strip().lower() for arg in args])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def merge_and_ingest_graph(
        self, 
        chunk_id: str, 
        file_id: str, 
        extracted_relations: list[dict[str, Any]]
    ) -> None:
        """文档解析管道调用的核心入口，采用统一的懒加载 Session 管理事务边界。"""
        if not extracted_relations:
            return

        # 延迟导入，防止出现由于服务循环依赖引起的启动异常
        from dao.repositories import GraphRepository 

        async with self.oracle_session as session:
            # 绑定当前业务事务周期的专属 Repo 实例
            repo = GraphRepository(session)

            async def _process_single_relation(rel: dict[str, Any]) -> None:
                try:
                    # 1. 处理源节点 (Source Vertex)
                    src_id = await self._process_vertex_fusion(
                        repo=repo,
                        name=rel["source_name"],
                        v_type=rel["source_type"],
                        new_desc=rel.get("source_desc") or "",
                        chunk_id=chunk_id
                    )

                    # 2. 处理目标节点 (Target Vertex)
                    tgt_id = await self._process_vertex_fusion(
                        repo=repo,
                        name=rel["target_name"],
                        v_type=rel["target_type"],
                        new_desc=rel.get("target_desc") or "",
                        chunk_id=chunk_id
                    )

                    # 3. 处理关系边及其溯源映射 (Edge & Chunk Map)
                    await self._process_edge_and_map(
                        repo=repo,
                        src_id=src_id,
                        tgt_id=tgt_id,
                        relation_type=rel["relation_type"],
                        edge_attributes=rel.get("relation_attributes") or {},
                        chunk_id=chunk_id,
                        file_id=file_id
                    )
                except Exception as e:
                    logger.error(f"[GraphIngestion] 处理图关系单条录入失败: {rel}, 错误: {str(e)}", exc_info=True)

            # 在当前的数据库会话事务内，并发处理当前文本块的所有三元组
            await asyncio.gather(*[_process_single_relation(r) for r in extracted_relations])
            
            # 显式提交当前 Chunk 级别的图结构变更
            await session.commit()
            logger.info(f"Successfully processed and committed graph network for chunk {chunk_id}")

    async def _process_vertex_fusion(self, repo: Any, name: str, v_type: str, new_desc: str, chunk_id: str) -> str:
        """百科体融合逻辑：查旧 -> LLM 融合 -> 重算 Vector -> Repo Upsert"""
        vertex_id = self._generate_md5_id(name, v_type)
        new_desc = new_desc.strip()
        
        # 统一使用传入的 repo 实例进行上下文查询
        existing_vertex = await repo.get_vertex_by_id(vertex_id)
        
        final_desc = new_desc
        final_vector = None
        attributes = {"last_updated_by_chunk": chunk_id}

        if existing_vertex:
            if existing_vertex.description and existing_vertex.description != final_desc:
                try:
                    final_desc = await self.llm_client.merge_descriptions(
                        entity_name=name,
                        old_desc=existing_vertex.description,
                        new_context=final_desc
                    )
                except Exception as llm_err:
                    logger.warning(f"[GraphIngestion] LLM 融合描述失败，降级沿用老描述. 错误: {llm_err}")
                    final_desc = existing_vertex.description
                
            if existing_vertex.description and final_desc == existing_vertex.description:
                final_vector = existing_vertex.name_vector
                if existing_vertex.attributes:
                    attributes.update(existing_vertex.attributes)
            else:
                final_vector = await self.embedding_client.get_vector(f"{name}: {final_desc}")
        else:
            final_vector = await self.embedding_client.get_vector(f"{name}: {final_desc}")

        sanitized_attrs = sanitize_dict_for_oracle_json(attributes)

        await repo.upsert_vertex(
            vertex_id=vertex_id,
            vertex_name=name,
            vertex_type=v_type,
            description=final_desc,
            attributes=sanitized_attrs,
            name_vector=final_vector
        )
        return vertex_id

    async def _process_edge_and_map(
        self, 
        repo: Any,
        src_id: str, 
        tgt_id: str, 
        relation_type: str, 
        edge_attributes: dict, 
        chunk_id: str, 
        file_id: str
    ) -> None:
        """处理关系的增量累加，并绑定当前切片源"""
        edge_id = self._generate_md5_id(src_id, tgt_id, relation_type)
        
        existing_edge = await repo.get_edge_by_id(edge_id)
        
        if existing_edge:
            new_weight = (existing_edge.weight or 1) + 1
            merged_attributes = existing_edge.attributes or {}
            if edge_attributes:
                merged_attributes.update(edge_attributes)
        else:
            new_weight = 1
            merged_attributes = edge_attributes or {}

        sanitized_edge_attrs = sanitize_dict_for_oracle_json(merged_attributes)

        # 1. 幂等更新边表
        await repo.upsert_edge(
            edge_id=edge_id,
            source_id=src_id,
            target_id=tgt_id,
            relation_type=relation_type,
            weight=new_weight,
            attributes=sanitized_edge_attrs
        )

        # 2. 写入关联映射表
        await repo.upsert_edge_chunk_map(
            edge_id=edge_id,
            chunk_id=chunk_id,
            file_id=file_id
        )