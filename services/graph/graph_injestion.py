import asyncio
import hashlib
from typing import Any
from loguru import logger

from core.database.oracle import get_session
from core.exceptions import *
from core.config.settings import get_prompt_config
from utils.sanitize import sanitize_dict_for_oracle_json
from dao.repositories import GraphRepository 
from .schemas import GraphAnalysis
from utils.clients import AIModelClient
from agent.prompt import default_prompt


class GraphIngestionService:
    """Graph Ingestion service for managing knowledge graph entity fusion and relation sync."""

    def __init__(self, embedding_model: str, llm_model: str) -> None:
        from utils.clients import AIModelClient # 延时引入
        self.model_client = AIModelClient()
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    @property
    def oracle_session(self):
        from core.database.oracle import get_session
        return get_session()

    def _generate_md5_id(self, *args: str) -> str:
        """全局一致的 ID 哈希生成器，确保相同实体的 ID 天然对齐"""
        content = "_".join([str(arg).strip().lower() for arg in args])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def extract_triplets(self, user_input_text: str, llm_model_name: str) -> GraphAnalysis:
        """利用大模型从输入文本中抽取知识图谱实体与关系"""
        from core.config.settings import get_prompt_config
        from agent.prompt import default_prompt
        try:
            prompt = await default_prompt.generate(
                get_prompt_config().graph_extractor, 
                text=user_input_text
            )
            data = await self.model_client.get_llm_json(
                model_name=llm_model_name,
                prompt=prompt,
                temperature=0
            )
            return GraphAnalysis(**data)
        except Exception as e:
            logger.error(f"图谱结构化抽取失败: {e}", exc_info=True)
            return GraphAnalysis(vertices=[], edges=[])
        
    async def merge_and_ingest_graph(
        self, 
        chunk_id: str, 
        file_id: str, 
        extracted_relations: list[dict[str, Any]]
    ) -> None:
        """文档解析管道调用的核心入口，采用统一的懒加载 Session 管理事务边界。"""
        if not extracted_relations:
            return

        async with self.oracle_session as session:
            repo = GraphRepository(session)
            
            # --- 优化点 1: 提取出当前 Chunk 中所有独特的实体，避免同批内并发冲突 ---
            unique_vertices = {}
            for rel in extracted_relations:
                src_id = self._generate_md5_id(rel["source_name"], rel["source_type"])
                if src_id not in unique_vertices:
                    unique_vertices[src_id] = {
                        "name": rel["source_name"],
                        "type": rel["source_type"],
                        "desc": rel.get("source_desc") or ""
                    }
                
                tgt_id = self._generate_md5_id(rel["target_name"], rel["target_type"])
                if tgt_id not in unique_vertices:
                    unique_vertices[tgt_id] = {
                        "name": rel["target_name"],
                        "type": rel["target_type"],
                        "desc": rel.get("target_desc") or ""
                    }

            # --- 优化点 2: 串行处理节点融合，彻底消除行锁冲突与重复向量计算 ---
            vertex_id_map = {}
            for v_id, v_info in unique_vertices.items():
                try:
                    actual_id = await self._process_vertex_fusion(
                        repo=repo,
                        vertex_id=v_id,
                        name=v_info["name"],
                        v_type=v_info["type"],
                        new_desc=v_info["desc"],
                        chunk_id=chunk_id
                    )
                    vertex_id_map[v_id] = actual_id
                except Exception as e:
                    logger.error(f"[GraphIngestion] 实体融合失败: {v_info['name']}, 错误: {str(e)}")

            # --- 优化点 3: 并发处理边和映射（核心修复：内部重构路由直接绑定至单一底层存储方案） ---
            async def _process_single_edge(rel: dict[str, Any]) -> None:
                try:
                    src_raw_id = self._generate_md5_id(rel["source_name"], rel["source_type"])
                    box_raw_id = self._generate_md5_id(rel["target_name"], rel["target_type"])
                    
                    src_id = vertex_id_map.get(src_raw_id)
                    tgt_id = vertex_id_map.get(box_raw_id)
                    if not src_id or not tgt_id:
                        return

                    # 【核心修复】：直接调用对齐后的仓储方法，拒绝调用不存在的方法
                    await repo.upsert_edge_with_map(
                        edge_id=self._generate_md5_id(src_id, tgt_id, rel["relation_type"]),
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=rel["relation_type"],
                        chunk_id=chunk_id,
                        file_id=file_id,
                        attributes=rel.get("relation_attributes") or {}
                    )
                except Exception as e:
                    logger.error(f"[GraphIngestion] 处理图关系单条边录入失败: {rel}, 错误: {str(e)}")

            await asyncio.gather(*[_process_single_edge(r) for r in extracted_relations])
            await session.commit()
            logger.info(f"Successfully processed and committed graph network for chunk {chunk_id}")

    async def _process_vertex_fusion(
        self, repo: Any, vertex_id: str, name: str, v_type: str, new_desc: str, chunk_id: str
    ) -> str:
        """百科体融合逻辑：智能检测老状态 -> LLM生成式覆盖融合 -> 矢量更新过滤机制"""
        from core.config.settings import get_prompt_config
        from agent.prompt import default_prompt
        from utils.sanitize import sanitize_dict_for_oracle_json

        new_desc = new_desc.strip()
        existing_vertex = await repo.get_vertex_by_id(vertex_id)
        
        final_desc = new_desc
        final_vector = None
        attributes = {
            "last_updated_by_chunk": chunk_id,
            "vertex_id": vertex_id,
            "id": vertex_id,
            "vertex_name": name,
            "name": name
        }

        has_valid_vertex = False
        if existing_vertex:
            if isinstance(existing_vertex, dict) and existing_vertex:
                has_valid_vertex = True
            elif hasattr(existing_vertex, "vertex_id") or hasattr(existing_vertex, "description"):
                has_valid_vertex = True

        if has_valid_vertex:
            def get_attr(obj: Any, key: str, default: Any = None) -> Any:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            old_description = get_attr(existing_vertex, "description")
            old_vector = get_attr(existing_vertex, "name_vector")
            old_attributes = get_attr(existing_vertex, "attributes")

            if old_description and final_desc and old_description != final_desc:
                try:
                    fusion_prompt = await default_prompt.generate(
                        get_prompt_config().graph_vertex_fusion,
                        name=name,
                        v_type=v_type,
                        old_desc=old_description,
                        new_desc=final_desc
                    )
                    llm_response = await self.model_client.get_llm_answer(
                        model_name=self.llm_model,
                        prompt=fusion_prompt
                    )
                    if llm_response and llm_response.strip():
                        final_desc = llm_response.strip()
                except Exception as llm_err:
                    logger.warning(f"[GraphIngestion] 调用 LLM 融合描述失败，降级沿用老描述. 错误: {llm_err}")
                    final_desc = old_description
                
            if old_description and final_desc == old_description:
                final_vector = old_vector
                if old_attributes and isinstance(old_attributes, dict):
                    attributes.update(old_attributes)
                    attributes["vertex_id"] = vertex_id
                    attributes["id"] = vertex_id
            else:
                final_vector = await self.model_client.get_embedding(self.embedding_model, f"{name}: {final_desc}")
        else:
            final_vector = await self.model_client.get_embedding(self.embedding_model, f"{name}: {final_desc}")

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