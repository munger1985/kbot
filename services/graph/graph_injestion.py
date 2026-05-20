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
        self.model_client = AIModelClient()
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    @property
    def oracle_session(self):
        """Provides a database session instance following the core architecture pattern."""
        return get_session()

    def _generate_md5_id(self, *args: str) -> str:
        """全局一致的 ID 哈希生成器，确保相同实体的 ID 天然对齐"""
        content = "_".join([str(arg).strip().lower() for arg in args])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def extract_triplets(self, user_input_text: str, llm_model_name: str) -> GraphAnalysis:
        """
        利用大模型从输入文本中抽取知识图谱实体与关系（三元组）
        
        Args:
            user_input_text (str): 待抽取的原始文本或文档片段
            llm_model_name (str): 用于执行抽取的 LLM 模型名称
            
        Returns:
            GraphAnalysis: 包含抽取出的顶点和边数据的 Pydantic 模型实例
        """
        try:
            # 1. 获取配置好的图谱抽取 Prompt 配置，并将文本作为入参渲染
            prompt = await default_prompt.generate(
                get_prompt_config().graph_extractor, 
                text=user_input_text
            )
            
            # 2. 调用模型客户端，温度设为 0 以保证抽取结构和 JSON 格式的绝对稳定
            data = await self.model_client.get_llm_json(
                model_name=llm_model_name,
                prompt=prompt,
                temperature=0
            )

            # 3. 实例化图谱数据模型，Pydantic 会自动处理字段映射与结构校验
            return GraphAnalysis(**data)

        except Exception as e:
            logger.error(f"图谱结构化抽取失败: {e}", exc_info=True)
            # 发生异常时返回一个空的图谱对象，防止上层管道由于 NoneType 崩溃
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
                # 收集源节点
                src_id = self._generate_md5_id(rel["source_name"], rel["source_type"])
                if src_id not in unique_vertices:
                    unique_vertices[src_id] = {
                        "name": rel["source_name"],
                        "type": rel["source_type"],
                        "desc": rel.get("source_desc") or ""
                    }
                
                # 收集目标节点
                tgt_id = self._generate_md5_id(rel["target_name"], rel["target_type"])
                if tgt_id not in unique_vertices:
                    unique_vertices[tgt_id] = {
                        "name": rel["target_name"],
                        "type": rel["target_type"],
                        "desc": rel.get("target_desc") or ""
                    }

            # --- 优化点 2: 串行/线性处理节点融合，彻底消除行锁冲突与重复 LLM/Embedding 开销 ---
            vertex_id_map = {}  # 缓存结果便于后续构建边
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

            # --- 优化点 3: 节点就绪后，并发安全地处理边和溯源映射 ---
            async def _process_single_edge(rel: dict[str, Any]) -> None:
                try:
                    src_raw_id = self._generate_md5_id(rel["source_name"], rel["source_type"])
                    tgt_raw_id = self._generate_md5_id(rel["target_name"], rel["target_type"])
                    
                    # 确保节点成功写入才构建边
                    src_id = vertex_id_map.get(src_raw_id)
                    tgt_id = vertex_id_map.get(tgt_raw_id)
                    if not src_id or not tgt_id:
                        return

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
                    logger.error(f"[GraphIngestion] 处理图关系单条边录入失败: {rel}, 错误: {str(e)}", exc_info=True)

            # 并发处理当前文本块的所有边（边的 ID 包含类型，通常在该批内不容易冲突）
            await asyncio.gather(*[_process_single_edge(r) for r in extracted_relations])
            
            # 显式提交当前 Chunk 级别的图结构变更
            await session.commit()
            logger.info(f"Successfully processed and committed graph network for chunk {chunk_id}")

    async def _process_vertex_fusion(
        self, repo: Any, vertex_id: str, name: str, v_type: str, new_desc: str, chunk_id: str
    ) -> str:
        """百科体融合逻辑：查旧 -> 构造 Prompt 调用 get_llm_answer -> 重算 Vector -> Repo Upsert"""
        new_desc = new_desc.strip()
        
        # 1. 查库获取现有节点状态
        existing_vertex = await repo.get_vertex_by_id(vertex_id)
        
        final_desc = new_desc
        final_vector = None
        attributes = {"last_updated_by_chunk": chunk_id}

        if existing_vertex:
            # 如果已有描述，且新提取出的描述不为空、且不与老描述完全一致，触发在线百科融合
            if existing_vertex.description and final_desc and existing_vertex.description != final_desc:
                try:
                    # 3. 使用异步提示词管理器生成 Prompt（支持 DB 动态覆盖）
                    fusion_prompt = await default_prompt.generate(
                        get_prompt_config().graph_vertex_fusion,
                        name=name,
                        v_type=v_type,
                        old_desc=existing_vertex.description,
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
                    final_desc = existing_vertex.description
                
            # 增量判别：如果融合结果和库里一致，免去昂贵的 Embedding 开销
            if existing_vertex.description and final_desc == existing_vertex.description:
                final_vector = existing_vertex.name_vector
                if existing_vertex.attributes:
                    attributes.update(existing_vertex.attributes)
            else:
                # 产生新信息，重算向量
                final_vector = await self.model_client.get_embedding(self.embedding_model, f"{name}: {final_desc}")
        else:
            # 全新节点，直接计算标准向量
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