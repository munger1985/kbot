import asyncio
import hashlib
from typing import Any
from loguru import logger

from core.database.oracle import get_session
from core.exceptions import *
from core.config import get_prompt_config
from dao.repositories import GraphRepository 
from .schemas import GraphAnalysis
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.exceptions import handle_exception


class GraphService:
    """图谱导入服务，负责管理知识图谱实体与关系的导入。"""

    def __init__(self):
        self.model_client = AIModelClient()

    @property
    def db_session(self):
        return get_session()

    def _generate_md5_id(self, *args: str) -> str:
        """全局一致的 ID 哈希生成器，确保相同实体的 ID 天然对齐"""
        content = "_".join([str(arg).strip().lower() for arg in args])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def extract_triplets(self, user_input_text: str, llm_model_name: str, domain_name: str, domain_description: str) -> GraphAnalysis:
        """利用大模型从输入文本中抽取知识图谱实体与关系"""
        try:
            prompt = await default_prompt.generate(
                get_prompt_config().graph_extractor, 
                text=user_input_text,
                domain_name=domain_name,
                domain_description=domain_description
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
        kb_id: int, 
        chunk_id: str, 
        file_id: str, 
        llm_model: str,
        embedding_model: str,
        extracted_relations: list[dict[str, Any]]
    ) -> None:
        """文档解析管道调用的核心入口，采用统一的懒加载 Session 管理事务边界。"""
        if not extracted_relations:
            return

        # =================================================================
        # 第一步：前置深度清洗，彻底拔除带引号的脏 Key 和脏 Value
        # =================================================================
        cleaned_relations: list[dict[str, Any]] = []
        for raw_rel in extracted_relations:
            if not isinstance(raw_rel, dict):
                continue
            
            clean_rel = {}
            for k, v in raw_rel.items():
                # 1. 拔除 Key 里的各种奇葩单双引号并转为小写
                clean_key = str(k).strip().strip("'").strip('"').strip().lower()
                
                # 2. 拔除字符串 Value 里的可能由大模型生成的内嵌引号
                if isinstance(v, str):
                    clean_val = v.strip().strip("'").strip('"').strip()
                elif isinstance(v, dict):
                    # 递归清理 attributes 内部的脏数据
                    clean_val = {
                        str(sub_k).strip().strip("'").strip('"').strip(): 
                        (sub_v.strip().strip("'").strip('"').strip() if isinstance(sub_v, str) else sub_v)
                        for sub_k, sub_v in v.items()
                    }
                else:
                    clean_val = v
                
                clean_rel[clean_key] = clean_val
            cleaned_relations.append(clean_rel)

        # 后续业务逻辑完全基于已清洗干净的 cleaned_relations 执行
        async with self.db_session as session:
            repo = GraphRepository(session)
            
            # --- 1: 提取出当前 Chunk 中所有独特的实体，避免同批内并发冲突 ---
            unique_vertices: dict[str, dict[str, str]] = {}
            for rel in cleaned_relations:
                # 此时所有的 Key 已经被清洗为标准小写，直接安心获取
                s_name = str(rel.get("source_name") or "")
                s_type = str(rel.get("source_type") or "Entity")
                s_desc = str(rel.get("source_desc") or "")

                t_name = str(rel.get("target_name") or "")
                t_type = str(rel.get("target_type") or "Entity")
                t_desc = str(rel.get("target_desc") or "")

                if s_name:
                    src_id = self._generate_md5_id(s_name, s_type)
                    if src_id not in unique_vertices:
                        unique_vertices[src_id] = {"name": s_name, "type": s_type, "desc": s_desc}
                
                if t_name:
                    tgt_id = self._generate_md5_id(t_name, t_type)
                    if tgt_id not in unique_vertices:
                        unique_vertices[tgt_id] = {"name": t_name, "type": t_type, "desc": t_desc}

            # --- 2: 串行处理节点融合，彻底消除行锁冲突与重复向量计算 ---
            vertex_id_map: dict[str, str] = {}
            for v_id, v_info in unique_vertices.items():
                try:
                    actual_id = await self._process_vertex_fusion(
                        repo=repo,
                        kb_id=kb_id,
                        vertex_id=v_id,
                        name=v_info["name"],
                        v_type=v_info["type"],
                        new_desc=v_info["desc"],
                        chunk_id=chunk_id,
                        llm_model=llm_model,
                        embedding_model=embedding_model
                    )
                    vertex_id_map[v_id] = actual_id
                except Exception as e:
                    logger.error(f"[GraphService] 实体融合失败: {v_info['name']}, 错误: {str(e)}")

            # --- 3: 串行处理边和映射 ---
            async def _process_single_edge(edge_dict: dict[str, Any], kb_id: int) -> None:
                try:
                    source_name = str(edge_dict.get("source_name") or "")
                    source_type = str(edge_dict.get("source_type") or "Entity")
                    target_name = str(edge_dict.get("target_name") or "")
                    target_type = str(edge_dict.get("target_type") or "Entity")
                    relation_type = str(edge_dict.get("relation_type") or "ASSOCIATE")
                    
                    relation_attributes = edge_dict.get("relation_attributes")
                    if not isinstance(relation_attributes, dict):
                        relation_attributes = {}

                    # 严格拦截空核心字段
                    if not source_name or not target_name:
                        return

                    # 从映射表安全获取顶点融合后的真实 ID
                    src_raw_id = self._generate_md5_id(source_name, source_type)
                    box_raw_id = self._generate_md5_id(target_name, target_type)
                    
                    src_id = vertex_id_map.get(src_raw_id)
                    tgt_id = vertex_id_map.get(box_raw_id)
                    
                    if not src_id or not tgt_id:
                        logger.warning(f"[GraphService] 找不到顶点映射 ID，跳过边处理. SrcRaw: {src_raw_id}, TgtRaw: {box_raw_id}")
                        return

                    generated_edge_id = self._generate_md5_id(src_id, tgt_id, relation_type)
                    
                    await repo.upsert_edge_with_map(
                        kb_id=kb_id,
                        edge_id=generated_edge_id,
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=relation_type,
                        chunk_id=chunk_id,
                        file_id=file_id,
                        attributes=relation_attributes
                    )
                except Exception as e:
                    logger.error(f"[GraphService] 处理图关系单条边录入失败: {edge_dict}, 错误详情: {str(e)}", exc_info=True)

            for r in cleaned_relations:
                await _process_single_edge(r, kb_id)

            await session.commit()
            logger.info(f"成功处理并提交图谱网络，块 ID: {chunk_id}")

    async def _process_vertex_fusion(
        self, repo: GraphRepository, kb_id: int, vertex_id: str, name: str, v_type: str, new_desc: str, chunk_id: str, llm_model: str, embedding_model: str
    ) -> str:
        """百科体融合逻辑"""

        new_desc = new_desc.strip()
        
        # 1. 探测 repo.get_vertex_by_id 阶段
        try:
            existing_vertex = await repo.get_vertex_by_id(kb_id=kb_id, vertex_id=vertex_id)
        except Exception as e:
            handle_exception(e, f"[GraphService] 获取顶点失败")
                
        final_desc = new_desc
        final_vector = None
        
        attributes = {
            "last_updated_by_chunk": chunk_id
        }

        has_valid_vertex = False
        if existing_vertex:
            if isinstance(existing_vertex, dict) and existing_vertex:
                has_valid_vertex = True
            elif hasattr(existing_vertex, "vertex_id") or hasattr(existing_vertex, "description"):
                has_valid_vertex = True

        if has_valid_vertex:
            # 2. 探测 get_attr 提取阶段
            def get_attr(obj: Any, key: str, default: Any = None) -> Any:
                try:
                    if isinstance(obj, dict):
                        val = obj.get(key, default)
                        return val
                    val = getattr(obj, key, default)
                    return val
                except Exception as e:
                    handle_exception(e, f"[GraphService] 获取属性失败: {key}")

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
                        model_name=llm_model,
                        prompt=fusion_prompt
                    )
                    if llm_response and llm_response.strip():
                        final_desc = llm_response.strip()
                except Exception as e:
                    logger.warning(f"[GraphService] 调用 LLM 进行百科体融合失败, 错误: {e}")
                    final_desc = old_description
                
            if old_description and final_desc == old_description:
                final_vector = old_vector
                if old_attributes and isinstance(old_attributes, dict):
                    try:
                        cleaned_old_attrs = old_attributes.copy()
                        banned_keys = {"id", "vertex_id", "name", "vertex_name", "type", "vertex_type"}
                        for key in banned_keys:
                            cleaned_old_attrs.pop(key, None)
                        attributes.update(cleaned_old_attrs)
                    except Exception as e:
                        handle_exception(e, f"[GraphService] 清理旧属性失败")
            else:
                final_vector = await self.model_client.get_embedding(embedding_model, f"{name}: {final_desc}")
        else:
            final_vector = await self.model_client.get_embedding(embedding_model, f"{name}: {final_desc}")

        # 4. 探测 最终 Repo 写入阶段
        try:
            await repo.upsert_vertex(
                kb_id=kb_id,
                vertex_id=vertex_id,
                vertex_name=name,
                vertex_type=v_type,
                description=final_desc,
                attributes=attributes,
                name_vector=final_vector
            )
            logger.success(f"[GraphService] 更新顶点成功: {name}")
        except Exception as e:
            handle_exception(e, f"[GraphService] 更新顶点失败")

        return vertex_id
    
    async def delete_graph_by_file(self, kb_id: int, file_ids: list[str]) -> None:
        """根据文件ID删除图谱"""
        try:
            async with self.db_session as session:
                repo = GraphRepository(session)
                await repo.delete_graph_by_file(kb_id=kb_id, file_ids=file_ids)
                logger.success(f"[GraphService] 删除图谱成功")
        except Exception as e:
            handle_exception(e, f"[GraphService] 根据文件ID删除图谱失败")

    async def delete_graph_by_kb(self, kb_id: int) -> None:
        """根据知识库ID删除图谱"""
        try:
            async with self.db_session as session:
                repo = GraphRepository(session)
                await repo.delete_graph_by_knowledge_base(kb_id=kb_id)
                logger.success(f"[GraphService] 删除图谱成功")
        except Exception as e:
            handle_exception(e, f"[GraphService] 根据知识库ID删除图谱失败")