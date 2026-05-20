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
            
            # --- 1: 提取出当前 Chunk 中所有独特的实体，避免同批内并发冲突 ---
            # 统一提取逻辑，同时包容大写和小写 Key，消灭第一关的 KeyError
            unique_vertices: dict[str, dict[str, str]] = {}
            for rel in extracted_relations:
                s_name = str(rel.get("source_name") or rel.get("SOURCE_NAME") or "")
                s_type = str(rel.get("source_type") or rel.get("SOURCE_TYPE") or "")
                s_desc = str(rel.get("source_desc") or rel.get("SOURCE_DESC") or "")

                t_name = str(rel.get("target_name") or rel.get("TARGET_NAME") or "")
                t_type = str(rel.get("target_type") or rel.get("TARGET_TYPE") or "")
                t_desc = str(rel.get("target_desc") or rel.get("TARGET_DESC") or "")

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
                        vertex_id=v_id,
                        name=v_info["name"],
                        v_type=v_info["type"],
                        new_desc=v_info["desc"],
                        chunk_id=chunk_id
                    )
                    vertex_id_map[v_id] = actual_id
                except Exception as e:
                    logger.error(f"[GraphIngestion] 实体融合失败: {v_info['name']}, 错误: {str(e)}")

            # --- 3: 串行处理边和映射（精准适配 Oracle 大小写感知） ---
            async def _process_single_edge(edge_dict: dict[str, Any]) -> None:
                try:
                    # 一步到位：提取并强转为明确类型，不留任何 Any | None 隐患
                    source_name = str(edge_dict.get("source_name") or edge_dict.get("SOURCE_NAME") or "")
                    source_type = str(edge_dict.get("source_type") or edge_dict.get("SOURCE_TYPE") or "")
                    target_name = str(edge_dict.get("target_name") or edge_dict.get("TARGET_NAME") or "")
                    target_type = str(edge_dict.get("target_type") or edge_dict.get("TARGET_TYPE") or "")
                    relation_type = str(edge_dict.get("relation_type") or edge_dict.get("RELATION_TYPE") or "")
                    
                    # 属性特殊处理，确保是 dict
                    raw_attrs = edge_dict.get("relation_attributes") or edge_dict.get("RELATION_ATTRIBUTES")
                    relation_attributes = raw_attrs if isinstance(raw_attrs, dict) else {}

                    # 严格拦截空核心字段
                    if not source_name or not target_name or not relation_type:
                        logger.warning(f"[GraphIngestion] 关系数据缺失关键核心字段，跳过: {edge_dict}")
                        return

                    # 1. 计算原始 MD5 ID (此时静态检查器 100% 确认入参为非空 str)
                    src_raw_id = self._generate_md5_id(source_name, source_type)
                    box_raw_id = self._generate_md5_id(target_name, target_type)
                    
                    # 2. 从映射表安全获取顶点融合后的真实 ID
                    src_id = vertex_id_map.get(src_raw_id)
                    tgt_id = vertex_id_map.get(box_raw_id)
                    
                    if not src_id or not tgt_id:
                        logger.warning(f"[GraphIngestion] 找不到顶点映射 ID，跳过边处理. SrcRaw: {src_raw_id}, TgtRaw: {box_raw_id}")
                        return

                    # 3. 严格匹配 repo.upsert_edge_with_map 签名
                    generated_edge_id = self._generate_md5_id(src_id, tgt_id, relation_type)
                    
                    await repo.upsert_edge_with_map(
                        edge_id=generated_edge_id,
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=relation_type,
                        chunk_id=chunk_id,
                        file_id=file_id,
                        attributes=relation_attributes
                    )
                except Exception as e:
                    logger.error(f"[GraphIngestion] 处理图关系单条边录入失败: {edge_dict}, 错误详情: {str(e)}", exc_info=True)

            # 保持标准的 for 循环顺序安全处理
            for r in extracted_relations:
                await _process_single_edge(r)

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
        
        if existing_vertex is not None:
            try:
                logger.info(f"[GraphProbe] 实体名称: {name}, 原始对象类型: {type(existing_vertex)}")
                if hasattr(existing_vertex, "keys"):
                    logger.info(f"[GraphProbe] 检测到类似字典结构，keys: {list(existing_vertex.keys())}")
                if hasattr(existing_vertex, "__dict__"):
                    logger.info(f"[GraphProbe] 检测到类/ORM结构，__dict__.keys: {list(existing_vertex.__dict__.keys())}")
                if hasattr(existing_vertex, "_mapping"):
                    logger.info(f"[GraphProbe] 检测到 SQLAlchemy RowMapping，映射 keys: {list(existing_vertex._mapping.keys())}")
            except Exception as probe_err:
                logger.warning(f"[GraphProbe] 探测日志输出本身报错: {probe_err}", exc_info=True)
                
        final_desc = new_desc
        final_vector = None
        
        attributes = {
            "last_updated_by_chunk": chunk_id
        }

        has_valid_vertex = False
        if existing_vertex:
            if isinstance(existing_vertex, dict) and existing_vertex:
                has_valid_vertex = True
            elif hasattr(existing_vertex, "vertex_id") or hasattr(existing_vertex, "description") or hasattr(existing_vertex, "_mapping"):
                has_valid_vertex = True

        if has_valid_vertex:
            # 强化版 get_attr：同时兼容小写、全大写，以及 SQLAlchemy 的 RowMapping 防御
            def get_attr(obj: Any, key: str, default: Any = None) -> Any:
                key_upper = key.upper()
                # 1. 如果是标准的 dict
                if isinstance(obj, dict):
                    return obj.get(key) if key in obj else obj.get(key_upper, default)
                # 2. 如果是 SQLAlchemy 的 RowMapping 或者是带 _mapping 的代理对象
                if hasattr(obj, "_mapping") and obj._mapping is not None:
                    return obj._mapping.get(key) if key in obj._mapping else obj._mapping.get(key_upper, default)
                # 3. 如果是标准 ORM 实体对象
                return getattr(obj, key, getattr(obj, key_upper, default))

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
                    cleaned_old_attrs = old_attributes.copy()
                    banned_keys = {"id", "vertex_id", "name", "vertex_name", "type", "vertex_type"}
                    for k in banned_keys:
                        cleaned_old_attrs.pop(k, None)
                        cleaned_old_attrs.pop(k.upper(), None) # 同步移除可能的大写遗留
                    
                    attributes.update(cleaned_old_attrs)
            else:
                final_vector = await self.model_client.get_embedding(self.embedding_model, f"{name}: {final_desc}")
        else:
            final_vector = await self.model_client.get_embedding(self.embedding_model, f"{name}: {final_desc}")

        # 在送入清理函数前，对 attributes 做一层极端 KeyError 异常拦截防御
        try:
            sanitized_attrs = sanitize_dict_for_oracle_json(attributes)
        except KeyError as json_ke:
            logger.warning(f"[GraphIngestion] sanitize_dict_for_oracle_json 触发了内部 Key 异常 (错误键: {json_ke})，启动降级清洗方案。")
            # 降级：只保留完全安全的本地确定的基础 kv，剔除引发报错的历史脏字典
            safe_attrs = {"last_updated_by_chunk": chunk_id}
            sanitized_attrs = sanitize_dict_for_oracle_json(safe_attrs)

        await repo.upsert_vertex(
            vertex_id=vertex_id,
            vertex_name=name,
            vertex_type=v_type,
            description=final_desc,
            attributes=sanitized_attrs,
            name_vector=final_vector
        )
        return vertex_id