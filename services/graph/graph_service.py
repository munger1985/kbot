import asyncio
import hashlib
import time
from typing import Any
from loguru import logger

from core.database.oracle import get_session
from core.exceptions import *
from core.config import get_prompt_config
from dao.repositories import GraphRepository, AgentConfRepository
from .schemas import GraphAnalysis
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.exceptions import handle_exception
from services.search import TxtBaseSearchResult, GraphBaseSearch
from services.kb import ModelParams
from services.basic import AgentService


class GraphService:
    """图谱导入服务，负责管理知识图谱实体与关系的导入。"""

    def __init__(self):
        self.model_client = AIModelClient()
        self.graph_search = GraphBaseSearch()
        self.agent_service = AgentService()

    @property
    def db_session(self):
        return get_session()

    def _generate_md5_id(self, *args: str) -> str:
        """全局一致的 ID 哈希生成器，确保相同实体的 ID 天然对齐"""
        content = "_".join([str(arg).strip().lower() for arg in args])
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def extract_triplets(self, user_input_text: str, llm_model_name: str, 
                               domain_name: str, domain_description: str,
                               kb_name: str, kb_description: str) -> GraphAnalysis:
        """利用大模型从输入文本中抽取知识图谱实体与关系"""
        try:
            prompt = await default_prompt.generate(
                get_prompt_config().graph_extractor, 
                text=user_input_text,
                domain_name=domain_name,
                domain_description=domain_description,
                kb_name=kb_name,
                kb_description=kb_description
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

    async def get_graph_context(
        self,
        db_session,
        agent_id: int,
        question: str,
        vertex_names: list[str],
        security_level: int,
        model_params: ModelParams,
        tags: list[str] = []
    ) -> tuple[list[TxtBaseSearchResult], list[float]]:
        """
        核心业务流水线：问题向量化 -> 并行图谱子图游走召回 -> 结果剪枝聚合
        """
        # 1. 向量化（如有需要，某些图谱混合检索可能用到查询向量，对齐 DocService）
        query_vec = await self._get_embedding(question, model_params.txt_embedding_model)

        # 2. 获取该智能体挂载的知识库配置（支持 1个 Agent 挂载多个图谱知识库）
        conf_repo = AgentConfRepository(db_session)
        agent_confs = await conf_repo.get_by_agent(agent_id)

        # 3. 多路并行图谱检索
        logger.info(f"开始为智能体 {agent_id} 执行图谱空间网络检索，安全等级：{security_level}")
        start_time = time.time()
        
        graph_tasks = []
        for conf in agent_confs:
            # 优先从配置表读取图谱专属参数，若无则常数兜底
            search_top_k = int(conf.search_top_k or 5)
            # 假设图深度存储在扩展字段中，或默认取 2 
            max_depth = int(getattr(conf, "max_depth", 2) or 2)
            tool_weight = float(conf.tool_weight or 1.2)

            graph_tasks.append(self.graph_search.search_by_graph(
                kb_id=int(conf.kb_id),
                vertex_names=vertex_names,
                search_top_k=search_top_k,
                max_depth=max_depth,
                weight=tool_weight,
                security_level=int(security_level)
            ))

        # 执行并行分布式拓扑网络游走
        raw_results = await asyncio.gather(*graph_tasks, return_exceptions=True)
        retrieved_results = []
        
        for i, res in enumerate(raw_results):
            current_kb = agent_confs[i].kb_id if i < len(agent_confs) else "Unknown"
            if isinstance(res, Exception):
                logger.error(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 执行失败：{res}")
                continue
            elif isinstance(res, dict):
                # search_by_graph 返回 {"graph_result": [...]}
                graph_items = res.get("graph_result", [])
                if isinstance(graph_items, list):
                    retrieved_results.extend(graph_items)
                    logger.info(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 成功返回 {len(graph_items)} 条拓扑路径记录")
                else:
                    logger.warning(f"图谱知识库任务 {i} (KB_ID: {current_kb}) graph_result 不是列表类型")
            elif isinstance(res, list):
                # 兼容旧版直接返回列表的接口
                retrieved_results.extend(res)
                logger.info(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 成功返回 {len(res)} 条拓扑路径记录")
            else:
                logger.warning(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 返回格式异常: {type(res)}")

        # 4. 图谱层面的排序/剪枝逻辑（此处可根据权重、距离直接过滤，对齐重排占位）
        final_results = self._apply_graph_filter(retrieved_results)

        logger.info(f"图谱空间检索完成，耗时：{time.time() - start_time:.2f}s，最终合并 {len(final_results)} 条实体关系边")
        return final_results, query_vec

    async def _get_embedding(self, content: str, model_name: str) -> list[float]:
        content = content.strip() if content else ""
        if not content:
            raise ParamValueError(f"图谱相关检索内容不能为空")
        vec = await self.model_client.get_embedding(model_name, content)
        if not vec:
            raise InternalServerError("图谱检索前置嵌入向量生成失败")
        return vec

    def _apply_graph_filter(self, results: list[TxtBaseSearchResult]) -> list[TxtBaseSearchResult]:
        # 根据图节点或边的权值进行基本排序，确保下游 Reasoning 层能拿到关联度最高的实体属性
        results.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
        return results
    

    async def align_vertices_by_embedding(
        self,
        keywords: list[str],
        agent_id: int,
        top_k: int = 3
    ) -> list[str]:
        """
        [Service 层封装]
        多关键词语义实体消歧对齐算子。
        并发计算多路关键词的 Embedding 向量，并并发驱动 Oracle 向量引擎完成近义碰撞与交叉去重。
        
        :param keywords: 规划层或上下文提取出的模糊关键词/白话短语列表
        :param agent_id: 知识库/智能体隔离 ID
        :param top_k: 每个关键词辐射匹配的实体上限数量
        :return: 100% 存在于图谱中的真实节点名称列表
        """
        if not keywords:
            return []

        # 清洗初始输入的无意义、空字符串
        valid_keywords = [kw.strip() for kw in keywords if kw and len(kw.strip()) >= 2]
        if not valid_keywords:
            return keywords
        
        # 根据 agent_id 获取embedding_model
        model_params = await self.agent_service.get_agent_model_params(agent_id)
        embedding_model = model_params.txt_embedding_model
        
        # 定义内部子任务：负责单词的 [向量转换 -> 数据库近邻查询] 闭环
        async def process_single_keyword(kw: str, graph_repo: GraphRepository) -> list[str]:
            try:
                # 1. 生成关键词的向量表示
                kw_embedding = await self._get_embedding(content=kw, model_name=embedding_model)
                if not kw_embedding:
                    return [kw] # 向量化失败则保留原词，交由字面量硬碰防御

                # 2. 调用数据库查询近邻实体

                db_hits = await graph_repo.get_vertex_names_by_embedding(
                    kb_id=agent_id,
                    keyword_embedding=kw_embedding,
                    top_k=top_k
                )
                
                # 如果这个关键词在向量检索中沉寂（图谱里可能还没有这类实体点），原样返还以提供字面量硬匹配的机会
                return db_hits if db_hits else [kw]
                
            except Exception as ex:
                logger.warning(f"[GraphService] 单个关键词 '{kw}' 执行语义消歧碰撞失败，降级保持原样。异常原因: {ex}")
                return [kw]

        aligned_names_set = set()
        async with self.db_session as session:
            graph_repo = GraphRepository(session)

            try:
                tasks = [process_single_keyword(kw, graph_repo) for kw in valid_keywords]
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)

                # 3. 内存聚合与跨路去重
                for res_list in completed_results:
                    if isinstance(res_list, list):
                        for name in res_list:
                            aligned_names_set.add(name)

                final_aligned_vertices = list(aligned_names_set)
                
                logger.info(f"[GraphService] 实体消歧对齐完成。原始关键词: {valid_keywords} -> 融合去重后图节点: {final_aligned_vertices}")
                return final_aligned_vertices

            except Exception as global_err:
                logger.error(f"[GraphService] 调度全局向量实体对齐引发未知崩溃: {global_err}，安全降级回原始词", exc_info=True)
                return keywords