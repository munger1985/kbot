import os
import uuid
from loguru import logger
from datetime import datetime
from ..parser_schema import DocParserParams, FileParams, ChunkResult
from .docling_service import ParserService
from .txt_to_md import TxtToMarkdownParser
from dao.repositories import FileRepository, TxtChunkRepository
from dao.entities import TxtChunkEntity
from core.database import db_instance
from core.config.settings import get_prompt_config
from core.dictionary import FileStatus, ProcessPriority
from core.exceptions import DataNotFoundException, DatabaseException
from utils.clients import AIModelClient
from services.basic import AIModelService, DomainService
from agent.prompt import default_prompt
from ..parsers.metadata_extractor import MetadataExtractor
from dao.repositories.doc_meta_repo import DocMetaRepository
from dao.repositories.doc_relation_repo import DocRelationRepository


class FileProcessor:
    """文件处理核心类，负责文档解析和向量嵌入全流程业务逻辑
    
    核心职责：
    1. 从数据库获取待处理文件列表
    2. 调用解析服务处理不同格式文档
    3. 生成文本分块的向量嵌入
    4. 保存分块数据并更新文件处理状态
    """
    def __init__(self):
        """初始化文件处理器，加载依赖服务"""
        self.parser = ParserService()          # 文档解析服务
        self.model_client = AIModelClient()    # AI模型客户端
        self.model_service = AIModelService()  # AI模型管理服务
        self.domain_service = DomainService()  # 业务域管理服务
    
    @property
    def db_session(self):
        """Oracle数据库会话的懒加载属性
        
        特性：
        - 首次访问时才创建会话（如调用 service.add()/service.get()）
        - 避免初始化时创建无用的数据库连接
        """
        return db_instance().get_session()
    
    async def get_pending_files(self) -> list[tuple[int, float, FileParams]]:
        """从数据库获取待处理文件列表
        
        仅执行数据查询操作，不修改文件状态（PARSING状态在worker接任务时更新）
        
        Returns:
            元组列表，格式为 (优先级数值, 时间戳, 文件参数对象)
        """
        result = []
        async with self.db_session as session:
            file_repo = FileRepository(session)  # 文件数据仓库
            
            try:
                # 查询状态为已审核（APPROVED）的待解析文件
                files = await file_repo.get_by_status(FileStatus.APPROVED)
            except DataNotFoundException as e:
                logger.warning(e.message)
                return result
            except DatabaseException as e:
                logger.error(f"数据库查询异常：{str(e)}")
                return result
            except Exception as e:
                logger.exception(f"获取待处理文件失败：{str(e)}")
                return result

            for file in files:
                # 跳过解析参数为空的文件
                if not file.chunk_parser:
                    msg = f"文件 {file.file_id} 解析参数为空，跳过处理"
                    logger.warning(msg)
                    await self._update_file_status(file.file_id, FileStatus.FAILED, msg)
                    continue

                txt_embed_model = file.chunk_parser.get("txt_embedding_model", None) # 文本嵌入模型
                llm_model = file.chunk_parser.get("llm_model", None) # LLM模型
                vlm_model = file.chunk_parser.get("vlm_model", None) # 视觉语言模型
                
                if not txt_embed_model:
                    msg = f"文件 {file.file_id} 解析参数缺少文本嵌入模型配置，无法生成向量，跳过处理"
                    logger.warning(msg)
                    await self._update_file_status(file.file_id, FileStatus.FAILED, msg)
                    continue

                if not llm_model:
                    msg = f"文件 {file.file_id} 解析参数缺少LLM模型，跳过处理"
                    logger.warning(msg)
                    await self._update_file_status(file.file_id, FileStatus.FAILED, msg)
                    continue

                # 抽取图片保存路径：文件所在目录下，以文件ID命名的子文件夹
                dir_name = os.path.dirname(file.file_path)
                image_dir = os.path.join(dir_name, file.file_id)

                # 获取VLM提示词配置
                img2txt_prompt = file.chunk_parser.get("img2txt_prompt", None)
                if not img2txt_prompt:
                    img2txt_prompt = await default_prompt.generate(get_prompt_config().image2text)

                # 将数据库中的字典类型解析参数转换为DocParserParams对象
                doc_params = DocParserParams(
                    chunk_size=file.chunk_parser.get("chunk_size", 800),
                    min_chunk_len=file.chunk_parser.get("min_chunk_len", 100),
                    generate_picture_images=file.chunk_parser.get("generate_picture_images", True),
                    image_scale=file.chunk_parser.get("image_scale", 2.0),
                    image_dir=image_dir,
                    do_ocr=file.chunk_parser.get("do_ocr", False),
                    ocr_engine=file.chunk_parser.get("ocr_engine", None),
                    ocr_model=file.chunk_parser.get("ocr_model", None),
                    vlm_model=vlm_model,
                    llm_model=llm_model,
                    img2txt_prompt=img2txt_prompt,
                    enable_layout_clustering=file.chunk_parser.get("enable_layout_clustering", True),
                    enable_page_span_stitch=file.chunk_parser.get("enable_page_span_stitch", True),
                    enable_doc_metadata=file.chunk_parser.get("enable_doc_metadata", True),
                    engine_mode=file.chunk_parser.get("engine_mode", "auto"),
                    enable_chunk_reflection=file.chunk_parser.get("enable_chunk_reflection", False),
                    visual_model=file.chunk_parser.get("visual_model", ""),
                    kb_id=file.kb_id,
                )

                # 构建队列用的文件参数对象
                file_params = FileParams(
                    file_id=file.file_id,                               # 文件唯一标识
                    kb_id=file.kb_id,                              # 所属知识库ID
                    file_path=file.file_path if file.file_path is not None else "",  # 文件路径
                    file_ext=file.file_ext,                        # 文件扩展名
                    priority=file.process_priority or ProcessPriority.MEDIUM.value,  # 处理优先级
                    security_level=file.security_level if file.security_level is not None else 1,  # 安全级别（0 是合法值）
                    parser_params=doc_params,                      # 解析参数对象
                    biz_metadata=file.biz_metadata if file.biz_metadata is not None else {},  # 业务元数据
                    txt_embed_model=txt_embed_model                # 文本嵌入模型名称
                )

                # 添加到结果列表（优先级、时间戳、文件参数）
                timestamp = datetime.now().timestamp()
                result.append((file_params.priority, timestamp, file_params))
                logger.info(f"文件已加入处理队列：{file_params.file_path} (优先级：{ProcessPriority(file_params.priority).name})")
                
            return result


    async def process_file(self, file_params: FileParams):
        """文件处理主入口
        
        核心流程：
        1. 更新文件状态为处理中
        2. 前置校验（文件存在性、模型配置）
        3. 特殊处理TXT文件（转换为MD）
        4. 调用Docling解析服务生成分块
        5. 生成文本嵌入向量
        6. 保存分块数据并更新状态
        
        Args:
            file_params: 包含所有处理配置的文件参数对象
        """
        # 更新文件状态为处理中（worker已接收到任务）
        await self._update_file_status(file_params.file_id, FileStatus.PARSING, "工作线程已接收任务，准备解析")
        
        # 前置校验（文件存在性、模型配置）
        if not await self._check_file(file_params):
            return
        
        try:
            logger.info(f"开始处理文件：{file_params.file_path}...")
            chunks = []

            # TXT文件特殊处理（Docling不直接支持TXT，先转换为MD）
            if file_params.file_path.endswith(".txt"):
                # TXT转Markdown
                file_content = TxtToMarkdownParser().process(file_params.file_path)
                new_file_path = file_params.file_path.replace(".txt", ".md")
                
                # 写入转换后的MD文件
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                logger.info(f"TXT文件已转换为MD：{new_file_path}")
                
                # 更新文件路径为新的MD文件
                file_params.file_path = new_file_path
            
            # 调用Docling解析服务（输出分块格式用于向量嵌入）
            result = await self.parser.parse_file(
                file_id=file_params.file_id,
                file_path=file_params.file_path,
                parser_params=file_params.parser_params,
                output_format="chunks"  # 指定输出分块格式
            )

            # 处理解析结果
            if isinstance(result, list):
                # 为解析后的分块生成向量嵌入
                embeddings = await self._get_embeddings(result, file_params)
                
                if not embeddings:
                    logger.error(f"文件 {file_params.file_path} 解析结果为空或零维度")
                    await self._update_file_status(
                        file_params.file_id, 
                        FileStatus.FAILED, 
                        "文件解析结果为空或零维度"
                    )
                    return
                else:
                    # 将带嵌入向量的分块保存到数据库
                    await self._save_chunks(file_params.kb_id, file_params.file_id, embeddings)
                
                # 提取文档元数据（替代原 Graph 提取）
                if file_params.parser_params.enable_doc_metadata:
                    logger.info(f"提取文档 {file_params.file_path} 的结构化元数据...")
                    try:
                        await self._save_doc_metadata(
                            file_params=file_params,
                            chunks=embeddings,
                        )
                        logger.success(f"文档元数据提取完成")
                    except Exception as e:
                        logger.error(f"文档元数据提取失败: {str(e)}")

                # 更新文件状态为已解析
                await self._update_file_status(
                    file_params.file_id, 
                    FileStatus.PARSED, 
                    f"成功保存 {len(chunks)} 个带嵌入向量的文本分块"
                )
                
            else:
                logger.error(f"文件 {file_params.file_path} 解析结果非预期的列表格式")
                await self._update_file_status(
                    file_params.file_id, 
                    FileStatus.FAILED, 
                    "文件解析结果非预期的列表格式"
                )
        
        except Exception as e:
            msg = f"处理文件 {file_params.file_id} 时发生异常：{str(e)}"
            logger.error(msg, exc_info=True)
            await self._update_file_status(file_params.file_id, FileStatus.FAILED, msg)
        
    async def _update_file_status(self, file_id: str, status: FileStatus, message: str) -> None:
        """数据库文件状态更新辅助方法
        
        Args:
            file_id: 文件唯一标识
            status: 新的文件状态枚举值
            message: 状态变更日志信息
        """
        async with self.db_session as session:
            file_repo = FileRepository(session)
            await file_repo.update_file_status(
                file_ids=[file_id],
                status=status,
                log_msg=message
            )

    async def rollback_parsing_files(self) -> int:
        """关闭时将 PARSING 状态的文件回滚为 APPROVED，防止卡在中间态"""
        from dao.repositories import FileRepository
        async with self.db_session as session:
            repo = FileRepository(session)
            files = await repo.get_by_status(FileStatus.PARSING)
            if not files:
                return 0
            ids = [f.id for f in files]
            await repo.update_file_status(
                file_ids=ids,
                status=FileStatus.APPROVED,
                log_msg="服务关闭，回滚解析状态"
            )
            logger.warning(f"回滚 {len(ids)} 个 PARSING → APPROVED 文件: {ids}")
            return len(ids)

    async def _get_embeddings(self, parser_results: list[ChunkResult], file_params: FileParams) -> list[TxtChunkEntity]:
        """为解析后的文本分块生成向量嵌入，并封装为TxtChunk实体
        
        处理流程：
        1. 校验嵌入模型配置
        2. 提取所有文本内容
        3. 微批次处理生成嵌入向量（避免API限流）
        4. 校验向量与文本数量一致性
        5. 封装为数据库实体对象
        
        Args:
            parser_results: Docling解析器返回的原始分块列表（含路径、层级等信息）
            file_params: 业务参数（知识库ID、文件ID、业务元数据、安全级别等）

        Returns:
            带嵌入向量和完整路径层级的分块实体列表
        """
        # 校验嵌入模型配置
        model = file_params.txt_embed_model
        if not model:
            logger.error(f"知识库 {file_params.kb_id} 未配置文本嵌入模型")
            return []

        if not parser_results:
            logger.warning("解析结果为空，跳过向量嵌入生成")
            return []

        # 1. ★ Contextual Retrieval: embed 上下文化文本 (search_helper)
        #    而非原始 content。search_helper 包含: 文档摘要 + 层级路径 + 标题 + 内容前缀
        all_texts = []
        for i, item in enumerate(parser_results):
            if not item.content:
                continue
            all_texts.append(item.search_helper or item.content)

        if not all_texts:
            logger.warning("解析结果为空，跳过向量嵌入生成")
            return []

        # 提取有效索引（与 all_texts 过滤条件一致）
        valid_indices = [i for i, item in enumerate(parser_results) if item.content]
        
        # 2. 配置微批次大小
        batch_size = await self.model_service.get_embedding_batch_size(embedding_model_name=model)
        micro_batch_size = batch_size or 10
        all_embeddings = []

        try:
            # 3. 微批次处理（避免API限流/超时）
            for i in range(0, len(all_texts), micro_batch_size):
                batch_texts = all_texts[i : i + micro_batch_size]

                logger.info(
                    f"处理嵌入批次 {i//micro_batch_size + 1}，"
                    f"进度：{i}/{len(all_texts)}"
                )

                # 调用嵌入服务（生产环境可添加重试逻辑）
                response = await self.model_client.call_embedding_model(
                    model_name=model,
                    texts=batch_texts,
                    batch_size=micro_batch_size
                )

                if response:
                    all_embeddings.extend([res.embedding for res in response])
                else:
                    raise Exception(f"批次 {i} 嵌入服务返回空响应，可能存在内部错误")

            # 4. 校验嵌入向量与文本数量一致性
            if len(all_embeddings) != len(all_texts):
                raise Exception(
                    f"嵌入向量与文本数量不匹配："
                    f"文本({len(all_texts)}) vs 向量({len(all_embeddings)})"
                )

            # 5. 构建带完整元数据的TxtChunkEntity对象
            chunks = []
            for i, (text, emb) in enumerate(zip(all_texts, all_embeddings)):
                original_idx = valid_indices[i]
                item = parser_results[original_idx]
                unique_id = str(uuid.uuid4())

                chunk = TxtChunkEntity(
                    chunk_id=unique_id,
                    chunk_num=item.chunk_num or 0,
                    chunk_type=item.chunk_type,
                    kb_id=file_params.kb_id,
                    file_id=file_params.file_id,
                    content=item.content,                         # 原始内容
                    header=item.header,
                    doc_summary=item.doc_summary,
                    search_helper=item.search_helper,
                    embedding=emb,                                # 从 search_helper 生成
                    chunk_metadata=item.metadata.model_dump(),
                    biz_metadata=file_params.biz_metadata,
                    security_level=file_params.security_level,
                    hierarchy_path=item.hierarchy_path,
                    hierarchy_depth=item.hierarchy_depth,
                    heading_level=item.heading_level,
                    parent_chunk_id=item.parent_chunk_id,
                    section_id=item.section_id,
                )
                chunks.append(chunk)

            logger.success(f"成功生成 {len(chunks)} 个文本嵌入向量")
            return chunks

        except Exception as e:
            logger.error(f"生成向量嵌入失败：{str(e)}", exc_info=True)
            return []

    async def _save_chunks(self, kb_id: int, file_id: str, chunks: list[TxtChunkEntity]):
        """将带嵌入向量的解析分块保存到 ParadeDB"""
        async with self.db_session as session:
            chunk_repo = TxtChunkRepository(session)
            await chunk_repo.create(chunks=chunks)
        logger.info(f"文件 {file_id} 已成功保存 {len(chunks)} 个文本分块")

    async def _check_file(self, file_params: FileParams) -> bool:
        """文件预处理校验（文件存在性 + 嵌入模型配置）
        
        Args:
            file_params: 文件参数对象
            
        Returns:
            校验通过返回True，失败返回False（并更新文件状态为失败）
        """
        try:
            # 校验嵌入模型配置
            if file_params.txt_embed_model is None:
                msg = f"知识库 {file_params.kb_id} 未配置文本嵌入模型"
                logger.error(msg)
                await self._update_file_status(file_params.file_id, FileStatus.FAILED, msg)
                return False

            # 校验文件物理存在性
            if not os.path.exists(file_params.file_path):
                msg = f"文件路径不存在：{file_params.file_path}"
                logger.error(msg)
                await self._update_file_status(file_params.file_id, FileStatus.FAILED, msg)
                return False
            
            return True
                
        except Exception as e:
            msg = f"校验文件 {file_params.file_id} 时发生异常：{str(e)}"
            logger.error(msg, exc_info=True)
            await self._update_file_status(file_params.file_id, FileStatus.FAILED, msg)
            return False

    async def _save_doc_metadata(self, file_params: FileParams,
                                  chunks: list[TxtChunkEntity]) -> None:
        """提取并保存文档结构化元数据（替代原 Graph 提取）。

        从已生成的 chunk 中提取文本快照，每个文档调用 1 次 LLM。
        同时注入运行时统计信息（chunk_count、page_count、biz_metadata）。
        """
        llm_model = file_params.parser_params.llm_model
        if not llm_model:
            logger.warning("未配置 LLM 模型，跳过文档元数据提取")
            return

        kb_id = file_params.kb_id
        file_id = file_params.file_id

        # 运行时统计：chunk 数量、页数、业务元数据
        chunk_count = len(chunks)
        page_count = 0
        for c in chunks:
            try:
                meta = getattr(c, 'chunk_metadata', None)
                if isinstance(meta, dict):
                    pn = meta.get("page_num", 0)
                else:
                    pn = 0
                if isinstance(pn, (int, float)) and pn > page_count:
                    page_count = int(pn)
            except Exception:
                pass
        # fallback: 从 doc_summary 取全局摘要作为 doc_abstract 兜底
        fallback_abstract = (chunks[0].doc_summary or "") if chunks else ""
        biz_metadata = file_params.biz_metadata or {}

        try:
            # 从 chunk 中提取文本快照（前 5000 字）
            text_snapshot = "\n".join([
                c.search_helper or c.content for c in chunks[:20]
            ])[:5000]
            logger.info(
                f"[DocMeta] 文本快照: {len(chunks)} chunks, "
                f"快照长度={len(text_snapshot)}, "
                f"第一条搜索助手={getattr(chunks[0], 'search_helper', None) if chunks else 'N/A'}, "
                f"第一条内容={chunks[0].content[:100] if chunks else 'N/A'}"
            )

            if len(text_snapshot) < 50:
                logger.warning(f"文档 {file_id} 文本快照不足 50 字 ({len(text_snapshot)}字)，跳过元数据提取")
                return

            extractor = MetadataExtractor(self.model_client)
            result = await extractor.extract_from_text(
                text_snapshot=text_snapshot,
                llm_model=llm_model,
                kb_id=kb_id,
                file_id=file_id,
            )
            if not result:
                return

            # 合并运行时统计 + LLM 提取的元数据
            meta = result["meta"]
            meta["chunk_count"] = chunk_count
            meta["page_count"] = page_count
            meta["biz_metadata"] = biz_metadata
            # LLM 可能返回空摘要，用全局摘要兜底
            if not meta.get("doc_abstract"):
                meta["doc_abstract"] = fallback_abstract

            # 保存元数据
            async with self.db_session as session:
                meta_repo = DocMetaRepository(session)
                await meta_repo.upsert(kb_id, file_id, meta)

                # 保存引用关系
                if result.get("relations"):
                    rel_repo = DocRelationRepository(session)
                    await rel_repo.batch_insert(result["relations"])

        except Exception as e:
            logger.error(f"[FileProcessor] 文档元数据提取失败: {e}")