import os
import uuid
from loguru import logger
from datetime import datetime
from ..parser_schema import DocParserParams, FileParams
from .docling_service import ParserService
from .txt_to_md import TxtToMarkdownParser
from dao.repositories import FileRepository, KBRepository, TxtChunkRepository
from dao.entities import TxtChunkEntity
from core.database.oracle import get_session
from core.config.settings import get_app_config
from core.dictionary import FileStatus, ProcessPriority
from core.exceptions import DataNotFoundException, DatabaseException
from utils.clients import AIModelClient
from services.ai_model import AIModelService



class FileProcessor:
    """文件处理类，负责文件解析和处理的业务逻辑"""
    def __init__(self):
        self.parser = ParserService()
        self.model_client = AIModelClient()
        self.model_service = AIModelService()
        self.app_id = get_app_config().app_id
    
    @property
    def oracle_session(self):
        # 只有当第一次调用 service.add() 或 service.get() 时才会触发
        return get_session()
    
    async def get_pending_files(self) -> list[tuple[int, float, FileParams]]:
        """
        从数据库获取待处理的文件。
        仅负责拉取数据，不在此处更新状态为 PARSING。
        
        返回:
            包含(优先级, 时间戳, 文件参数)元组的列表
        """
        result = []
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            kb_repo = KBRepository(session)
            try:
                files = await file_repo.get_by_status(FileStatus.APPROVED)
            except DataNotFoundException as e:
                logger.warning(e.message)
                return result
            except DatabaseException as e:
                logger.error(f"数据库查询错误: {str(e)}")
                return result
            except Exception as e:
                logger.exception(f"获取待处理文件失败: {str(e)}")
                return result

            for file in files:
                if not file.parser_params:
                    msg = f"文件 {file.id} 的解析器参数为空，跳过处理"
                    logger.warning(msg)
                    await self._update_file_status(file.id, FileStatus.PARSE_FAILED, msg)
                    continue
                
                models = await kb_repo.get_model_by_id(file.kb_id)
                if not models:
                    msg = f"知识库 {file.kb_id} 未配置模型，跳过处理"
                    logger.warning(msg)
                    await self._update_file_status(file.id, FileStatus.PARSE_FAILED, msg)
                    continue

                txt_embed_model_id = models.get("txt_embed_model_id", None)
                img2txt_model_id = models.get("img2txt_model_id", None)
                embed_model = None
                vlm_model = None
                if txt_embed_model_id:
                    embed_model = await self.model_service.get_model_name_by_id(txt_embed_model_id)
                if img2txt_model_id:
                    vlm_model = await self.model_service.get_model_name_by_id(img2txt_model_id)

                # 抽取图片保存在文件所在目录下，以文件ID命名的文件夹中
                dir_name = os.path.dirname(file.file_path)
                image_dir = os.path.join(dir_name, file.file_id)

                # 获取vlm的prompt
                use_vlm = file.parser_params.get("use_vlm", False)
                vlm_prompt = file.parser_params.get("vlm_prompt", None)
                
                # parser_params 从数据库取出后是 dict，需要转换为 DocParserParams 对象
                doc_params = DocParserParams(
                    chunk_size=file.parser_params.get("chunk_size", 512),
                    overlap=file.parser_params.get("overlap", 20),
                    min_chunk_len=file.parser_params.get("min_chunk_len", 10),
                    generate_picture_images=file.parser_params.get("generate_picture_images", False),
                    image_scale=file.parser_params.get("image_scale", 1.0),
                    image_dir=image_dir,
                    do_ocr=file.parser_params.get("do_ocr", False),
                    ocr_engine=file.parser_params.get("ocr_engine", None),
                    use_vlm=use_vlm,
                    vlm_model=vlm_model,
                    vlm_prompt=vlm_prompt
                )

                file_params = FileParams(
                    file_id=file.id,
                    kb_id=file.kb_id,
                    file_path=file.file_path if file.file_path is not None else "",
                    file_ext=file.file_ext,
                    priority = file.process_priority or ProcessPriority.MEDIUM.value,
                    security_level = file.security_level or 1,
                    parser_params=doc_params,
                    biz_metadata=file.biz_metadata if file.biz_metadata is not None else {},
                    txt_embed_model=embed_model
                )

                timestamp = datetime.now().timestamp()  # 获取当前时间戳
                # 添加到结果列表
                result.append((file_params.priority, timestamp, file_params))
                logger.info(f"已添加文件到处理队列: {file_params.file_path} (优先级: {ProcessPriority(file_params.priority)})")
                
            return result


    async def process_file(self, file_params: FileParams):
        """
        处理文件的入口方法
        
        参数:
            file_params: 文件参数对象
        """
        # 1. 进入此方法说明 Worker 已经拿到任务了，此时更新状态
        await self._update_file_status(file_params.file_id, FileStatus.PARSING, "Worker 接收任务，准备解析")
        
        # 检查文件是否存在
        if not await self._check_file(file_params):
            return
        
        try:
            logger.info(f"开始处理文件: {file_params.file_path}...")
            chunks = []

            # 处理文件内容
            if file_params.file_path.endswith(".txt"):
                # 因为docling不支持直接解析txt文件，所以先转换为md
                file_content = TxtToMarkdownParser().process(file_params.file_path)
                new_file_path = file_params.file_path.replace(".txt", ".md")
                
                # 将转换后的md内容写入新文件
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                logger.info(f"已将txt文件转换为md文件: {new_file_path}")
                
                # 更新文件路径为新的md文件路径
                file_params.file_path = new_file_path
            
            # 调用 Docling 处理文件
            result = await self.parser.parse_file(
                file_path=file_params.file_path,
                parser_params=file_params.parser_params,
                output_format="chunks" # 指定输出格式为 chunks
            )

            # 构造落库结果集
            if isinstance(result, list):
                embeddings = await self._get_embeddings(result, file_params)
                if not embeddings:
                    logger.error(f"文件 {file_params.file_path} 解析结果为空或维度为0")
                    # 更新文件状态为处理失败
                    await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果为空或维度为0")
                    return
                else:
                    # 保存chunks
                    await self._save_chunks(file_params.kb_id, file_params.file_id, embeddings)
            else:
                logger.error(f"文件 {file_params.file_path} 解析结果不是期望的列表格式")
                # 更新文件状态为处理失败
                await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果不是期望的列表格式")
        
        except Exception as e:
            msg = f"处理文件 {file_params.file_id} 时发生错误: {str(e)}"
            logger.error(msg)
            # 更新文件状态为处理失败
            await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        
    async def _update_file_status(self, file_id: str, status: FileStatus, message: str) -> None:
        """
        更新文件状态辅助方法

        Args:
            file_id: 文件ID
            status: 文件状态
            message: 状态信息
        """
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            await file_repo.update_file_status(
                file_id=file_id,
                status=status,
                log_msg=message
            )

    async def _get_embeddings(self, parser_results: list[dict], file_params: FileParams) -> list[TxtChunkEntity]:
        """
        获取文本的嵌入向量并封装为 TxtChunk 实体
        
        Args:
            parser_results: Docling 解析后的原始 chunk 列表 (包含 path_names, structure_level 等)
            file_params: 包含 kb_id, file_id, biz_metadata, security_level 等业务信息
            
        Returns:
            list[TxtChunk]: 带有向量和完整路径基因的 chunk 列表
        """
        model = file_params.txt_embed_model
        if not model:
            logger.error(f"知识库 {file_params.kb_id} 未配置文本嵌入模型")
            return []
        
        if not parser_results:
            return []

        # 1. 预先准备所有元数据
        all_texts = [item["content"] for item in parser_results]
        
        # 2. 定义微批次大小 (Micro-batch size)
        # 根据经验，32-64 是兼顾并发与稳定性的平衡点
        micro_batch_size = 32 
        all_embeddings = []

        try:
            # 3. 分片循环处理
            for i in range(0, len(all_texts), micro_batch_size):
                batch_texts = all_texts[i : i + micro_batch_size]
                
                logger.info(f"正在处理第 {i//micro_batch_size + 1} 组 Embedding, 进度: {i}/{len(all_texts)}")
                
                # 调用 Embedding 微服务
                # 这里如果单个 batch 失败，可以考虑加一个简单的 retry 机制
                response = await self.model_client.call_embedding_model(
                    model_name=model, 
                    texts=batch_texts, 
                    batch_size=len(batch_texts) 
                )
                
                if response:
                    all_embeddings.extend([res.embedding for res in response])
                else:
                    raise Exception(f"微批次 {i} 返回结果为空，可能 Embedding 微服务内部异常")

            # 4. 验证数量是否对齐
            if len(all_embeddings) != len(all_texts):
                raise Exception(f"向量数量对齐失败: 文本 {len(all_texts)} vs 向量 {len(all_embeddings)}")

            # 5. 组装最终实体列表
            chunks = []
            for i, (text, emb) in enumerate(zip(all_texts, all_embeddings)):
                item = parser_results[i]
                unique_id = str(uuid.uuid4())
                
                chunk = TxtChunkEntity(
                    chunk_id=unique_id,
                    kb_id=file_params.kb_id,
                    file_id=file_params.file_id,
                    content=text,
                    embedding=emb,
                    path_names=item.get("path_names", []),
                    structure_level=item.get("structure_level", 0),
                    chunk_type=item.get("chunk_type", "text"),
                    chunk_metadata=item.get("metadata", {}),
                    biz_metadata=file_params.biz_metadata,
                    security_level=file_params.security_level,
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"分片获取 Embedding 失败: {str(e)}")
            return []
        
    async def _save_chunks(self, kb_id: int, file_id: str, chunks: list[TxtChunkEntity]):
        """
        保存chunks到数据库（包含错误处理）

        Args:
            kb_id: 知识库ID
            file_id: 文件ID
            chunks: 文本片段列表
        """
        async with self.oracle_session as session:
            chunk_repo = TxtChunkRepository(session)
            try:
                # 1. 保存文本块
                await chunk_repo.create(chunks=chunks)
                # 2. 更新文件状态为已解析
                await self._update_file_status(file_id, FileStatus.PARSED, f"成功保存 {len(chunks)} 个 chunks")
                logger.info(f"成功保存 {len(chunks)} 个 chunks")
            except Exception as e:
                msg = f"保存 chunks 时发生异常: {str(e)}"
                logger.error(msg)
                await self._update_file_status(file_id, FileStatus.PARSE_FAILED, msg)

    async def _check_file(self, file_params: FileParams) -> bool:
        """
        检查文件嵌入模型和文件存在性
        
        Args:
            file_params: 文件参数对象
        """
        try:
            # 检查文本嵌入模型是否指定
            if file_params.txt_embed_model is None:
                msg = f"知识库 {file_params.kb_id} 未指定文本嵌入模型"
                logger.error(msg)
                # 更新文件状态为处理失败
                await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return False

            # 检查文件是否存在
            if not os.path.exists(file_params.file_path):
                logger.error(f"文件路径不存在: {file_params.file_path}")
                await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件路径不存在")
                return False
            
            return True
                
        except Exception as e:
            msg = f"处理文本文件 {file_params.file_id} 时发生错误: {str(e)}"
            logger.error(msg)
            await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False