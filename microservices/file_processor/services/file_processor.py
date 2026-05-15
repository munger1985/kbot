import os
import uuid
from loguru import logger
from datetime import datetime
from ..parser_schema import DocParserParams, FileParams, ChunkResult
from .docling_service import ParserService
from .txt_to_md import TxtToMarkdownParser
from dao.repositories import FileRepository, TxtChunkRepository
from dao.entities import TxtChunkEntity
from core.database.oracle import get_session
from core.config.settings import get_app_config, get_prompt_config
from core.dictionary import FileStatus, ProcessPriority, ChunkType
from core.exceptions import DataNotFoundException, DatabaseException
from utils.clients import AIModelClient
from utils.sanitize import sanitize_dict_for_oracle_json
from services.basic.ai_model import AIModelService
from agent.prompt import default_prompt


class FileProcessor:
    """File processing class responsible for document parsing and embedding business logic"""
    def __init__(self):
        self.parser = ParserService()
        self.model_client = AIModelClient()
        self.model_service = AIModelService()
        self.app_id = get_app_config().app_id
    
    @property
    def oracle_session(self):
        """Lazy initialization of Oracle database session"""
        # Session is only created when first accessed (service.add()/service.get())
        return get_session()
    
    async def get_pending_files(self) -> list[tuple[int, float, FileParams]]:
        """
        Retrieve pending files from database for processing.
        Only fetches data, does NOT update status to PARSING here.
        
        Returns:
            List of tuples containing (priority, timestamp, file_params)
        """
        result = []
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            
            try:
                # Get files with APPROVED status (ready for parsing)
                files = await file_repo.get_by_status(FileStatus.APPROVED)
            except DataNotFoundException as e:
                logger.warning(e.message)
                return result
            except DatabaseException as e:
                logger.error(f"Database query error: {str(e)}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get pending files: {str(e)}")
                return result

            for file in files:
                # Skip files with empty parser parameters
                if not file.chunk_parser:
                    msg = f"File {file.file_id} has empty parser parameters, skipping processing"
                    logger.warning(msg)
                    await self._update_file_status(file.file_id, FileStatus.PARSE_FAILED, msg)
                    continue

                txt_embed_model = file.chunk_parser.get("txt_embedding_model", None) # 文本嵌入模型
                llm_model = file.chunk_parser.get("llm_model", None) # LLM模型
                vlm_model = file.chunk_parser.get("vlm_model", None) # 视觉语言模型
                
                if not txt_embed_model:
                    msg = f"File {file.file_id} missing txt_embedding_model, skip processing"
                    logger.warning(msg)
                    await self._update_file_status(file.file_id, FileStatus.PARSE_FAILED, msg)
                    continue

                if not llm_model:
                    msg = f"File {file.file_id} missing LLM model, skip processing"
                    logger.warning(msg)
                    await self._update_file_status(file.file_id, FileStatus.PARSE_FAILED, msg)
                    continue

                # Create image storage directory (file ID named folder in file's directory)
                dir_name = os.path.dirname(file.file_path)
                image_dir = os.path.join(dir_name, file.file_id)

                # VLM configuration
                use_vlm = file.chunk_parser.get("use_vlm", False)
                img2txt_prompt = file.chunk_parser.get("img2txt_prompt", None)
                if not img2txt_prompt:
                    img2txt_prompt = await default_prompt.generate(get_prompt_config().image2text)
                
                # Convert dict parser params to DocParserParams object
                doc_params = DocParserParams(
                    chunk_size=file.chunk_parser.get("chunk_size", 512),
                    overlap=file.chunk_parser.get("overlap", 20),
                    min_chunk_len=file.chunk_parser.get("min_chunk_len", 10),
                    generate_picture_images=file.chunk_parser.get("generate_picture_images", False),
                    image_scale=file.chunk_parser.get("image_scale", 1.0),
                    image_dir=image_dir,
                    do_ocr=file.chunk_parser.get("do_ocr", False),
                    ocr_engine=file.chunk_parser.get("ocr_engine", None),
                    use_vlm=use_vlm,
                    vlm_model=vlm_model,
                    llm_model=llm_model,
                    img2txt_prompt=img2txt_prompt
                )

                # Create FileParams object for queue
                file_params = FileParams(
                    file_id=file.file_id,
                    kb_id=file.kb_id,
                    file_path=file.file_path if file.file_path is not None else "",
                    file_ext=file.file_ext,
                    priority=file.process_priority or ProcessPriority.MEDIUM.value,
                    security_level=file.security_level or 1,
                    parser_params=doc_params,
                    biz_metadata=file.biz_metadata if file.biz_metadata is not None else {},
                    txt_embed_model=txt_embed_model
                )

                # Add to result list with priority and timestamp
                timestamp = datetime.now().timestamp()
                result.append((file_params.priority, timestamp, file_params))
                logger.info(f"Added file to processing queue: {file_params.file_path} (Priority: {ProcessPriority(file_params.priority)})")
                
            return result


    async def process_file(self, file_params: FileParams):
        """
        Main file processing entry point
        
        Args:
            file_params: File parameters object containing all processing config
        """
        # Update status to PARSING since worker has picked up the task
        await self._update_file_status(file_params.file_id, FileStatus.PARSING, "Worker received task, preparing to parse")
        
        # Pre-processing validation
        if not await self._check_file(file_params):
            return
        
        try:
            logger.info(f"Starting file processing: {file_params.file_path}...")
            chunks = []

            # Special handling for TXT files (convert to MD first since Docling doesn't support TXT directly)
            if file_params.file_path.endswith(".txt"):
                # Convert TXT to Markdown
                file_content = TxtToMarkdownParser().process(file_params.file_path)
                new_file_path = file_params.file_path.replace(".txt", ".md")
                
                # Write converted content to new MD file
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                logger.info(f"Converted TXT to MD file: {new_file_path}")
                
                # Update file path to new MD file
                file_params.file_path = new_file_path
            
            # Process file with Docling (output as chunks for embedding)
            result = await self.parser.parse_file(
                file_id=file_params.file_id,
                file_path=file_params.file_path,
                parser_params=file_params.parser_params,
                output_format="chunks"  # Specify chunk output format
            )

            # Process parsing results
            if isinstance(result, list):
                # Generate embeddings for parsed chunks
                embeddings = await self._get_embeddings(result, file_params)
                
                if not embeddings:
                    logger.error(f"File {file_params.file_path} parsing result is empty or zero-dimensional")
                    await self._update_file_status(
                        file_params.file_id, 
                        FileStatus.PARSE_FAILED, 
                        "File parsing result is empty or zero-dimensional"
                    )
                    return
                else:
                    # Save chunks with embeddings to database
                    await self._save_chunks(file_params.kb_id, file_params.file_id, embeddings)
            else:
                logger.error(f"File {file_params.file_path} parsing result is not expected list format")
                await self._update_file_status(
                    file_params.file_id, 
                    FileStatus.PARSE_FAILED, 
                    "File parsing result is not expected list format"
                )
        
        except Exception as e:
            msg = f"Error processing file {file_params.file_id}: {str(e)}"
            logger.error(msg, exc_info=True)
            await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        
    async def _update_file_status(self, file_id: str, status: FileStatus, message: str) -> None:
        """
        Helper method to update file status in database

        Args:
            file_id: File ID
            status: New file status
            message: Status log message
        """
        async with self.oracle_session as session:
            file_repo = FileRepository(session)
            await file_repo.update_file_status(
                file_ids=[file_id],
                status=status,
                log_msg=message
            )

    async def _get_embeddings(self, parser_results: list[ChunkResult], file_params: FileParams) -> list[TxtChunkEntity]:
        """
        Generate embeddings for parsed text chunks and package as TxtChunk entities

        Args:
            parser_results: Raw chunk list from Docling parser (contains path_names, structure_level, etc.)
            file_params: Business parameters (kb_id, file_id, biz_metadata, security_level, etc.)

        Returns:
            list[TxtChunkEntity]: Chunk list with embeddings and complete path hierarchy
        """
        # Validate embedding model configuration
        model = file_params.txt_embed_model
        if not model:
            logger.error(f"Knowledge base {file_params.kb_id} has no configured text embedding model")
            return []

        if not parser_results:
            logger.warning("Empty parser results, skipping embedding generation")
            return []
        
        # 1. Extract all text content for embedding and filter empty strings
        all_texts = []
        for i, item in enumerate(parser_results):
            content = item.content
            if not content:
                continue
                
            all_texts.append(content)

        if not all_texts:
            logger.warning("All text chunks are empty after filtering, skipping embedding generation")
            return []

        # Keep track of valid indices to match embeddings with parser results
        valid_indices = [i for i, item in enumerate(parser_results) if item.content]

        # 2. Configure micro-batch size (32-64 is optimal balance of concurrency and stability)
        batch_size = await self.model_service.get_embedding_batch_size(embedding_model_name=model)
        micro_batch_size = batch_size or 10
        all_embeddings = []

        try:
            # 3. Process in micro-batches to avoid API limits/timeouts
            for i in range(0, len(all_texts), micro_batch_size):
                batch_texts = all_texts[i : i + micro_batch_size]

                logger.info(
                    f"Processing embedding batch {i//micro_batch_size + 1}, "
                    f"progress: {i}/{len(all_texts)}"
                )

                # Call embedding service (add retry logic if needed for production)
                response = await self.model_client.call_embedding_model(
                    model_name=model,
                    texts=batch_texts,
                    batch_size=micro_batch_size
                )

                if response:
                    all_embeddings.extend([res.embedding for res in response])
                else:
                    raise Exception(f"Empty response for batch {i}, embedding service may have internal error")

            # 4. Validate embedding-text count match
            if len(all_embeddings) != len(all_texts):
                raise Exception(
                    f"Embedding-text count mismatch: "
                    f"texts ({len(all_texts)}) vs embeddings ({len(all_embeddings)})"
                )

            # 5. Create TxtChunkEntity objects with complete metadata
            chunks = []
            for i, (text, emb) in enumerate(zip(all_texts, all_embeddings)):
                # Use valid_indices to map back to original parser results
                original_idx = valid_indices[i]
                item = parser_results[original_idx]
                unique_id = str(uuid.uuid4())
                # Sanitize JSON metadata fields to prevent Oracle JSON syntax errors
                biz_metadata = file_params.biz_metadata or {}

                chunk = TxtChunkEntity(
                    chunk_id=unique_id,
                    chunk_num=item.chunk_num,
                    chunk_type=item.chunk_type,
                    kb_id=file_params.kb_id,
                    file_id=file_params.file_id,
                    content=text,
                    header=item.header,
                    doc_summary=item.doc_summary,
                    search_helper=item.search_helper,
                    embedding=emb,
                    chunk_metadata=item.metadata.model_dump(),
                    biz_metadata=biz_metadata,
                    security_level=file_params.security_level,
                )
                chunks.append(chunk)

            logger.info(f"Successfully generated {len(chunks)} embeddings")
            return chunks

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            return []
    
    async def _save_chunks(self, kb_id: int, file_id: str, chunks: list[TxtChunkEntity]):
        """
        Save parsed chunks with embeddings to database (with error handling)

        Args:
            kb_id: Knowledge base ID
            file_id: File ID
            chunks: List of text chunk entities with embeddings
        """
        async with self.oracle_session as session:
            chunk_repo = TxtChunkRepository(session)
            try:
                # Save chunks to database
                await chunk_repo.create(chunks=chunks)
                
                # Update file status to PARSED
                await self._update_file_status(
                    file_id, 
                    FileStatus.PARSED, 
                    f"Successfully saved {len(chunks)} chunks with embeddings"
                )
                
                logger.info(f"Successfully saved {len(chunks)} chunks for file {file_id}")
                
            except Exception as e:
                msg = f"Error saving chunks to database: {str(e)}"
                logger.error(msg, exc_info=True)
                await self._update_file_status(file_id, FileStatus.PARSE_FAILED, msg)

    async def _check_file(self, file_params: FileParams) -> bool:
        """
        Validate file existence and embedding model configuration
        
        Args:
            file_params: File parameters object
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check embedding model configuration
            if file_params.txt_embed_model is None:
                msg = f"Knowledge base {file_params.kb_id} has no text embedding model configured"
                logger.error(msg)
                await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return False

            # Check file existence
            if not os.path.exists(file_params.file_path):
                msg = f"File path does not exist: {file_params.file_path}"
                logger.error(msg)
                await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return False
            
            return True
                
        except Exception as e:
            msg = f"Error validating file {file_params.file_id}: {str(e)}"
            logger.error(msg, exc_info=True)
            await self._update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False