import uuid
import json
import pdfplumber
import pandas as pd
from pathlib import Path
from PIL import Image
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTImage, LTFigure
from loguru import logger
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.data_dict import FileStatus, ChunkType, SplitStrategy
from core.config import settings
from utils.call_models import call_embedding_model, call_vlm_model_for_parsing_picture
from utils.common_methods import check_text_file
import traceback


class PDFPlumberParser:
    """PDF file parser class with optimized processing"""

    def __init__(self, file_params: FileParams):
        self.file_params = file_params
        self.pdf_path = Path(file_params.file_path)
        self.output_dir = self.pdf_path.parent / "output"
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"

        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.text_content: list[dict] = []
        self.text_chunks: list[dict] = []
        self.images_info: list[dict] = []
        self.tables_info: list[dict] = []
        self.page_content: list[dict] = []  # Stores complete page content with placeholders

        self.chunk_size = 0
        self.chunk_overlap = 0

    async def parse(self) -> bool:
        """Main parsing method with optimized flow"""
        split_strategy = int(self.file_params.parser.get("split_strategy", SplitStrategy.FIXED_SIZE.value))
        file_repo = KbotMdKbFilesRepository()

        if split_strategy == SplitStrategy.PAGE.value:
            try:
                # Extract all content by page
                _, images_info, _  = self.extract_all_per_page()

                r = await self._process_images_embeddings(images_info)
                logger.debug(f"================={r}")
                if not r:
                    return False
                # Process text and table embeddings
                if not await self._process_embeddings_per_page():
                    return False

                # Save parsed metadata
                parsed_metadata = self.make_parsed_metadata()
                await file_repo.update_file_parsed_metadata(self.file_params.file_id, parsed_metadata)

                self.print_summary()
                return True

            except Exception as e:
                logger.error(f"Error processing PDF file: {str(e)}")
                logger.exception('asdasd',e)
                await file_repo.update_file_status(
                    self.file_params.file_id,
                    FileStatus.PARSE_FAILED,
                    str(e)
                )
                return False
        elif split_strategy == SplitStrategy.FIXED_SIZE.value:
            try:
                self.chunk_size = int(self.file_params.parser.get("chunk_size", 500))
                self.chunk_overlap = int(self.file_params.parser.get("chunk_overlap", 50))

                _,images_info,_= self.extract_all_by_fixed_size()
                if not await self._process_images_embeddings(images_info):
                    return False

                # Process text and table embeddings
                if not await self._process_embeddings_by_fixed_size():
                    return False

                # Save parsed metadata
                parsed_metadata = self.make_parsed_metadata()
                await file_repo.update_file_parsed_metadata(self.file_params.file_id, parsed_metadata)

                self.print_summary()
                return True

            except Exception as e:
                logger.exception("Error processing PDF file:  {}", e)
                tb = traceback.TracebackException.from_exception(e)
                errMsg= ''.join(tb.format())
                await file_repo.update_file_status(
                    self.file_params.file_id,
                    FileStatus.PARSE_FAILED,
                    errMsg
                )
                return False

        else:
            logger.warning(f"Unrecognized split strategy: {split_strategy}")
            return False
    async def _process_images_embeddings(self,images_info) -> bool:
        if self.file_params.img2txt == 1:
        # if self.file_params.parser.get("extract_images", False):
            model_unique_name = "KBOT112/QwenVL"
            prompt_unique_name = "DEFAULT/image2text"
            chunks = []
            chunk_metas = []
            for eachImage in images_info:
                description_file = Path(eachImage['file_path'] + ".description")
                if not description_file.exists():
                    model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
                        self.file_params.img2txt_model)  # type: ignore

                    image_description = await call_vlm_model_for_parsing_picture(model_unique_name, prompt_unique_name, # type: ignore
                                                                           eachImage['file_path']) 
                    if image_description:
                        description_file.write_text(
                            image_description,
                            encoding='utf-8'
                        )
                        chunk_metas.append({
                            "chunk_type": ChunkType.IMAGE,
                            "page_num": eachImage['page_num'],
                            "image_id": eachImage['uuid'],
                        })
                        chunks.append(image_description)
            text_embedding_model = await KbotMdModelsRepository().get_unique_name_by_id(
                self.file_params.txt_embed_model # type: ignore
            )
            if not text_embedding_model:
                msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
                logger.error(msg)
                await self._update_file_status(FileStatus.PARSE_FAILED, msg)
                return False
            embeddings_list = await call_embedding_model(text_embedding_model, chunks)
            if not embeddings_list or len(embeddings_list) != len(chunks):
                msg = f"Embedding model {model_unique_name} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
                logger.error(msg)
                await self._update_file_status(FileStatus.PARSE_FAILED, msg)
                return False

                # Create embedding entities
            embed_entities = []
            for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
                embed_entity = KbotBizTxtEmbedding(
                    kb_id=self.file_params.kb_id,
                    embed_id=meta['image_id'],
                    chunk_doc=chunk,
                    chunk_metadata=json.dumps(meta),
                    file_id=self.file_params.file_id,
                    embedding=embeddings_list[idx].embedding,
                    security_level=self.file_params.security_level
                )
                embed_entities.append(embed_entity)

            # Save all embeddings in one batch
            return await self._save_embeddings(embed_entities)
        else:
            return True
        

    async def _process_embeddings_per_page(self) -> bool:
        """Process all content embeddings in a unified way"""
        model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
            self.file_params.txt_embed_model  # type: ignore
        )
        if not model_unique_name:
            msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Prepare all content chunks for embedding
        chunks = []
        chunk_metas = []

        # Add text content
        for text_item in self.text_content:
            if not text_item['text'].strip():
                continue

            chunks.append(text_item['text'])
            chunk_metas.append({
                "chunk_type": ChunkType.TEXT,
                "page_num": text_item['page_num']
            })

        # Add table content
        for table in self.tables_info:
            if not self._is_table_valid(table['file_path']):
                continue

            with open(table['file_path'], 'r', encoding='utf-8') as f:
                table_text = f.read()
                if table_text.strip():
                    chunks.append(table_text)
                    chunk_metas.append({
                        "chunk_type": ChunkType.TABLE,
                        "page_num": table['page_num']
                    })

        if not chunks:
            logger.warning("No valid content chunks found for embedding")
            return True  # Consider empty content as success

        # Get all embeddings in one call
        embeddings_list = await call_embedding_model(model_unique_name, chunks)
        if not embeddings_list or len(embeddings_list) != len(chunks):
            msg = f"Embedding model {model_unique_name} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Create embedding entities
        embed_entities = []
        for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
            embed_entity = KbotBizTxtEmbedding(
                kb_id=self.file_params.kb_id,
                embed_id=str(uuid.uuid4()),
                chunk_doc=chunk,
                chunk_metadata=json.dumps(meta),
                file_id=self.file_params.file_id,
                embedding=embeddings_list[idx].embedding,
                security_level=self.file_params.security_level
            )
            embed_entities.append(embed_entity)

        # Save all embeddings in one batch
        return await self._save_embeddings(embed_entities)


    async def _process_embeddings_by_fixed_size(self) -> bool:
        """Process text and table embeddings for by fixed size"""
        model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
            self.file_params.txt_embed_model  # type: ignore
        )
        if not model_unique_name:
            msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Prepare all content chunks for embedding
        chunks = []
        chunk_metas = []

        # Add text content
        for text_item in self.text_chunks:
            if not text_item['text'].strip():
                continue

            chunks.append(text_item['text'])
            chunk_metas.append({
                "chunk_type": ChunkType.TEXT,
                "page_num": text_item['page_num']
            })

        # Add table content
        for table in self.tables_info:
            if not self._is_table_valid(table['file_path']):
                continue

            with open(table['file_path'], 'r', encoding='utf-8') as f:
                table_text = f.read()
                if table_text.strip():
                    chunks.append(table_text)
                    chunk_metas.append({
                        "chunk_type": ChunkType.TABLE,
                        "page_num": table['page_num']
                    })

        if not chunks:
            logger.warning("No valid content chunks found for embedding")
            return True  # Consider empty content as success

        # Get all embeddings in one call
        embeddings_list = await call_embedding_model(model_unique_name, chunks)
        if not embeddings_list or len(embeddings_list) != len(chunks):
            msg = f"Embedding model {model_unique_name} returned invalid results (expected {len(chunks)}, got {len(embeddings_list) if embeddings_list else 0})"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

        # Create embedding entities
        embed_entities = []
        for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
            embed_entity = KbotBizTxtEmbedding(
                kb_id=self.file_params.kb_id,
                embed_id=str(uuid.uuid4()),
                chunk_doc=chunk,
                chunk_metadata=json.dumps(meta),
                file_id=self.file_params.file_id,
                embedding=embeddings_list[idx].embedding,
                security_level=self.file_params.security_level
            )
            embed_entities.append(embed_entity)

        # Save all embeddings in one batch
        return await self._save_embeddings(embed_entities)


    async def _save_embeddings(self, embeddings: list[KbotBizTxtEmbedding]) -> bool:
        """Save embeddings to database with error handling"""
        if not embeddings:
            return False

        try:
            repo = KbotBizTxtEmbeddingRepository()
            result = await repo.create(kb_id=self.file_params.kb_id, embeddings=embeddings)
            if not result:
                msg = "Failed to save embeddings (repository returned False)"
                logger.error(msg)
                await self._update_file_status(FileStatus.PARSE_FAILED, msg)
                return False

            logger.info(f"Successfully saved {len(embeddings)} embeddings")
            return True

        except Exception as e:
            msg = f"Exception while saving embeddings: {str(e)}"
            logger.error(msg)
            await self._update_file_status(FileStatus.PARSE_FAILED, msg)
            return False

    async def _update_file_status(self, status: FileStatus, message: str) -> None:
        """Helper method to update file status"""
        await KbotMdKbFilesRepository().update_file_status(
            self.file_params.file_id,
            status,
            message
        )

    def extract_all_per_page(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Extract all content from PDF by page"""
        logger.info(f"Parsing file: {self.pdf_path}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")

                    # Extract text and tables
                    page_text, page_tables = self._extract_text_and_tables(page, page_num)

                    # Extract images
                    page_images = self._extract_images_from_page(page_num)

                    # Combine content
                    combined = self._combine_page_content(page_text, page_images, page_tables, page_num)
                    self.page_content.append({'page_num': page_num, 'content': combined})
                    print(445, combined)

        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise

        return self.text_content, self.images_info, self.tables_info

    def extract_all_by_fixed_size(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Extract all content from PDF by page"""
        logger.info(f"Parsing file: {self.pdf_path}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")

                    # Extract text and tables
                    page_text, page_tables = self._extract_text_and_tables(page, page_num)

                    # Extract images
                    page_images = self._extract_images_from_page(page_num)

                    # Combine content
                    # combined = self._combine_page_content(page_text, page_images, page_tables, page_num)
                    # self.page_content.append({'page_num': page_num, 'content': page_text})

        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
        for page in self.text_content:
            page_num = page['page_num']
            content = page['text']
            for chunk_start in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_text = content[chunk_start:chunk_start + self.chunk_size]
                self.text_chunks.append({'page_num': page_num, 'text': chunk_text})
        return self.text_chunks, self.images_info, self.tables_info

    def _extract_text_and_tables(self, page, page_num: int) -> tuple[str, list[dict]]:
        """Extract text and tables from a page"""
        page_text = ""
        page_tables = []

        try:
            # Extract tables
            tables = page.find_tables()
            for table in tables:
                table_data = table.extract()
                if not table_data:
                    continue

                table_uuid = str(uuid.uuid4())
                csv_path = self.tables_dir / f"table_{table_uuid}.csv"

                # Save as CSV
                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                table_info = {
                    'uuid': table_uuid,
                    'filename': csv_path.name,
                    'page_num': page_num,
                    'file_path': str(csv_path.absolute()),
                    'rows': len(table_data),
                    'columns': len(table_data[0]) if table_data and table_data[0] else 0,
                    'bbox': table.bbox
                }

                self.tables_info.append(table_info)
                page_tables.append(table_info)
                logger.debug(f"Saved table: {csv_path.absolute()} (page {page_num})")

            # Extract text (excluding table areas)
            page_text = page.extract_text() or ""
            if page_text.strip():
                self.text_content.append({
                    'page_num': page_num,
                    'text': page_text.strip()
                })

        except Exception as e:
            logger.error(f"Error extracting text/tables from page {page_num}: {e}")
            page_text = page.extract_text() or ""

        return page_text, page_tables

    def _extract_images_from_page(self, page_num: int) -> list[dict]:
        """Extract images from a page using pdfminer"""
        page_images = []

        try:
            with open(self.pdf_path, 'rb') as file:
                rsrcmgr = PDFResourceManager()
                device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)

                for current_page_num, page in enumerate(PDFPage.get_pages(file), 1):
                    if current_page_num == page_num:
                        interpreter.process_page(page)
                        layout = device.get_result()
                        page_images = self._process_layout_images(layout, page_num)
                        break

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")

        return page_images

    def _process_layout_images(self, layout, page_num: int) -> list[dict]:
        """Process layout to extract images"""
        images = []

        for obj in layout:
            if isinstance(obj, LTImage):
                image_info = self._save_image(obj, page_num)
                if image_info:
                    images.append(image_info)
            elif isinstance(obj, LTFigure):
                images.extend(self._process_layout_images(obj, page_num))

        return images

    def _save_image(self, lt_image, page_num: int) -> dict | None:
        """Save an extracted image"""
        try:
            # Skip background images (starting at 0,0)
            if hasattr(lt_image, 'bbox'):
                x0, y0, _, _ = lt_image.bbox
                if x0 == 0 or y0 == 0:
                    logger.debug(f"Skipping background image at page {page_num}")
                    return None

            # Get image data
            result = self._get_image_data(lt_image)
            if not result:
                return None

            image_data, ext, pil_image = result
            image_uuid = str(uuid.uuid4())

            # Save image
            if pil_image:
                image_path = self.images_dir / f"{image_uuid}.png"
                pil_image.save(image_path, format="PNG")
                width, height = pil_image.size
            else:
                image_path = self.images_dir / f"{image_uuid}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                width = getattr(lt_image, 'width', 0)
                height = getattr(lt_image, 'height', 0)

            # Skip small images
            if width < 200 or height < 200:
                logger.debug(f"Skipping small image ({width}x{height}) at page {page_num}")
                return None

            # Create image info
            image_info = {
                'uuid': image_uuid,
                'filename': image_path.name,
                'page_num': page_num,
                'file_path': str(image_path.absolute()),
                'width': width,
                'height': height,
                'bbox': getattr(lt_image, 'bbox', None),
                'format': ext
            }

            self.images_info.append(image_info)
            logger.debug(f"Saved image: {image_path.absolute()} (page {page_num})")
            return image_info

        except Exception as e:
            logger.error(f"Error saving image from page {page_num}: {e}")
            return None

    def _get_image_data(self, lt_image) -> tuple[bytes, str, Image.Image | None] | None:
        """Extract image data from LTImage object"""
        if not hasattr(lt_image, 'stream') or not lt_image.stream:
            return None

        stream = lt_image.stream
        filters = stream.get('Filter', [])
        if isinstance(filters, list):
            filters = filters[0] if filters else None

        if not filters:
            return stream.get_rawdata(), 'bin', None

        filter_name = getattr(filters, 'name', '')
        if filter_name == 'DCTDecode':
            return stream.get_rawdata(), 'jpg', None
        elif filter_name == 'JPXDecode':
            return stream.get_rawdata(), 'jp2', None
        elif filter_name == 'FlateDecode':
            try:
                width = int(stream.get('Width', 0))
                height = int(stream.get('Height', 0))
                color_space = stream.get('ColorSpace')
                data = stream.get_data()

                mode = 'RGB'  # default
                if color_space:
                    if isinstance(color_space, list):
                        color_space = color_space[0]
                    if hasattr(color_space, 'name'):
                        if color_space.name == 'DeviceGray':
                            mode = 'L'
                        elif color_space.name == 'DeviceCMYK':
                            mode = 'CMYK'

                pil_image = Image.frombytes(mode, (width, height), data)
                return data, 'png', pil_image
            except Exception as e:
                logger.error(f"Error decoding FlateDecode image: {e}")
                return None
        else:
            return stream.get_rawdata(), 'bin', None

    def _combine_page_content(self, text: str, images: list[dict], tables: list[dict], page_num: int) -> str:
        """Combine page content with placeholders
        especially for split by page strategy
        """
        content = [f"\n{'=' * 20} Page {page_num} {'=' * 20}\n"]

        if images:
            content.append("\n=== Images ===\n")
            content.extend(f"[image:{img['uuid']}]\n" for img in images)

        if tables:
            content.append("\n=== Tables ===\n")
            content.extend(f"[table:{table['uuid']}]\n" for table in tables)

        if text.strip():
            content.append("\n=== Text ===\n")
            content.append(text)

        return ''.join(content)

    def make_parsed_metadata(self) -> str:
        """Generate metadata JSON with placeholders"""
        valid_tables = [t for t in self.tables_info if self._is_table_valid(t['file_path'])]

        metadata = {
            'images': [
                {
                    'uuid': img['uuid'],
                    'placeholder': f"[image:{img['uuid']}]",
                    'filename': img['filename'],
                    'page_num': img['page_num'],
                    'file_path': img['file_path']
                } for img in self.images_info
            ],
            'tables': [
                {
                    'uuid': table['uuid'],
                    'placeholder': f"[table:{table['uuid']}]",
                    'filename': table['filename'],
                    'page_num': table['page_num'],
                    'file_path': table['file_path']
                } for table in valid_tables
            ]
        }

        return json.dumps(metadata, ensure_ascii=False, indent=2)

    def _is_table_valid(self, csv_path: str) -> bool:
        """Check if a table CSV file contains valid content"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in content)
        except Exception:
            return False

    def save_results(self):
        """Save all extracted results to files"""
        try:
            # Save text with placeholders
            # full_text = "\n".join(page['content'] for page in self.page_content)
            # (self.output_dir / "extracted_text_with_placeholders.txt").write_text(full_text, encoding='utf-8')

            # Save pure text
            # pure_text = "\n".join(item['text'] for item in self.text_content)
            # (self.output_dir / "extracted_text_only.txt").write_text(pure_text, encoding='utf-8')

            # Save metadata files
            # (self.output_dir / "images_info.json").write_text(
            #     json.dumps(self.images_info, ensure_ascii=False, indent=2),
            #     encoding='utf-8'
            # )

            # (self.output_dir / "tables_info.json").write_text(
            #     json.dumps(self.tables_info, ensure_ascii=False, indent=2),
            #     encoding='utf-8'
            # )

            # (self.output_dir / "placeholders_mapping.json").write_text(
            #     self.make_parsed_metadata(),
            #     encoding='utf-8'
            # )

            logger.info(f"All results saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def print_summary(self):
        """Print parsing summary"""
        logger.info("\n" + "=" * 50)
        logger.info("PDF Parsing Complete!")
        logger.info("=" * 50)
        logger.info(f"Text paragraphs extracted: {len(self.text_content)}")
        logger.info(f"Images extracted: {len(self.images_info)}")
        logger.info(f"Tables extracted: {len(self.tables_info)}")
        logger.info(f"Output directory: {self.output_dir}")

        pages_with_text = {item['page_num'] for item in self.text_content}
        pages_with_images = {item['page_num'] for item in self.images_info}
        pages_with_tables = {item['page_num'] for item in self.tables_info}

        logger.info(f"Pages with text: {len(pages_with_text)}")
        logger.info(f"Pages with images: {len(pages_with_images)}")
        logger.info(f"Pages with tables: {len(pages_with_tables)}")
        logger.info("=" * 50)

    # async def _process_embeddings(self) -> bool:
    #         """Process text and table embeddings with optimized database operations"""
    #         model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(
    #             self.file_params.txt_embed_model # type: ignore
    #         )
    #         if not model_unique_name:
    #             msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
    #             logger.error(msg)
    #             await self._update_file_status(FileStatus.PARSE_FAILED, msg)
    #             return False

    #         # Process text embeddings
    #         text_embeddings = await self._create_text_embeddings(model_unique_name)
    #         if text_embeddings is None:
    #             return False

    #         # Process table embeddings
    #         table_embeddings = await self._create_table_embeddings(model_unique_name)
    #         if table_embeddings is None:
    #             return False

    #         return True

    # async def _create_text_embeddings(self, model_unique_name: str) -> list[KbotBizTxtEmbedding] | None:
    #     """Create embeddings for text content"""
    #     texts = [item['text'] for item in self.text_content if item['text'].strip()]
    #     if not texts:
    #         return []

    #     embeddings_list = await call_embedding_model(model_unique_name, texts)
    #     if not embeddings_list:
    #         msg = f"Embedding model {model_unique_name} returned None."
    #         logger.error(msg)
    #         await self._update_file_status(FileStatus.PARSE_FAILED, msg)
    #         return None

    #     embed_entities = []
    #     for idx, text_item in enumerate(self.text_content):
    #         if not text_item['text'].strip():
    #             continue

    #         embed_entity = KbotBizTxtEmbedding(
    #             embed_id=str(uuid.uuid4()),
    #             chunk_doc=text_item['text'],
    #             chunk_metadata=json.dumps({
    #                 "chunk_type": ChunkType.TEXT,
    #                 "split_strategy": SplitStrategy.BY_PAGE.value,
    #                 "file_path": str(self.pdf_path),
    #                 "page_num": text_item['page_num']
    #             }),
    #             file_id=self.file_params.file_id,
    #             embedding=embeddings_list[idx].embedding
    #         )
    #         embed_entities.append(embed_entity)

    #     return await self._save_embeddings(embed_entities)

    # async def _create_table_embeddings(self, model_unique_name: str) -> list[KbotBizTxtEmbedding] | None:
    #     """Create embeddings for table content"""
    #     texts = []
    #     valid_tables = []

    #     for table in self.tables_info:
    #         if not self._is_table_valid(table['file_path']):
    #             continue

    #         with open(table['file_path'], 'r', encoding='utf-8') as f:
    #             table_text = f.read()
    #             if table_text.strip():
    #                 texts.append(table_text)
    #                 valid_tables.append(table)

    #     if not texts:
    #         return []

    #     embeddings_list = await call_embedding_model(model_unique_name, texts)
    #     if not embeddings_list:
    #         msg = f"Embedding model {model_unique_name} returned None."
    #         logger.error(msg)
    #         await self._update_file_status(FileStatus.PARSE_FAILED, msg)
    #         return None

    #     embed_entities = []
    #     for idx, table in enumerate(valid_tables):
    #         with open(table['file_path'], 'r', encoding='utf-8') as f:
    #             table_text = f.read()
    #             if not table_text.strip():
    #                 continue

    #             embed_entity = KbotBizTxtEmbedding(
    #                 embed_id=str(uuid.uuid4()),
    #                 chunk_doc=table_text,
    #                 chunk_metadata=json.dumps({
    #                     "chunk_type": ChunkType.TEXT,
    #                     "split_strategy": SplitStrategy.BY_PAGE.value,
    #                     "file_path": str(self.pdf_path),
    #                     "page_num": table['page_num']
    #                 }),
    #                 file_id=self.file_params.file_id,
    #                 embedding=embeddings_list[idx].embedding
    #             )
    #             embed_entities.append(embed_entity)

    #     return await self._save_embeddings(embed_entities)


async def process_pdf(file_params: FileParams) -> bool:
    """
    Process PDF file by extracting content and generating embeddings

    Args:
        file_params: File parameters including path and processing options

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    if not check_text_file(file_params):
        return False

    try:
        logger.info(f"Processing PDF file: {file_params.file_path}")
        parser = PDFPlumberParser(file_params)
        r = await parser.parse()
        if r:
            msg = f"Successfully parsed {file_params.file_path} (file id: {file_params.file_id})"
            await KbotMdKbFilesRepository().update_file_status(
                file_params.file_id,
                FileStatus.PARSED,
                str(msg)
            )
            return True
        else:
            msg = f"Failed to parse {file_params.file_path} (file id: {file_params.file_id})"
            await KbotMdKbFilesRepository().update_file_status(
                file_params.file_id,
                FileStatus.PARSE_FAILED,
                str(msg)
            )
            return False
    except Exception as e:
        msg = f"Error processing PDF file: {file_params.file_path}, error: {str(e)}"
        logger.error(msg)
        await KbotMdKbFilesRepository().update_file_status(
            file_params.file_id,
            FileStatus.PARSE_FAILED,
            msg
        )
        return False