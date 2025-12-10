import pandas as pd
import json
import uuid

from loguru import logger
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus, ChunkType, SplitStrategy
from utils.call_models import CallModel
from .common import check_text_file, update_file_status, save_embeddings

from .summary_parser import SummaryParser

class CSVParser:
    def __init__(self, file_params: FileParams):
        self.csv_file = file_params.file_path
        self.file_params = file_params

    def parse_csv_to_json(self):
        """Parse CSV file using pandas DataFrame and convert to JSON format.
        Returns a list where each item is a JSON object with headers as keys."""
        try:
            # Read CSV using pandas
            df = pd.read_csv(self.csv_file)
            # Convert DataFrame to list of dicts (JSON objects)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            return None

    async def save_to_database(self, json_data):
        """Save JSON data to database"""
        if not json_data:
            return False

        if not self.file_params.txt_embed_model:
            msg = f"Embedding model not found for id: {self.file_params.txt_embed_model}"
            logger.error(msg)
            await update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        chunks = [json.dumps(row) for row in json_data]
        logger.debug(chunks)
        chunk_metas = [{"chunk_type": ChunkType.TABLE} for _ in json_data]

        embeddings_list = await CallModel().call_embedding_model(self.file_params.txt_embed_model, chunks)
        if not embeddings_list or len(embeddings_list) != len(chunks):
            msg = f"Embedding model returned invalid results"
            logger.error(msg)
            await update_file_status(self.file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        embed_entities = [
            KbotBizTxtEmbedding(
                kb_id=self.file_params.kb_id,
                embed_id=str(uuid.uuid4()),
                chunk_doc=chunk,
                chunk_metadata=json.dumps(meta), # type: ignore
                biz_metadata=self.file_params.biz_metadata,
                file_id=self.file_params.file_id,
                embedding=embeddings_list[idx].embedding,
                security_level=self.file_params.security_level,
                status=1
            )
            for idx, (chunk, meta) in enumerate(zip(chunks, chunk_metas))
        ]
        if self.file_params.enable_summary:
            summary_result = await SummaryParser.process_summary(file_params=self.file_params, embed_entities=embed_entities)
            return summary_result
        return await save_embeddings(self.file_params, embed_entities)

    async def parse(self):
        split_strategy = int(self.file_params.parser.get("split_strategy", SplitStrategy.ROW.value))
        if split_strategy == SplitStrategy.ROW.value:

            """Parse CSV file and save to database"""
            json_data = self.parse_csv_to_json()
            if json_data:
                return await self.save_to_database(json_data)
            return True
        else:
            logger.warning(f"Unrecognized split strategy: {split_strategy}")
            return False


async def process_csv(file_params: FileParams) -> bool:
    """
    Process csv file by extracting content and generating embeddings

    Args:
        file_params: File parameters including path and processing options

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    is_success=False
    file_status=FileStatus.PARSE_FAILED
    if not await check_text_file(file_params):
        return False

    try:
        logger.info(f"Processing CSV file: {file_params.file_path}")
        parser = CSVParser(file_params)
        r = await parser.parse()
        if r:
            msg = f"Successfully parsed {file_params.file_path} (file id: {file_params.file_id})"
            file_status=FileStatus.PARSED
            is_success=True
        else:
            msg = f"Failed to parse {file_params.file_path} (file id: {file_params.file_id})"
            file_status=FileStatus.PARSE_FAILED
            is_success=False

        await KbotMdKbFilesRepository().update_file_status(
            file_params.file_id,
            file_status,
            str(msg)
        )
        return is_success
    except Exception as e:
        msg = f"Error processing csv file: {file_params.file_path}, error: {str(e)}"
        logger.exception(e)
        logger.error(msg)
        await KbotMdKbFilesRepository().update_file_status(
            file_params.file_id,
            FileStatus.PARSE_FAILED,
            msg
        )
        return False
