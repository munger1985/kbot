import uuid
import json
from loguru import logger
from .file_params import FileParams
from core.dictionary import FileStatus
from core.config.settings import get_prompt_config
from .common import update_file_status, save_embeddings
from utils.call_models import CallModel
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import ChunkType

class SummaryParser:
    """摘要总结处理器"""

    @staticmethod
    async def process_summary(file_params: FileParams, embed_entities: list[KbotBizTxtEmbedding]) -> bool:
        """处理摘要总结"""
        # 1. 检查摘要总结模型是否存在
        if file_params.summary_model is None:
            msg = f"摘要总结模型不存在，无法进行摘要总结，file_id: {file_params.file_id}"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        
        # 2. 调用模型进行摘要总结
        prompt_config = get_prompt_config() 
        prompt_name = prompt_config.summary
        summary_prompt = await KbotMdPromptRepository().get_prompt_by_unique_name(prompt_name)
        if not summary_prompt:
            msg = f"摘要总结提示词不存在，使用默认提示词"
            logger.warning(msg)
            summary_prompt = "请对以下文本进行总结，提炼出核心内容和关键信息。要求摘要简洁、准确、连贯。待总结文本：\n{chunk}\n"
        else:
            summary_prompt = str(summary_prompt)

        summary_results = []

        # 2.1 将文本块替换到摘要模板中
        for embed_entity in embed_entities:
            prompt = summary_prompt.replace("{chunk}", embed_entity.chunk_doc)
            summary = await SummaryParser.generate_summary(embed_entity.chunk_doc, file_params.summary_model, prompt)
            if summary:
                summary_results.append(summary)
            else:
                msg = f"摘要总结模型调用失败，文本块: {embed_entity.chunk_doc}"
                logger.warning(msg)

        if len(summary_results) == 0:
            msg = f"摘要总结模型调用失败，文件: {file_params.file_path}"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False

        # 3. 调用 embedding 模型将摘要结果转换为向量
        if file_params.txt_embed_model is None:
            msg = f"文本 embedding 模型不存在，无法将摘要结果转换为向量，文件: {file_params.file_path}"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        
        response_data = await CallModel().call_embedding_model(
            file_params.txt_embed_model,
            summary_results,
            batch_size=2
        )
        if response_data is None:
                msg = f"获取文件 {file_params.file_path} 的 embedding 向量失败"
                logger.error(msg)
                await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return False
        else:
            logger.info(f"成功获取 {len(response_data)} 个 embedding 向量")

        # 4. 将摘要结果写入数据库
        embeddings = [item.embedding for item in response_data]
        summary_entities = []
        chunk_num = 1
        for summary_result, embedding in zip(summary_results, embeddings):
            # 保存 embedding 向量到向量数据库
            summary_entity = KbotBizTxtEmbedding(
                embed_id=str(uuid.uuid4()),
                chunk_doc=summary_result,
                chunk_metadata={"chunk_type": ChunkType.SUMMARY, 
                                "chunk_num": chunk_num, 
                                "source_embed_id": embed_entities[chunk_num - 1].embed_id, 
                                "page_num": embed_entities[chunk_num - 1].chunk_metadata.get("page_num", 1)},
                biz_metadata=file_params.biz_metadata,
                file_id=file_params.file_id,
                kb_id=file_params.kb_id,
                embedding=embedding,
                security_level=file_params.security_level,
                status=1
            )
            summary_entities.append(summary_entity)
            chunk_num += 1
        return await save_embeddings(file_params, summary_entities)

    @staticmethod
    async def generate_summary(chunk: str, summary_model_id: int, prompt: str) -> str:
        """生成摘要"""
        if summary_model_id is None:
            logger.error("摘要模型ID未提供，无法生成摘要")
            return ""
        
        summary = ""
        try:
            async for response in CallModel().call_llm_model(
                summary_model_id,
                prompt,
                stream=False
                ):
                try:
                    json_response = json.loads(response)
                    summary = json_response.get("choices")[0].get("message").get("content", "").strip()
                    logger.debug(f"摘要模型返回 JSON 格式提取结果: {summary}")
                except Exception as e:
                    logger.warning(f"摘要模型返回结果非 JSON 格式，直接使用文本结果，错误信息: {e}")
                    summary = response.strip()
                
                if not summary:
                    logger.warning(f"摘要模型调用失败，文本块: {chunk}")
                else:
                    logger.debug(f"摘要结果: {summary}")
        except Exception as e:
            logger.error(f"调用摘要模型失败: {e}")
        
        return summary
