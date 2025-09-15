import uuid
from loguru import logger
from .file_params import FileParams
from core.dictionary import FileStatus
from configuration import ConfigManager
from .common import update_file_status, save_embeddings
from utils.call_models import CallModel
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import ChunkType

async def process_summary(chunks: list[str], file_params: FileParams) -> bool:
    """处理摘要总结"""
    # 1. 检查摘要总结模型是否存在
    if file_params.summary_model is None:
        msg = f"摘要总结模型不存在，无法进行摘要总结，file_id: {file_params.file_id}"
        logger.error(msg)
        await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False
    
    # 2. 调用模型进行摘要总结
    model_config = ConfigManager.get_model_config() 
    prompt_name = model_config.prompt.summary
    summary_prompt = await KbotMdPromptRepository().get_prompt_by_unique_name(prompt_name)
    if not summary_prompt:
        msg = f"摘要总结提示词不存在，使用默认提示词"
        logger.warning(msg)
        summary_prompt = "请对以下文本进行总结，生成一段简洁的摘要，要求涵盖核心内容与关键信息，保持客观准确，避免冗余细节。\n文本内容如下：\n{chunk}\n摘要："
    else:
        summary_prompt = str(summary_prompt)

    summary_results = []

    # 2.1 将文本块替换到摘要模板中
    for chunk in chunks:
        summary_prompt = summary_prompt.replace("{chunk}", chunk)
        
    # 2.2 调用模型进行摘要总结
        async for summary in CallModel().call_llm_model(
            file_params.summary_model,
            summary_prompt,
            stream=False
            ):
            summary_results.append(summary)

            if not summary:
                msg = f"摘要总结模型调用失败，文本块: {chunk}"
                logger.warning(msg)
                
        logger.debug(f"摘要总结结果: {summary}")

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
    embed_entities = []
    chunk_num = 1
    for chunk, embedding in zip(chunks, embeddings):
        # 保存 embedding 向量到向量数据库
        embed_entity = KbotBizTxtEmbedding(
            embed_id=str(uuid.uuid4()),
            chunk_doc=chunk,
            chunk_metadata={"chunk_type": ChunkType.SUMMARY, "chunk_num": chunk_num},
            file_id=file_params.file_id,
            kb_id=file_params.kb_id,
            embedding=embedding,
            security_level=file_params.security_level
        )
        embed_entities.append(embed_entity)
        chunk_num += 1
    return await save_embeddings(file_params, embed_entities)
