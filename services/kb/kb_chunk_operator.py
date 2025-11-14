from loguru import logger
from core.config.settings import get_prompt_config
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from utils.call_models import CallModel
from services.dataparse.summary_parser import SummaryParser



class KBChunkOperator:
    """
    知识库分片操作类
    """

    async def edit_file_chunk(self, embed_id: str, file_id: str, kb_id: int, new_chunk: str) -> bool:
        """
        编辑文件分片
        
        参数:
            embed_id: 分片ID
            file_id: 文件ID
            kb_id: 知识库ID
            new_chunk: 新分片内容
        
        返回:
            bool: 编辑是否成功
        """
        # 获取知识库的向量模型
        kb = await KbotMdKbRepository().get_by_id(kb_id)
        if kb is None:
            logger.error(f"知识库 {kb_id} 不存在，无法更新分片")
            return False
        embed_model = kb.txt_embed_model_id
        if embed_model is None:
            logger.error(f"知识库 {kb_id} 没有向量模型，无法更新分片")
            return False
        
        # 获取新分片的向量
        response_data = await CallModel().call_embedding_model(embed_model, [new_chunk])
        if response_data is None:
            logger.error(f"获取分片 chunk: {embed_id} 的 embedding 向量失败")
            return False
        else:
            logger.info(f"成功获取分片 chunk: {embed_id} 的 embedding 向量")
            embeddings = [item.embedding for item in response_data]

        # 更新向量库中的分片信息
        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在，无法更新分片")
            return False

        try:
            r = await embed_repo.update_chunk(embed_id=embed_id, new_chunk=new_chunk, new_embedding=embeddings[0])
            if r:
                logger.info(f"成功更新文件 {file_id} 的分片 {embed_id}")
            else:
                logger.warning(f"未找到文件 {file_id} 的分片 {embed_id}，未进行更新")
        except Exception as e:
            logger.error(f"更新文件 {file_id} 的分片 {embed_id} 失败: {str(e)}")
            return False
        
        # 如果知识库启用摘要，则重新生成该分片的摘要并更新摘要的向量
        if kb.enable_summary:
            logger.debug(f"知识库 {kb_id} 启用摘要，更新文件 {file_id}, 分片 {embed_id} 的摘要")
            # 获取摘要的 embed id
            summary_chunk_id = await embed_repo.get_summary_id_by_chunk_id(file_id=file_id, chunk_id=embed_id)
            if summary_chunk_id is None:
                logger.error(f"未找到文件 {file_id} 的分片 {embed_id} 的摘要记录，无法更新摘要")
                return False

            # 获取摘要模型
            summary_model = kb.summary_model_id
            if summary_model is None:
                logger.error(f"知识库 {kb_id} 没有摘要模型，无法更新分片摘要")
                return False
            
            # 获取摘要提示词
            model_config = get_prompt_config() 
            prompt_name = model_config.summary
            summary_prompt = await KbotMdPromptRepository().get_prompt_by_unique_name(prompt_name)
            if not summary_prompt:
                msg = f"摘要总结提示词不存在，使用默认提示词"
                logger.warning(msg)
                summary_prompt = "请对以下文本进行总结，提炼出核心内容和关键信息。要求摘要简洁、准确、连贯。待总结文本：\n{chunk}\n"
            else:
                summary_prompt = str(summary_prompt)
            
            prompt = summary_prompt.replace("{chunk}", new_chunk)

            # 获取新分片的摘要
            summary = await SummaryParser.generate_summary(chunk=new_chunk, summary_model_id=summary_model, prompt=prompt)

            if summary is None:
                logger.error(f"获取分片 chunk: {embed_id} 的摘要失败")
                return False
            else:
                logger.debug(f"成功获取分片 chunk: {embed_id} 的摘要: {summary}")

            # 获取分片摘要的向量
            response_data = await CallModel().call_embedding_model(embed_model, [new_chunk])
            if response_data is None:
                logger.error(f"获取分片 chunk: {embed_id} 摘要的 embedding 向量失败")
                return False
            else:
                logger.info(f"成功获取分片 chunk: {embed_id} 摘要的 embedding 向量")
                embeddings = [item.embedding for item in response_data]
            
            # 更新向量库中的分片摘要信息
            try:
                r = await embed_repo.update_chunk(embed_id=summary_chunk_id, new_chunk=summary, new_embedding=embeddings[0])
                if r:
                    logger.info(f"成功更新文件 {file_id} 的分片 {embed_id}")
                    return True
                else:
                    logger.warning(f"未找到文件 {file_id} 的分片 {embed_id}，未进行更新")
            except Exception as e:
                logger.error(f"更新文件 {file_id} 的分片 {embed_id} 失败: {str(e)}")
                return False
        return True
    
    async def delete_file_chunk(self, embed_id: str, file_id: str, kb_id: int) -> bool:
        """
        删除文件分片
        
        参数:
            embed_id: 分片ID
            file_id: 文件ID
            kb_id: 知识库ID
        
        返回:
            bool: 删除是否成功
        """
        # 获取知识库的参数
        kb = await KbotMdKbRepository().get_by_id(kb_id)
        if kb is None:
            logger.error(f"知识库 {kb_id} 不存在，无法删除分片")
            return False
        
        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在，无法删除分片")
            return False
        
        chunk_ids = [embed_id]
        # 如果知识库启用摘要，则删除该分片的摘要
        if kb.enable_summary:
            logger.debug(f"知识库 {kb_id} 启用摘要，删除文件 {file_id}, 分片 {embed_id} 的摘要")
            # 获取摘要的 embed id
            summary_chunk_id = await embed_repo.get_summary_id_by_chunk_id(file_id=file_id, chunk_id=embed_id)
            if summary_chunk_id is None:
                logger.warning(f"未找到文件 {file_id} 的分片 {embed_id} 的摘要记录，跳过删除摘要")
            else:
                chunk_ids.append(summary_chunk_id)
        try:
            r = await embed_repo.delete_by_embed_ids(chunk_ids)
            if r:
                logger.info(f"成功删除文件 {file_id} 的分片 {embed_id}")
                return True
            else:
                logger.warning(f"删除文件 {file_id} 的分片 {embed_id} 失败，未找到该分片")
                return False
        except Exception as e:
            logger.error(f"删除文件 {file_id} 的分片 {embed_id} 失败: {str(e)}")
            return False
        
    async def toogle_file_chunk_status(self, kb_id: int, chunk_id: str, status: int) -> bool:
        """
        切换文件分片状态
        
        参数:
            kb_id: 知识库ID
            chunk_id: 分片ID
            status: 新状态(0-禁用 1-启用)
        
        返回:
            bool: 切换是否成功
        """
        if status not in [0, 1]:
            logger.error(f"无效的状态值: {status}，必须是0或1")
            return False
        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在，无法切换分片状态")
            return False

        try:
            r = await embed_repo.update_status_by_chunk_id(chunk_id=chunk_id, status=status)
            if r > 0:
                logger.info(f"成功切换文件分片 {chunk_id} 的状态为 {status}")
                return True
            else:
                logger.warning(f"未找到文件分片 {chunk_id} ，未进行状态切换")
                return False
        except Exception as e:
            logger.error(f"切换文件分片 {chunk_id} 状态失败: {str(e)}")
            return False
        

    async def get_chunks_by_file_id(self, kb_id: int, file_id: str) -> list[dict]:
        """
        获取文件分片
        
        参数:
            kb_id: 知识库ID
            file_id: 文件ID
        
        返回:
            list[dict]: 分片列表
        """
        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在，无法获取分片")
            return []
        
        try:
            chunks = await embed_repo.get_chunks_by_file_id(file_id=file_id)
            if chunks:
                logger.info(f"成功获取文件 {file_id} 的分片")
                result = []
                for chunk in chunks:
                    result.append(chunk.to_dict())
                return result
            else:
                logger.warning(f"未找到文件 {file_id} 的分片")
                return []
        except Exception as e:
            logger.error(f"获取文件 {file_id} 的分片失败: {str(e)}")
            return []
        
    async def update_chunk_description(
            self,
            embed_id: str,
            kb_id: int,
            description: str
        ) -> bool:
        """更新知识库文件的分片描述"""
        kb = await KbotMdKbRepository().get_by_id(kb_id)
        if kb is None:
            logger.error(f"知识库 {kb_id} 不存在，无法更新分片 {embed_id} 的描述")
            return False
        embed_model = kb.txt_embed_model_id
        if embed_model is None:
            logger.error(f"知识库 {kb_id} 没有向量模型，无法更新分片")
            return False
        
        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在，无法更新分片 {embed_id} 的描述")
            return False
        
        # 1. 获取分片原文
        chunk_doc = await embed_repo.get_chunk_doc_by_id(embed_id=embed_id)
        if chunk_doc is None:
            logger.error(f"未找到分片 {embed_id} 的原文，无法更新描述")
            return False
        
        # 2. 将描述添加到分片原文中后重新生成向量
        chunk_doc_with_desc = f"文本描述: {description}\n 原文: {chunk_doc}"
        
        
        # 3. 获取新分片的向量
        response_data = await CallModel().call_embedding_model(embed_model, [chunk_doc_with_desc])
        if response_data is None:
            logger.error(f"获取分片 chunk: {embed_id} 描述后的 embedding 向量失败")
            return False
        else:
            logger.info(f"成功获取分片 chunk: {embed_id} 描述后的 embedding 向量")
            embeddings = [item.embedding for item in response_data]

        # 4. 更新分片描述和向量
        try:
            r = await embed_repo.update_chunk_description(embed_id=embed_id, description=description, embeddings=embeddings[0])
            if r:
                logger.info(f"成功更新分片 {embed_id} 的描述")
            else:
                logger.warning(f"未找到分片 {embed_id} ，未更新描述")
            
            return r
        except Exception as e:
            logger.error(f"更新分片 {embed_id} 的描述失败: {str(e)}")
            return False

    async def update_chunk_tags(
            self,
            file_id: str,
            kb_id: int,
            tags: list[str]
        ) -> bool:
        """更新知识库文件的分片标签"""
        kb = await KbotMdKbRepository().get_by_id(kb_id)
        if kb is None:
            logger.error(f"知识库 {kb_id} 不存在，无法更新文件 {file_id} 的标签")
            return False
        
        embed_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)
        if embed_repo is None:
            logger.error(f"知识库 {kb_id} 对应的向量库不存在，无法更新文件 {file_id} 的标签")
            return False
        
        # 更新分片标签
        try:
            r = await embed_repo.update_tags(file_id=file_id, tags=tags)

            if r:
                logger.info(f"成功更新文件 {file_id} 分片的标签")
            else:
                logger.warning(f"更新文件 {file_id} 分片的标签失败")
            
            return r
        
        except Exception as e:
            logger.error(f"更新文件 {file_id} 分片的标签失败: {str(e)}")
            return False
