"""image-search-skill 核心实现。

调用 VisualSearchEngine.search() 执行双向互检索，
返回图文配对列表。
"""

import json
from loguru import logger


class ImageSearchSkill:
    """图片搜知识库技能"""

    async def execute(
        self,
        image_base64: str,
        query: str = "",
        kb_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> str:
        """执行视觉搜索

        Args:
            image_base64: base64 编码的图片
            query: 辅助文本
            kb_ids: 知识库 ID 列表
            top_k: 返回数量

        Returns:
            JSON 格式的搜索结果字符串
        """
        from services.visual.search_engine import VisualSearchEngine

        engine = VisualSearchEngine()
        results = await engine.search(
            query=query,
            image_base64=image_base64,
            kb_ids=kb_ids,
            top_k=top_k,
        )

        if not results:
            return json.dumps({
                "status": "empty",
                "message": "未找到视觉相似的内容",
                "results": [],
            }, ensure_ascii=False)

        output = []
        for i, r in enumerate(results):
            item = {
                "rank": i + 1,
                "file_id": r.file_id,
                "page_no": r.page_no,
                "image_path": r.page_image_path,
                "description": r.image_description,
                "similarity": round(r.similarity, 4),
                "source": r.source,
                "text": " ".join(r.text_snippets[:3]) if r.text_snippets else "",
                "text_snippet_count": len(r.text_snippets),
            }
            output.append(item)

        result_str = json.dumps({
            "status": "ok",
            "count": len(output),
            "results": output,
        }, ensure_ascii=False)

        logger.info(
            f"[ImageSearch] found {len(output)} results "
            f"(visual={sum(1 for r in results if r.source in ('visual','both'))}, "
            f"text={sum(1 for r in results if r.source in ('text','both'))})"
        )
        return result_str
