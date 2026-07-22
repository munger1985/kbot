"""文档元数据提取器 — 从文档中提取结构化元数据。

每个文档调用 1 次 LLM，同时产出 doc_metadata 和 doc_relation 数据。
提取逻辑仅在 enable_doc_metadata=True 时执行。
"""
from typing import Any
from loguru import logger
from platform_clients import AIModelClient


EXTRACT_PROMPT = """从以下文档内容中提取结构化信息，返回 JSON：

{
  "doc_name": "文档正式名称（从标题或文件名推断，尽量精确）",
  "doc_type": "standard",
  "doc_number": "标准号/编号（如 Q/XXX-2024-B，无则为空字符串）",
  "doc_version": "版本号（如 V2.0，无则为空字符串）",
  "doc_date": "文档日期（ISO 格式如 2024-03-15，从封面/页眉/正文推断，无则为空字符串）",
  "doc_abstract": "文档摘要（2-3句话概述文档主题、范围和用途，不超过200字）",
  "doc_keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
  "doc_references": [
    {
      "doc_name": "被引用的文档名称",
      "chapter": "引用的章节号（如 第3章、第三章，无则为空）",
      "section": "引用的节号（如 3.1.2，无则为空）",
      "context": "引用处的上下文原文（1-2句话）"
    }
  ]
}

规则：
- doc_type 取值: standard（标准规范）/ report（报告）/ contract（合同）/ manual（手册）/ other（其他）
- doc_number 提取标准号、文件编号等，不要提取页码、版本号
- doc_date 从文档封面、页眉页脚、正文日期字段中推断，格式 YYYY-MM-DD
- doc_abstract 用 2-3 句话概括文档主题、适用范围和核心内容
- doc_keywords 提取 3-5 个最能代表文档内容的关键词
- doc_references 提取文档中明确提到的 "参考XX文档"、"依据XX标准"、"见XX文件第X章" 等引用
- 如果文档中没有引用其他文档，doc_references 返回空数组 []
- 只输出 JSON，不要加任何说明文字

文档片段：
{text_snippet}
"""


class MetadataExtractor:
    """文档元数据提取器"""

    def __init__(self, model_client: AIModelClient | None = None):
        self.model_client = model_client or AIModelClient()

    async def extract_from_text(self, text_snapshot: str, llm_model: str,
                                 kb_id: int, file_id: str) -> dict[str, Any] | None:
        """从文本快照中提取结构化元数据（1 次 LLM 调用）。

        Returns:
            {"meta": {...}, "relations": [...]} 或 None
        """
        if not text_snapshot or len(text_snapshot) < 50:
            logger.warning(f"[MetadataExtractor] 文本不足，跳过: {file_id}")
            return None

        prompt = EXTRACT_PROMPT.replace("{text_snippet}", text_snapshot)
        try:
            result = await self.model_client.get_llm_json(
                model_name=llm_model,
                prompt=prompt,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"[MetadataExtractor] LLM JSON 提取失败: {e}")
            return None

        if not result or not isinstance(result, dict):
            return None

        meta = {
            "kb_id": kb_id,
            "file_id": file_id,
            "doc_name": str(result.get("doc_name") or ""),
            "doc_type": str(result.get("doc_type") or "other"),
            "doc_number": str(result.get("doc_number") or ""),
            "doc_version": str(result.get("doc_version") or ""),
            "doc_date": str(result.get("doc_date") or ""),
            "doc_abstract": str(result.get("doc_abstract") or ""),
            "doc_keywords": result.get("doc_keywords") or [],
            "doc_references": result.get("doc_references") or [],
        }

        relations = []
        for ref in meta.pop("doc_references", []):
            if isinstance(ref, dict) and ref.get("doc_name"):
                relations.append({
                    "kb_id": kb_id,
                    "source_file_id": file_id,
                    "target_doc_name": str(ref.get("doc_name") or ""),
                    "target_chapter": str(ref.get("target_chapter") or ref.get("chapter") or ""),
                    "target_section": str(ref.get("target_section") or ref.get("section") or ""),
                    "context_snippet": str(ref.get("context") or ""),
                    "confidence": 0.85,
                })

        logger.info(
            f"[MetadataExtractor] 提取完成: doc_name='{meta['doc_name']}', "
            f"keywords={len(meta['doc_keywords'])}, relations={len(relations)}"
        )
        return {"meta": meta, "relations": relations}

