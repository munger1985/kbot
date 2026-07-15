"""视觉索引数据清理工具。
在删除文件/知识库/重新解析时调用。
已合并 page_visual_index：统一清理 extracted_images 表。
"""

import shutil
from pathlib import Path
from loguru import logger
from dao.repositories import ExtractedImageRepository


async def cleanup_visual_data(
    file_storage: str,
    kb_id: str,
    file_ids: list[str] | None = None,
) -> None:
    """清理视觉索引数据（Oracle VECTOR 表 + 图片文件目录）。"""
    try:
        repo = ExtractedImageRepository()
        if file_ids:
            await repo.delete_by_file_ids(file_ids)
        else:
            await repo.delete_by_kb_id(kb_id)
        logger.info("[VisualCleanup] db: kb=%s files=%s", kb_id, file_ids or "ALL")

        root = Path(file_storage).resolve()
        kb_dir = root / kb_id
        if file_ids and kb_dir.exists():
            for sub in kb_dir.iterdir():
                if sub.is_dir():
                    for fid in file_ids:
                        img_dir = sub / fid
                        if img_dir.exists():
                            shutil.rmtree(img_dir, ignore_errors=True)
                            logger.info("[VisualCleanup] removed: %s", img_dir)
        elif not file_ids and kb_dir.exists():
            shutil.rmtree(kb_dir, ignore_errors=True)
            logger.info("[VisualCleanup] removed: %s", kb_dir)
    except Exception as e:
        logger.warning("[VisualCleanup] failed: %s", e)
