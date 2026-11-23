"""公开写入入口按当前作用域的有效模型目录校验新增绑定。"""

from fastapi import HTTPException, Request

from main_api.api.models import load_model_catalog
from platform_core.contracts.knowledge_parsing import LibraryIndexModels, LibraryParseModels
from platform_core.dictionary import ModelCategory


async def validate_parse_models(request: Request, models: LibraryParseModels | LibraryIndexModels | None, *, current=None) -> None:
    if models is None:
        return
    previous = current or {}
    changes = {
        role: (str(model_id), category)
        for role, category in (("ocr", ModelCategory.OCR), ("vlm", ModelCategory.VLM),
                               ("embedding", ModelCategory.TXT_EMBEDDING), ("visual_embedding", ModelCategory.IMG_EMBEDDING))
        if (model_id := getattr(models, role, None)) and str(model_id) != str(previous.get(role) or "")
    }
    if not changes:
        return
    catalog = await load_model_catalog(request)
    available = {(str(row["model_id"]), int(row["category"])) for row in catalog}
    for role, binding in changes.items():
        if binding not in available:
            raise HTTPException(422, {"code": "KNOWLEDGE_PARSE_MODEL_UNAVAILABLE",
                "message": f"{role.upper()} 模型的类别不匹配或不在当前可用目录中"})
