"""固定模型配置的单向量回执；不用请求自报的 revision 冒充实际模型。"""

import math


async def dense_representation(model, *, text, input_role):
    config = model.config
    if not (config.model_revision and config.processor_revision and config.input_template
            and config.representation_dimension and input_role in {"QUERY", "DOCUMENT"}):
        raise ValueError("模型尚未登记不可变表示空间")
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        raise ValueError("表示输入超过预算")
    response = await model.embed(texts=[text], batch_size=1, is_query=input_role == "QUERY")
    if len(response.data) != 1 or response.data[0].index != 0:
        raise ValueError("模型表示回执身份无效")
    vector = response.data[0].embedding
    if len(vector) != config.representation_dimension or any(
            type(value) not in {int, float} or not math.isfinite(value) for value in vector):
        raise ValueError("模型表示维度或数值无效")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("模型返回零向量或数值溢出")
    return {"vector": [float(value / norm) for value in vector],
        "dimension": config.representation_dimension, "model_revision": config.model_revision,
        "processor_revision": config.processor_revision, "input_template": config.input_template,
        "normalization": "L2", "similarity": "COSINE", "directions": ["TEXT_TO_TEXT"]}
