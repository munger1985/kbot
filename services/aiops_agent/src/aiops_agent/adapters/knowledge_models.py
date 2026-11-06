"""显式模型空间适配；拒绝未经登记的方向、池化和 revision 混用。"""

import base64
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.projections.contracts import Representation
from knowledge_algorithms.retrieval.maxsim import TokenMatrix


class KnowledgeModelGateway:
    def __init__(self, *, visual_client, dense_client, auth_context, image_reader, models, max_elements):
        self.visual_client, self.dense_client, self.auth_context = visual_client, dense_client, auth_context
        self.image_reader, self.models, self.max_elements = image_reader, dict(models), max_elements

    async def represent(self, value, spec):
        require(spec.space_id in self.models, "模型空间未在可信 App 组合根登记")
        frozen = self.models[spec.space_id]
        require(frozen["model_ref"] == spec.model_ref, "模型绑定与冻结空间不一致")
        direction = value.kind + "_TO_" + spec.input_kind
        require(direction in spec.directions, "当前模型空间没有登记该查询方向")
        if spec.kind == "DENSE":
            require(value.kind == "TEXT" and spec.input_kind == "TEXT" and self.dense_client is not None,
                    "未登记此单向量跨模态能力")
            data = await self.dense_client.represent({frozen["wire_model_field"]: frozen["model_ref"],
                "text": value.text, "input_role": value.input_role}, auth_context=self.auth_context)
            self.validate_space(data, spec)
            result = Representation(spec.space_id, vector=tuple(data["vector"]))
        else:
            require(self.visual_client is not None, "未配置原生多向量模型端口")
            payload = {frozen["wire_model_field"]: frozen["model_ref"],
                       "input_kind": value.kind, "direction": direction}
            if value.kind == "TEXT":
                payload["text"] = value.text
            else:
                raw, mime = await self.image_reader(value.image_handle)
                require(mime in {"image/png", "image/jpeg", "image/webp"}, "模型输入图片格式未登记")
                payload["image_base64"] = base64.b64encode(raw).decode("ascii")
            data = await self.visual_client.represent(payload, auth_context=self.auth_context)
            self.validate_space(data, spec)
            require(len(data["vectors"]) * spec.dimension <= self.max_elements, "模型表示超过预算")
            result = Representation(spec.space_id, matrix=TokenMatrix(spec.space_id,
                tuple(tuple(row) for row in data["vectors"]), tuple(data["mask"])))
        result.validate(spec, self.max_elements)
        return result

    @staticmethod
    def validate_space(data, spec):
        require(type(data) is dict and data.get("model_revision") == spec.revision
                and data.get("processor_revision") == spec.processor_revision
                and data.get("input_template") == spec.input_template and data.get("dimension") == spec.dimension
                and data.get("normalization") == spec.normalization and data.get("similarity") == spec.similarity
                and type(data.get("directions")) is list and set(data["directions"]) >= set(spec.directions),
                "模型返回空间与冻结投影不一致")
