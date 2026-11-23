"""显式模型空间适配；拒绝未经登记的方向、池化和 revision 混用。"""

import base64
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.projections.contracts import Representation
from knowledge_algorithms.retrieval.maxsim import TokenMatrix
from platform_clients.knowledge_computation import KnowledgeComputationClientError
from platform_core.contracts.model import NATIVE_VISUAL_IMAGE_INPUT_UNSUPPORTED, NATIVE_VISUAL_MAX_IMAGE_BYTES


class KnowledgeModelGateway:
    def __init__(self, *, visual_client, dense_client, auth_context, image_reader, models, max_elements):
        self.visual_client, self.dense_client, self.auth_context = visual_client, dense_client, auth_context
        self.image_reader, self.models, self.max_elements = image_reader, dict(models), max_elements

    async def represent(self, value, spec):
        if spec.binding_revision is not None:
            model_payload = {"model_id": spec.model_ref, "expected_revision": spec.binding_revision}
        else:
            require(spec.space_id in self.models, "模型空间未在可信 App 组合根登记")
            frozen = self.models[spec.space_id]
            require(frozen["model_ref"] == spec.model_ref, "模型绑定与冻结空间不一致")
            model_payload = {"model_id": spec.model_ref}
        direction = value.kind + "_TO_" + spec.input_kind
        require(direction in spec.directions, "当前模型空间没有登记该查询方向")
        if spec.kind == "DENSE":
            require(value.kind == "TEXT" and spec.input_kind == "TEXT" and self.dense_client is not None,
                    "未登记此单向量跨模态能力")
            data = await self.dense_client.represent({**model_payload,
                "text": value.text, "input_role": value.input_role}, auth_context=self.auth_context)
            self.validate_space(data, spec)
            result = Representation(spec.space_id, vector=tuple(data["vector"]))
        else:
            require(self.visual_client is not None, "未配置原生多向量模型端口")
            payload = {**model_payload,
                       "input_kind": value.kind, "direction": direction}
            if value.kind == "TEXT":
                payload["text"] = value.text
            else:
                raw, mime = await self.image_reader(value.image_handle)
                require(mime in {"image/png", "image/jpeg", "image/webp"}, "模型输入图片格式未登记")
                if len(raw) > NATIVE_VISUAL_MAX_IMAGE_BYTES:
                    raise AlgorithmError(ErrorCode.IMAGE_INPUT_UNSUPPORTED, "图片超过视觉表示接口字节上限",
                                         stage="visual_input")
                payload["image_base64"] = base64.b64encode(raw).decode("ascii")
            try:
                data = await self.visual_client.represent(payload, auth_context=self.auth_context)
            except KnowledgeComputationClientError as exc:
                if (value.kind == "IMAGE" and exc.status_code == 422
                        and exc.code == NATIVE_VISUAL_IMAGE_INPUT_UNSUPPORTED):
                    raise AlgorithmError(ErrorCode.IMAGE_INPUT_UNSUPPORTED, "视觉模型不接受此图像输入规格",
                                         stage="visual_input") from None
                if exc.status_code == 409 and exc.code == "MODEL_REVISION_CHANGED":
                    raise AlgorithmError(ErrorCode.MODEL_REVISION_CHANGED, "冻结的视觉模型版本已改变",
                                         stage="visual_model") from None
                raise
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
