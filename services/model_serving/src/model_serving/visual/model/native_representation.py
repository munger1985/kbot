"""原生 ColQwen token 表示；保留 padding mask，不做均值池化。"""

import asyncio
import base64
from io import BytesIO
import math
import warnings


class NativeColQwenRepresentation:
    async def represent(self, *, input_kind, text=None, image_base64=None, direction):
        config = self.config
        if (not config.model_revision or not config.processor_revision
                or not config.input_template or direction not in config.retrieval_directions):
            raise ValueError("模型未登记不可变 revision、处理器或检索方向")
        if not direction.startswith(input_kind + "_TO_"):
            raise ValueError("模型输入模态与检索方向不一致")
        if input_kind not in {"TEXT", "IMAGE"} or (text is not None) == (image_base64 is not None):
            raise ValueError("多模态输入必须唯一")
        if (input_kind == "TEXT") != (text is not None):
            raise ValueError("输入模态不一致")
        if self._processor is None or self._model is None:
            raise ValueError("模型实例尚未启动")
        image = None
        if image_base64 is not None:
            if len(image_base64) > config.max_image_bytes * 4 // 3 + 8:
                raise ValueError("图像超过字节预算")
            raw = base64.b64decode(image_base64, validate=True)
            if len(raw) > config.max_image_bytes:
                raise ValueError("图像超过字节预算")
            from PIL import Image
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as source:
                    if source.width * source.height > config.max_image_pixels or getattr(source, "n_frames", 1) != 1:
                        raise ValueError("图像像素或帧数超过预算")
                    source.load()
                    image = source.convert("RGB")
        elif not isinstance(text, str) or not text.strip() or len(text) > config.max_query_chars:
            raise ValueError("查询文本超过预算或为空")

        def operation():
            import torch
            inputs = (self._processor.process_queries([text]) if input_kind == "TEXT"
                      else self._processor.process_images([image]))
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
            if outputs.ndim != 3 or outputs.shape[0] != 1 or outputs.shape[2] != config.dimension:
                raise ValueError("原生多向量输出形状与模型配置不一致")
            if outputs.shape[1] * outputs.shape[2] > config.max_representation_elements:
                raise ValueError("原生多向量输出超过预算")
            attention = inputs.get("attention_mask")
            if attention is None or tuple(attention.shape) != tuple(outputs.shape[:2]):
                raise ValueError("原生多向量缺少对应有效 token mask")
            vectors = outputs[0].detach().cpu().float().tolist()
            mask = [bool(value) for value in attention[0].detach().cpu().tolist()]
            if not any(mask) or not all(math.isfinite(v) for row in vectors for v in row):
                raise ValueError("原生表示包含非有限值或没有有效 token")
            # 明确按 token 进行 float32/L2 规范化，避免低精度范数误差；
            # 不合并 token，也不将多向量均值池化为单向量。
            normalized = []
            for row, valid in zip(vectors, mask):
                norm = math.hypot(*row)
                if valid and (not math.isfinite(norm) or norm <= 0):
                    raise ValueError("有效 token 的原生表示范数无效")
                normalized.append([value / norm for value in row] if valid else row)
            vectors = normalized
            return {"vectors": vectors, "mask": mask, "dimension": config.dimension,
                    "model_revision": config.model_revision, "processor_revision": config.processor_revision,
                    "input_template": config.input_template, "normalization": "L2", "similarity": "DOT",
                    "directions": list(config.retrieval_directions)}
        try:
            return await self._run_cpu_inference(operation)
        finally:
            if image is not None:
                image.close()

    async def _run_cpu_inference(self, operation):
        # 取消等待者不等于终止原生推理线程；必须保留槽位直到线程真正退出。
        async with self._inference_lock:
            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                raise
