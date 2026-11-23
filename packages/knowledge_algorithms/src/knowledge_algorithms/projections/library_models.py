"""从可信模型目录冻结库专属索引空间；查询只使用发布清单中的空间。"""

from dataclasses import replace
import re

from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.retrieval import ChannelSpec
from ..contracts.wire import decode
from .contracts import ProjectionProfile, RepresentationSpec


async def library_projection(base, models, resolve):
    specs = []
    for role, kind, input_kind in (("embedding", "DENSE", "TEXT"), ("visual_embedding", "MULTIVECTOR", "IMAGE")):
        model_id = (models or {}).get(role)
        if not model_id:
            continue
        try:
            binding = await resolve(role, model_id)
            value = dict(binding["representation"])
            revision = binding["revision"]
            require(str(binding["model_id"]) == str(model_id) and value["kind"] == kind
                    and value["input_kind"] == input_kind and re.fullmatch(r"[a-f0-9]{64}", revision) is not None,
                    "模型目录返回了不匹配的表示空间")
            spec = decode(RepresentationSpec, {
                "model_ref": str(model_id), "inference_ref": binding["inference_ref"],
                "binding_revision": revision, "revision": value.pop("model_revision"),
                **{**value, "directions": [direction for direction in value["directions"] if direction.endswith("_TO_" + input_kind)]},
            })
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise AlgorithmError(ErrorCode.MODEL_SPACE_MISMATCH,
                "该库绑定的向量模型不可用或未登记完整表示规格", stage="index_model") from exc
        specs.append(spec)
    return replace(base, representations=tuple(specs))


def published_profile(base, records, *, retain_registered=False, allows=None):
    """查询向量与已发布的代次绑定，不用当前库配置重新解释历史向量。"""
    registered = {c.representation_spec.space_id for c in base.channels if c.representation_spec is not None}
    spaces = {}
    for record in records:
        manifest = record["generation_manifest"]
        for spec in decode(ProjectionProfile, manifest["profile"]).representations:
            if ((spec.binding_revision is not None or (retain_registered and spec.space_id in registered))
                and (allows is None or allows(spec, manifest))):
                spaces[spec.space_id] = spec
    require(len(spaces) <= 64, "发布表示空间数量超过检索预算")
    channels = [c for c in base.channels if c.representation_spec is None]
    for key, spec in sorted(spaces.items()):
        for direction in sorted(spec.directions):
            channel = key + ":" + direction
            channels.append(ChannelSpec(channel, channel, spec.kind, direction, 1., 100, 60., False, spec))
    return replace(base, channels=tuple(channels))
