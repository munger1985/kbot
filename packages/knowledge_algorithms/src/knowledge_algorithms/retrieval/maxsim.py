"""多向量原生 late interaction 精确参考实现，不执行均值池化。"""

from dataclasses import dataclass
import math

from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import nonempty


@dataclass(frozen=True, slots=True)
class TokenMatrix:
    space_id: str
    vectors: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        nonempty(self.space_id)
        require(bool(self.vectors) and len(self.mask) == len(self.vectors), "矩阵及 mask 形状无效")
        require(all(type(v) is bool for v in self.mask) and any(self.mask), "mask 无有效 token")
        dimension = len(self.vectors[0])
        require(dimension > 0 and all(len(v) == dimension for v in self.vectors), "向量维度不一致")
        require(all(math.isfinite(x) for v in self.vectors for x in v), "矩阵含非有限数字")


@dataclass(frozen=True, slots=True)
class MaxSimSpec:
    space_id: str
    dimension: int
    similarity: str
    directions: frozenset[str]

    def __post_init__(self) -> None:
        nonempty(self.space_id)
        require(type(self.dimension) is int and self.dimension > 0, "表示维度无效")
        require(self.similarity in {"DOT", "COSINE"}, "未实现该原生相似度")
        require(bool(self.directions), "模型必须声明已验证方向")


def _similarity(a: tuple[float, ...], b: tuple[float, ...], metric: str) -> float:
    if metric == "COSINE":
        na, nb = math.hypot(*a), math.hypot(*b)
        require(0 < na < math.inf and 0 < nb < math.inf, "余弦向量范数为零或溢出")
        value = math.fsum((x / na) * (y / nb) for x, y in zip(a, b))
    else:
        try:
            value = math.fsum(x * y for x, y in zip(a, b))
        except (OverflowError, ValueError) as exc:
            raise AlgorithmError(ErrorCode.INVALID_CONTRACT, "原生点积溢出", stage="maxsim") from exc
    require(math.isfinite(value), "原生相似度溢出", stage="maxsim")
    return value


def maxsim(
    query: TokenMatrix, document: TokenMatrix, *, spec: MaxSimSpec,
    direction: str, query_weights: tuple[float, ...] | None = None,
) -> float:
    if query.space_id != spec.space_id or document.space_id != spec.space_id:
        raise AlgorithmError(ErrorCode.MODEL_SPACE_MISMATCH, "查询和文档表示空间不匹配", stage="maxsim")
    if direction not in spec.directions:
        raise AlgorithmError(ErrorCode.MODALITY_UNAVAILABLE, "模型未验证该检索方向", stage="maxsim")
    require(len(query.vectors[0]) == spec.dimension == len(document.vectors[0]), "矩阵维度与空间不匹配")
    weights = query_weights if query_weights is not None else (1.0,) * len(query.vectors)
    require(len(weights) == len(query.vectors)
            and all(math.isfinite(w) and w >= 0 for w in weights), "查询权重无效")
    require(any(w > 0 for w, valid in zip(weights, query.mask) if valid), "有效查询权重全为零")
    active_document = [v for v, valid in zip(document.vectors, document.mask) if valid]
    terms = [weight * max(_similarity(q, d, spec.similarity) for d in active_document)
             for q, valid, weight in zip(query.vectors, query.mask, weights) if valid and weight > 0]
    try:
        score = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise AlgorithmError(ErrorCode.INVALID_CONTRACT, "MaxSim 合计溢出", stage="maxsim") from exc
    require(math.isfinite(score), "MaxSim 合计溢出", stage="maxsim")
    return score
