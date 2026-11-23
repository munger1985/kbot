"""用有界矩阵块执行视觉 MaxSim，不改变已发布表示或候选集合。"""

import asyncio
from collections.abc import Awaitable, Callable
import math

import numpy as np

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.retrieval import ChannelRequest
from knowledge_algorithms.projections.contracts import Representation, RepresentationSpec


def _active_rows(representation: Representation | None, spec: RepresentationSpec) -> np.ndarray:
    matrix = representation.matrix if representation is not None else None
    require(matrix is not None, "多向量通道需要 token 矩阵", stage="maxsim")
    if representation.space_id != spec.space_id or matrix.space_id != spec.space_id:
        raise AlgorithmError(ErrorCode.MODEL_SPACE_MISMATCH, "查询和索引表示空间不匹配", stage="maxsim")
    require(len(matrix.vectors[0]) == spec.dimension, "矩阵维度与空间不匹配", stage="maxsim")
    rows = [row for row, valid in zip(matrix.vectors, matrix.mask) if valid]
    vectors = np.array(rows, dtype=np.float64)
    if spec.similarity == "COSINE":
        norms = np.fromiter((math.hypot(*row) for row in rows), dtype=np.float64, count=len(rows))
        require(bool(np.all((norms > 0) & np.isfinite(norms))), "余弦向量范数为零或溢出", stage="maxsim")
        vectors /= norms[:, None]
    return vectors


def prepare_maxsim(request: ChannelRequest) -> Callable[[Representation], Awaitable[float]]:
    spec = request.channel.representation_spec
    require(spec is not None and spec.kind == "MULTIVECTOR", "多向量通道缺少空间规格", stage="maxsim")
    if request.channel.direction not in spec.directions:
        raise AlgorithmError(ErrorCode.MODALITY_UNAVAILABLE, "模型未验证该检索方向", stage="maxsim")
    query = _active_rows(request.representation, spec)
    # 限制得分缓冲区和每个取消检查点之间的乘加工作量。
    tile = min(256, max(1, math.isqrt(16_777_216 // spec.dimension)))

    async def score(representation: Representation) -> float:
        document = _active_rows(representation, spec)
        maxima = np.full(len(query), -np.inf)
        for i in range(0, len(query), tile):
            query_rows = query[i:i + tile]
            best = maxima[i:i + tile]
            for j in range(0, len(document), tile):
                # 使用 NumPy 单线程内核，避免并发检索各自启动 BLAS 线程池。
                with np.errstate(over="ignore", invalid="ignore"):
                    dots = np.einsum("ik,jk->ij", query_rows, document[j:j + tile], optimize=False)
                require(bool(np.isfinite(dots).all()), "原生点积溢出", stage="maxsim")
                np.maximum(best, dots.max(axis=1), out=best)
                await asyncio.sleep(0)
        try:
            total = math.fsum(maxima)
        except (OverflowError, ValueError) as exc:
            raise AlgorithmError(ErrorCode.INVALID_CONTRACT, "MaxSim 合计溢出", stage="maxsim") from exc
        require(math.isfinite(total), "MaxSim 合计溢出", stage="maxsim")
        return total

    return score
