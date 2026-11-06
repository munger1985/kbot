"""先归并同一通道全部分片，再产生一份全局名次和一次 RRF 投票。"""

from dataclasses import dataclass
import math

from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import EvidenceKey, RetrievalSnapshot, nonempty


@dataclass(frozen=True, slots=True)
class NativeHit:
    key: EvidenceKey
    score: float

    def __post_init__(self) -> None:
        require(self.key.projection_ref is not None, "检索候选必须绑定确切投影")
        require(math.isfinite(self.score), "原生分数必须有限")


@dataclass(frozen=True, slots=True)
class ChannelShard:
    channel_key: str
    signal_family: str
    space_id: str
    scoring_contract: str
    statistics_ref: str
    snapshot: RetrievalSnapshot
    hits: tuple[NativeHit, ...]
    higher_is_better: bool = True
    approximate: bool = False
    exhausted: bool = True
    status_reason: str | None = None

    def __post_init__(self) -> None:
        nonempty(self.channel_key, self.signal_family, self.space_id,
                 self.scoring_contract, self.statistics_ref)
        if any(hit.key.owner_binding_ref != self.snapshot.owner_binding_ref for hit in self.hits):
            raise AlgorithmError(ErrorCode.SCOPE_CLOSURE_BROKEN,
                                 "通道返回了其他固定后端的对象", stage="channel")


@dataclass(frozen=True, slots=True)
class RankedChannel:
    channel_key: str
    signal_family: str
    snapshot: RetrievalSnapshot
    hits: tuple[NativeHit, ...]
    approximate: bool
    exhausted: bool
    status_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RankContribution:
    channel_key: str
    global_rank: int
    native_score: float
    contribution: float


@dataclass(frozen=True, slots=True)
class FusedHit:
    key: EvidenceKey
    score: float
    contributions: tuple[RankContribution, ...]


def merge_channel_shards(shards: tuple[ChannelShard, ...]) -> RankedChannel:
    require(bool(shards), "至少需要一个通道分片", stage="merge")
    first = shards[0]
    signature = lambda s: (s.channel_key, s.signal_family, s.space_id,
                           s.scoring_contract, s.statistics_ref,
                           s.snapshot, s.higher_is_better)
    if any(signature(shard) != signature(first) for shard in shards):
        raise AlgorithmError(ErrorCode.INCOMPARABLE_CHANNEL,
                             "分片空间、评分、语料统计或快照不可比", stage="merge")
    by_key: dict[EvidenceKey, NativeHit] = {}
    for shard in shards:
        for hit in shard.hits:
            previous = by_key.get(hit.key)
            require(previous is None or previous.score == hit.score,
                    "相同通道同一证据分数矛盾", stage="merge")
            by_key[hit.key] = hit
    ordered = sorted(by_key.values(), key=lambda hit: (
        -hit.score if first.higher_is_better else hit.score, hit.key.stable_key))
    return RankedChannel(first.channel_key, first.signal_family, first.snapshot,
                         tuple(ordered), any(s.approximate for s in shards),
                         all(s.exhausted for s in shards),
                         next((s.status_reason for s in shards if s.status_reason), None))


def reciprocal_rank_fusion(
    channels: tuple[RankedChannel, ...], *, weights: dict[str, float], constant: float,
) -> tuple[FusedHit, ...]:
    """只调整顺序；参数必须显式提供，绝不按相关性阈值删除身份。"""
    require(math.isfinite(constant) and constant >= 0, "RRF 常数必须有限且非负")
    names = [channel.channel_key for channel in channels]
    families = [channel.signal_family for channel in channels]
    require(len(set(names)) == len(names) and len(set(families)) == len(families),
            "同通道或同信号族必须先归并，不能重复投票", stage="fusion")
    require(set(weights) == set(names), "RRF 权重必须与通道一一对应", stage="fusion")
    require(all(math.isfinite(v) and v > 0 for v in weights.values()), "权重必须有限且为正")
    require(not channels or all(channel.snapshot == channels[0].snapshot for channel in channels),
            "融合不得混用授权或索引快照", stage="fusion")
    contributions: dict[EvidenceKey, list[RankContribution]] = {}
    for channel in sorted(channels, key=lambda c: c.channel_key):
        keys = [hit.key for hit in channel.hits]
        require(len(set(keys)) == len(keys), "通道名次包含重复身份", stage="fusion")
        for rank, hit in enumerate(channel.hits, 1):
            require(hit.key.owner_binding_ref == channel.snapshot.owner_binding_ref,
                    "融合证据不属于固定后端", stage="fusion")
            value = weights[channel.channel_key] / (constant + rank)
            require(math.isfinite(value), "RRF 分值溢出", stage="fusion")
            contributions.setdefault(hit.key, []).append(RankContribution(
                channel.channel_key, rank, hit.score, value))
    result = []
    for key, values in contributions.items():
        try:
            score = math.fsum(value.contribution for value in values)
        except (OverflowError, ValueError) as exc:
            raise AlgorithmError(ErrorCode.INVALID_CONTRACT, "RRF 合计溢出", stage="fusion") from exc
        require(math.isfinite(score), "RRF 合计溢出", stage="fusion")
        result.append(FusedHit(key, score, tuple(values)))
    return tuple(sorted(result, key=lambda hit: (-hit.score, hit.key.stable_key)))
