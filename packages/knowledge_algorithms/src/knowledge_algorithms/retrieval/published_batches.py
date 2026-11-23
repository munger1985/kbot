"""已发布投影按批次读取，向量评分不持有全库表示。"""

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
import json
import math

from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import fingerprint
from ..contracts.wire import decode
from ..projections.contracts import Representation
from .exact import vector_score
from .fusion import NativeHit


@dataclass(frozen=True, slots=True)
class PublishedBatch:
    handle: str
    projection_ref: str
    sequence: int
    payload_hash: str
    unit_count: int
    representation_limit: int

    async def read_units(self, store, max_bytes):
        data = bytearray()
        # 读取结束时验证对象句柄摘要；只缓冲一个批次。
        async with aclosing(store.stream(self.handle)) as blocks:
            async for part in blocks:
                if len(data) + len(part) > max_bytes:
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "单个已发布索引批次超过读取预算",
                                         stage="index_batch")
                data.extend(part)
        require(bool(data), "发布投影批次为空")
        try:
            batch = json.loads(data)
            require(type(batch) is dict and set(batch) == {"projection_ref", "sequence", "units", "payload_hash"}
                    and type(batch["sequence"]) is int and batch["sequence"] == self.sequence
                    and batch["projection_ref"] == self.projection_ref and batch["payload_hash"] == self.payload_hash
                    and type(batch["units"]) is list and len(batch["units"]) == self.unit_count
                    and fingerprint(batch["units"]) == self.payload_hash, "投影批次身份或摘要不一致")
        except AlgorithmError:
            raise
        except (ValueError, TypeError) as exc:
            raise AlgorithmError(ErrorCode.INVALID_CONTRACT, "投影批次不是合法的索引对象",
                                 stage="index_batch") from exc
        return batch["units"]


class PublishedVectorBatches:
    def __init__(self, *, store, max_bytes, max_elements, prepare_multivector=None):
        self.store, self.max_bytes, self.max_elements = store, max_bytes, max_elements
        self.prepare_multivector = prepare_multivector
        self._by_space = {}
        self._lock = asyncio.Lock()

    def add(self, batch, members):
        for space, positions in members.items():
            self._by_space.setdefault(space, []).append((batch, tuple(positions)))

    def fork(self):
        # 缓存只复用空间目录。每个检索保留自己的评分锁和解码缓冲区。
        view = PublishedVectorBatches(store=self.store, max_bytes=self.max_bytes, max_elements=self.max_elements,
                                      prepare_multivector=self.prepare_multivector)
        view._by_space = self._by_space
        return view

    async def search(self, index, request, *, candidate_filter=None):
        spec = request.channel.representation_spec
        hits = []
        score_multivector = None
        # 同一检索的多个表示空间不同时占用解码缓冲区。
        async with self._lock:
            for batch, positions in self._by_space.get(spec.space_id, ()):
                eligible = [(position, indexed) for position, indexed in positions
                            if index.eligible(indexed, request.query)
                            and (candidate_filter is None or candidate_filter(indexed))]
                if not eligible:
                    continue
                values = await batch.read_units(self.store, self.max_bytes)
                elements = 0
                for position, indexed in eligible:
                    matches = [raw for raw in values[position]["representations"] if raw["space_id"] == spec.space_id]
                    require(len(matches) == 1, "投影批次的向量目录与已验证位置不一致")
                    representation = decode(Representation, matches[0])
                    representation.validate(spec, batch.representation_limit)
                    elements += (len(representation.vector) if representation.vector is not None
                                 else sum(len(row) for row in representation.matrix.vectors))
                    if elements > self.max_elements:
                        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "单个索引批次超过表示预算",
                                             stage="index_batch")
                    if request.channel.kind == "MULTIVECTOR" and self.prepare_multivector is not None:
                        if score_multivector is None:
                            score_multivector = self.prepare_multivector(request)
                        score = await score_multivector(representation)
                    else:
                        score = vector_score(request, representation)
                    require(math.isfinite(score), "召回分数溢出")
                    hits.append(NativeHit(indexed.key, score))
                    del representation, matches
                del values
                await asyncio.sleep(0)
        # 合并后再截断，分批边界不改变全库排名或相同分数的顺序。
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.key.stable_key)))
