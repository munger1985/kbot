"""有界精确索引基线；统计覆盖整个授权语料，不把局部分数当全局名次。"""

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

from ..contracts.errors import require
from ..contracts.identity import EvidenceKey, fingerprint
from ..contracts.query import QueryInput
from ..contracts.retrieval import ChannelRequest
from ..projections.contracts import ProjectedUnit
from .fusion import NativeHit
from .maxsim import MaxSimSpec, _similarity, maxsim


def lexical_tokens(text: str) -> tuple[str, ...]:
    """中文字符及相邻双字、Unicode 单词；不带业务词或领域例外。"""
    value = unicodedata.normalize("NFKC", text).casefold()
    result = []
    for match in re.finditer(r"[\u3400-\u9fff]+|[^\W_]+", value):
        token = match.group()
        if "\u3400" <= token[0] <= "\u9fff":
            result.extend(token)
            result.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            result.append(token)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class IndexedUnit:
    projected: ProjectedUnit
    attributes: tuple[tuple[str, str], ...]
    image_sha256: str | None = None
    visual_hash: int | None = None

    @property
    def key(self):
        return self.projected.unit.key


def output_type(kind):
    if kind in {"FIGURE", "IMAGE"}:
        return "FIGURE"
    if kind in {"PHOTO", "TABLE", "PAGE", "REGION"}:
        return kind
    return "TEXT"


class ExactIndex:
    def __init__(self, units, *, max_units, max_text_chars, max_vector_elements,
                 tokenizer=lexical_tokens, tokenizer_revision="unicode-cjk-bigram/v1",
                 bm25_k1=1.2, bm25_b=0.75):
        require(all(type(n) is int and n > 0 for n in
                    (max_units, max_text_chars, max_vector_elements)), "精确索引预算无效")
        require(math.isfinite(bm25_k1) and bm25_k1 > 0
                and math.isfinite(bm25_b) and 0 <= bm25_b <= 1, "BM25 参数无效")
        self.units = tuple(units)
        require(len(self.units) <= max_units, "授权语料超出精确基线预算，需登记更大规模索引后端")
        require(len({unit.key for unit in self.units}) == len(self.units), "索引身份重复")
        self.tokenizer, self.k1, self.b = tokenizer, bm25_k1, bm25_b
        text_size, vector_size = 0, 0
        self.terms, self.frequency = {}, Counter()
        for indexed in self.units:
            unit = indexed.projected.unit
            text_size += len(unit.text) + len(unit.context)
            require(text_size <= max_text_chars, "索引文本超过显式内存预算")
            terms = Counter(tokenizer(unit.text + "\n" + unit.context))
            self.terms[indexed.key] = terms
            self.frequency.update(terms.keys())
            for representation in indexed.projected.representations:
                vector_size += (len(representation.vector) if representation.vector is not None
                                else sum(len(row) for row in representation.matrix.vectors))
            require(vector_size <= max_vector_elements, "索引表示超过显式内存预算")
        self.average_length = (sum(sum(t.values()) for t in self.terms.values()) / len(self.units)
                               if self.units else 0)
        self.statistics_ref = fingerprint({"revision": tokenizer_revision,
            "units": sorted(u.key.stable_key for u in self.units),
            "df": sorted(self.frequency.items()), "average_length": self.average_length,
            "k1": bm25_k1, "b": bm25_b})

    def eligible(self, indexed, query: QueryInput):
        unit = indexed.projected.unit
        if "MIXED" not in query.output_types and output_type(unit.kind.value) not in query.output_types:
            return False
        values = dict(indexed.attributes, stable_key=indexed.key.stable_key,
                      kind=unit.kind.value, document_version_ref=indexed.key.document_version_ref)
        for predicate in query.predicates:
            actual, expected = values.get(predicate.field), predicate.value
            if predicate.operator == "EQ":
                matched = actual == expected
            elif predicate.operator == "NE":
                matched = actual != expected
            elif predicate.operator == "PREFIX":
                matched = isinstance(actual, str) and isinstance(expected, str) and actual.startswith(expected)
            elif predicate.operator in {"LT", "LE", "GT", "GE"}:
                require(actual is None or type(actual) is type(expected), "结构化条件类型不一致")
                matched = actual is not None and {
                    "LT": lambda: actual < expected, "LE": lambda: actual <= expected,
                    "GT": lambda: actual > expected, "GE": lambda: actual >= expected
                }[predicate.operator]()
            else:
                require(False, "结构化条件操作未登记")
            if not matched:
                return False
        return True

    def search(self, request: ChannelRequest, *, image_sha256=None, visual_hash=None):
        channel, scores = request.channel, []
        query_terms = Counter(self.tokenizer(request.query.text or ""))
        for indexed in self.units:
            if not self.eligible(indexed, request.query):
                continue
            score = None
            if channel.kind == "LEXICAL":
                terms = self.terms[indexed.key]
                length = sum(terms.values())
                score = 0.0
                for word, query_count in query_terms.items():
                    count = terms.get(word, 0)
                    if not count:
                        continue
                    inverse = math.log1p((len(self.units) - self.frequency[word] + 0.5)
                                         / (self.frequency[word] + 0.5))
                    denominator = count + self.k1 * (
                        1 - self.b + self.b * length / max(self.average_length, 1))
                    score += query_count * inverse * count * (self.k1 + 1) / denominator
                if score == 0:
                    continue
            elif channel.kind in {"DENSE", "MULTIVECTOR"}:
                spec = channel.representation_spec
                representations = [r for r in indexed.projected.representations if r.space_id == spec.space_id]
                require(len(representations) <= 1, "同单元的表示空间重复")
                if not representations:
                    continue
                representation = representations[0]
                require(request.representation is not None, "查询表示缺失")
                if channel.kind == "DENSE":
                    require(representation.vector is not None and request.representation.vector is not None
                            and len(representation.vector) == len(request.representation.vector) == spec.dimension,
                            "查询和索引向量维度或种类不一致")
                    score = _similarity(request.representation.vector, representation.vector, spec.similarity)
                else:
                    require(representation.matrix is not None and request.representation.matrix is not None,
                            "多向量通道不能使用单向量")
                    score = maxsim(request.representation.matrix, representation.matrix,
                        spec=MaxSimSpec(spec.space_id, spec.dimension, spec.similarity, spec.directions),
                        direction=channel.direction)
            elif channel.kind == "NEAR_DUPLICATE":
                require(image_sha256 is not None, "近重复需要受控图片指纹")
                if indexed.image_sha256 == image_sha256:
                    score = 1.0
                elif visual_hash is not None and indexed.visual_hash is not None:
                    score = 1 - (visual_hash ^ indexed.visual_hash).bit_count() / 64
            if score is not None:
                require(math.isfinite(score), "召回分数溢出")
                scores.append(NativeHit(indexed.key, score))
        return tuple(sorted(scores, key=lambda hit: (-hit.score, hit.key.stable_key)))

    def ordered(self, query):
        values = [unit for unit in self.units if self.eligible(unit, query)]
        # 排序在整个授权资格范围执行，而不是对相关性 Top-K 做后处理。
        for order in reversed(query.order_by):
            def sort_key(unit):
                attributes = dict(unit.attributes, stable_key=unit.key.stable_key,
                                  kind=unit.projected.unit.kind.value,
                                  document_version_ref=unit.key.document_version_ref)
                value = attributes.get(order.field)
                return value is not None, value or ""
            values.sort(key=sort_key, reverse=order.descending)
        return tuple(unit.key for unit in values)
