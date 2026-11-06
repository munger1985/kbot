"""相关性搜索与完整资格范围排序不能相互冒充。"""

from dataclasses import asdict, dataclass
from enum import Enum
import math

from .errors import require
from .identity import fingerprint, nonempty


class QueryMode(str, Enum):
    RELEVANCE = "RELEVANCE"
    FILTERED_ORDER = "FILTERED_ORDER"
    EXHAUSTIVE = "EXHAUSTIVE"


@dataclass(frozen=True, slots=True)
class Predicate:
    field: str
    operator: str
    value: str | int | float | bool | None

    def __post_init__(self) -> None:
        nonempty(self.field, self.operator)
        require(type(self.value) in {str, int, float, bool, type(None)}, "谓词必须为标量")
        if type(self.value) is float:
            require(math.isfinite(self.value), "谓词数字必须有限")


@dataclass(frozen=True, slots=True)
class OrderBy:
    field: str
    descending: bool

    def __post_init__(self) -> None:
        nonempty(self.field)
        require(type(self.descending) is bool, "排序方向必须为布尔值")


@dataclass(frozen=True, slots=True)
class QueryInput:
    mode: QueryMode
    requested_count: int
    output_types: tuple[str, ...]
    text: str | None = None
    image_handles: tuple[str, ...] = ()
    combination: str | None = None
    predicates: tuple[Predicate, ...] = ()
    order_by: tuple[OrderBy, ...] = ()

    def __post_init__(self) -> None:
        require(isinstance(self.mode, QueryMode), "查询模式无效")
        require(type(self.requested_count) is int and self.requested_count > 0, "主命中数量无效")
        require(bool(self.output_types) and len(set(self.output_types)) == len(self.output_types),
                "输出类型必须非空且不重复")
        nonempty(*self.output_types, *self.image_handles)
        require(set(self.output_types) <= {"TEXT", "FIGURE", "PHOTO", "TABLE", "PAGE", "REGION", "MIXED"},
                "未登记的结果类型")
        if self.text is not None:
            nonempty(self.text)
        if self.mode == QueryMode.RELEVANCE:
            require(bool(self.text or self.image_handles), "相关性搜索需要文本或受控图片句柄")
            require(not self.order_by, "属性排序必须使用完整资格范围排序模式")
        else:
            require(not self.text and not self.image_handles,
                    "语义相关全集不可证明；原生排序仅接受可执行谓词")
            require(bool(self.order_by), "原生排序及穷尽分页必须显式提供稳定排序")
        if len(self.image_handles) > 1:
            require(self.combination in {"ANY", "ALL", "COMPOSED"}, "多图查询需声明组合语义")
        else:
            require(self.combination is None, "单输入不能声明多图组合")

    @property
    def query_hash(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """可执行字段来自固定 App 注册表，不来自模型或调用者自报。"""

    predicate_fields: frozenset[str]
    operators: frozenset[str]
    sortable_fields: frozenset[str]
    stable_identity_field: str
    ordered_modes: frozenset[QueryMode]

    def validate(self, query: QueryInput) -> None:
        for predicate in query.predicates:
            require(predicate.field in self.predicate_fields
                    and predicate.operator in self.operators, "谓词未在 App 后端登记")
        if query.mode != QueryMode.RELEVANCE:
            require(query.mode in self.ordered_modes, "后端不支持该原生查询模式")
            require(all(order.field in self.sortable_fields for order in query.order_by),
                    "排序字段未登记")
            require(query.order_by[-1].field == self.stable_identity_field,
                    "分页排序必须以稳定身份消除并列")
