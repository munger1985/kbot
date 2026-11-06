"""原生排序模式的独立入口，不调用 embedding、词法或融合。"""

from ..contracts.errors import require
from ..contracts.identity import AuthorizedScope
from ..contracts.query import QueryInput, QueryMode
from ..ports.ordered import OrderedPage, OrderedSearchPort


async def retrieve_ordered(
    backend: OrderedSearchPort, query: QueryInput, scope: AuthorizedScope,
) -> OrderedPage:
    require(query.mode in {QueryMode.FILTERED_ORDER, QueryMode.EXHAUSTIVE}, "该入口不执行相关性搜索")
    backend.binding.check_owner(scope.owner_binding_ref)
    backend.capabilities.validate(query)
    snapshot = await backend.snapshot(scope)
    snapshot.validate_scope(backend.binding, scope)
    page = await backend.search_ordered(query, scope, snapshot)
    require(page.snapshot == snapshot and page.query_hash == query.query_hash,
            "排序响应范围或查询指纹不匹配", stage="ordered")
    require(len(page.hits) <= query.requested_count and len(set(page.hits)) == len(page.hits),
            "排序响应主命中数量或身份无效", stage="ordered")
    require(page.scope_exhausted == (page.continuation is None), "范围未穷尽必须提供受控续取入口")
    for key in page.hits:
        backend.binding.check_owner(key.owner_binding_ref)
    return page
