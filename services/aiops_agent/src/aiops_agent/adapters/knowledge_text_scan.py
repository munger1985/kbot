"""可选扫描只生成当前允许视图文本，不读取原文件或公共索引。"""

from hashlib import sha256
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.ports.text_scan import AuthorizedTextSource


class PublishedTextSourceProvider:
    def __init__(self, backend):
        self.backend = backend

    async def validate(self, scope, snapshot):
        await self.backend.check(scope, snapshot)

    async def sources(self, scope, snapshot):
        await self.validate(scope, snapshot)
        for indexed in self.backend.index.units:
            unit = indexed.projected.unit
            if unit.text:
                yield AuthorizedTextSource(unit.key, unit.text, unit.sources,
                    sha256(unit.text.encode()).hexdigest(), fingerprint(encode(unit.sources)))
