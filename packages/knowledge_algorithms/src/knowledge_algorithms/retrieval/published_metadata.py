"""进程内复用已验证的索引元数据，不缓存授权或检索回执。"""

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import monotonic

from ..contracts.errors import require


class PublishedMetadataCache:
    def __init__(self, *, max_entries=4, max_bytes=32 * 1024 * 1024, ttl=300, clock=monotonic):
        require(max_entries > 0 and max_bytes > 0 and ttl > 0, "索引元数据缓存预算无效")
        self._max_entries, self._max_bytes, self._ttl = max_entries, max_bytes, ttl
        self._clock = clock
        self._entries = OrderedDict()
        self._bytes = 0
        # 限制冷加载同时解码的对象数；已命中的读取不等待冷加载。
        self._loader = asyncio.Lock()

    def _remove(self, key):
        self._bytes -= self._entries.pop(key)[1]

    def _lookup(self, key):
        now = self._clock()
        for expired, (expires_at, _, _) in tuple(self._entries.items()):
            if expires_at <= now:
                self._remove(expired)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key][2]
        return None

    async def get(self, key, build):
        cached = self._lookup(key)
        if cached is not None:
            return cached
        async with self._loader:
            cached = self._lookup(key)
            if cached is not None:
                return cached
            value, weight = await build()
            require(type(weight) is int and weight >= 0, "索引元数据缓存大小无效")
            if weight <= self._max_bytes:
                while self._entries and (len(self._entries) >= self._max_entries
                                         or self._bytes + weight > self._max_bytes):
                    self._remove(next(iter(self._entries)))
                self._entries[key] = (self._clock() + self._ttl, weight, value)
                self._bytes += weight
            return value


class _EvidenceView(Mapping):
    def __init__(self, templates, expires_at):
        self._templates, self._expires_at = templates, expires_at

    def __getitem__(self, key):
        return replace(self._templates[key], expires_at=self._expires_at)

    def __iter__(self):
        return iter(self._templates)

    def __len__(self):
        return len(self._templates)

    def __contains__(self, key):
        return key in self._templates


@dataclass(frozen=True, slots=True)
class PublishedMetadata:
    index: object
    items: Mapping
    relations: tuple
    vectors: object
    text_payloads: Mapping
    estimated_bytes: int

    def bind(self, *, store, max_bytes, expires_at):
        written = set()

        async def materialize(item):
            handle = item.content_handle
            text = self.text_payloads.get(handle)
            if text is not None and handle not in written:
                stored = await store.put_bytes(text.encode("utf-8"), max_bytes=max_bytes)
                require(stored.handle_ref == handle, "检索文本摘要与已验证允许视图不一致")
                written.add(handle)

        return (self.index, _EvidenceView(self.items, expires_at), self.relations,
                self.vectors.fork(), materialize)
