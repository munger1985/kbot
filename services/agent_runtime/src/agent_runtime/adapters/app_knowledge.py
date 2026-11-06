"""Runtime 只经固定 App 客户端读取证据，不导入 App 仓储。"""

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import encode


class AppEvidenceContentPort:
    def __init__(self, *, client, auth_context, retrieval_ref, evidence_set):
        require(client.owner == evidence_set.owner_binding_ref, "客户端与证据 owner 不一致")
        self.owner, self.client, self.auth_context = client.owner, client, auth_context
        self.retrieval_ref, self.evidence_set = retrieval_ref, evidence_set
        self.keys = {item.key for item in evidence_set.evidence}

    async def content(self, key, *, max_bytes):
        require(key in self.keys and key.owner_binding_ref == self.owner, "内容不在冻结检索回执")
        iterator = self.client.content(self.retrieval_ref, key.stable_key,
                                      auth_context=self.auth_context, max_bytes=max_bytes)
        try:
            async for block in iterator:
                yield block
        finally:
            await iterator.aclose()

    async def revalidate(self, keys):
        if not keys:
            return ()
        require(set(keys) <= self.keys, "内容复核超出冻结回执")
        keys = tuple(keys)
        require(len(set(keys)) == len(keys), "内容复核请求身份重复")
        rows = []
        for start in range(0, len(keys), 128):
            result = await self.client.read_manifest({"retrieval_ref": self.retrieval_ref,
                "evidence_keys": [encode(key) for key in keys[start:start + 128]]}, auth_context=self.auth_context)
            require(type(result) is dict and type(result.get("items")) is list, "内容复核响应格式错误")
            rows.extend(result["items"])
        by_stable = {key.stable_key: key for key in keys}
        from knowledge_algorithms.contracts.identity import EvidenceKey
        from knowledge_algorithms.contracts.wire import decode
        returned, valid = set(), []
        for row in rows:
            key = decode(EvidenceKey, row["evidence_key"])
            require(by_stable.get(key.stable_key) == key and key not in returned, "内容复核身份重复或替换")
            returned.add(key)
            if row["status"] == "AVAILABLE":
                valid.append(key)
        require(returned == set(keys), "内容复核遗漏身份")
        return tuple(valid)
