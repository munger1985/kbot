"""App 对纯计算客户端的值对象适配；不解析旧版 KC 格式。"""

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.wire import event_from_wire


class HttpParseComputation:
    def __init__(self, client, context_factory, *, retention_seconds=3600):
        require(0 < retention_seconds <= 3600, "计算制品保留期限无效")
        self.client, self.context_factory, self.retention_seconds = client, context_factory, retention_seconds

    async def submit(self, lease):
        result = await self.client.submit({"run":encode(lease.identity), "source":encode(lease.descriptor),
            "profile":encode(lease.profile), "expires_at":int(lease.lease_until.timestamp()) + self.retention_seconds},
            auth_context=self.context_factory(lease.scope))
        require(set(result) == {"job_ref"}, "计算回执字段不匹配")
        return result["job_ref"]

    async def events(self, lease, *, after_sequence, limit):
        require(lease.compute_ref is not None, "任务尚未绑定计算回执")
        result = await self.client.events(lease.compute_ref, after_sequence=after_sequence, limit=limit,
                                         auth_context=self.context_factory(lease.scope))
        require(isinstance(result.get("events"), list) and len(result["events"]) <= limit, "计算事件页无效")
        return tuple(event_from_wire(item) for item in result["events"])

    async def acknowledge(self, lease, *, sequence, event_digest):
        require(lease.compute_ref is not None, "任务尚未绑定计算回执")
        await self.client.acknowledge(lease.compute_ref, sequence=sequence, event_digest=event_digest,
                                     auth_context=self.context_factory(lease.scope))

    async def cancel(self, lease):
        if lease.compute_ref is not None:
            await self.client.cancel(lease.compute_ref, auth_context=self.context_factory(lease.scope))

    def artifact(self, lease, handle_ref):
        require(lease.compute_ref is not None, "任务尚未绑定计算回执")
        return self.client.artifact(lease.compute_ref, handle_ref, auth_context=self.context_factory(lease.scope))
