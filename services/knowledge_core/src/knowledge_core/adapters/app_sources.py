"""执行器的固定 App 源注册表；目标地址只来自组合根配置。"""

from knowledge_algorithms.contracts.errors import require


class RemoteAppSource:
    def __init__(self, job, client, context):
        self.descriptor, self.job, self.client, self.context = job.source, job, client, context

    def chunks(self):
        from knowledge_algorithms.contracts.wire import encode
        return self.client.source({"job_ref":self.job.run.job_ref, "identity":encode(self.job.run)},
                                  auth_context=self.context, max_bytes=self.descriptor.byte_count)


class FixedAppSourceRegistry:
    def __init__(self, *, clients, caller_service):
        require(set(clients) == {"knowledge_retrieval", "aiops"}, "源注册表必须固定登记两个 App")
        self.clients, self.caller_service = dict(clients), caller_service

    def reader(self, job, context):
        require(context.app_id == job.access.owner, "源委托 App 不匹配")
        # 派生只允许本任务源读取的内部委托；原主体及 Domain/Tenant 不变。
        delegated = context.model_copy(update={"calling_service":self.caller_service, "scopes":("knowledge.source.read",)})
        return RemoteAppSource(job, self.clients[job.access.owner], delegated)
