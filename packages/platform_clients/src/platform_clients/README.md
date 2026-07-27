# Platform Clients

本包拥有模型服务、Knowledge Core、SQL Executor 和运维集成的跨服务 HTTP
Client。Client 统一处理内部身份、超时和响应归一化，不 import 目标服务的
Repository、Entity 或 Application Service。

`KnowledgeCoreClient` 是 Main API、Agent Runtime 与 KC 之间的唯一 Python
调用边界。Portal API Key 不得进入这里；调用方传入可信 `AuthContext`，Client
为每次请求签发限定 KC audience 的短期内部 JWT。
