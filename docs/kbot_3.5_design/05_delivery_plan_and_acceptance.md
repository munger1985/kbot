# 3.5 实施计划、迁移与验收

实施的逐步顺序与每步完成门槛见 [实施路线图](06_implementation_roadmap.md)；本文保留阶段目标、迁移规则和总体验收标准。

## 实施阶段

1. **基础服务与入库**：建立 `KBOT_KC_*` DDL、Entity/Repository、健康检查、Bundle 上传 API、版本与 Job 状态机；Portal 改为一次 Bundle 上传。
2. **解析迁移**：从现有 `FileProcessor` 提取 Docling/OCR/VLM、切块与 embedding 逻辑为 Worker handler，通过 claim/result 协议回传，不访问旧 File/Chunk Repository。
3. **检索与关系**：实现 Parse View、Evidence、Discovery、Profile、确定性 Relation，以及 Excel `SPREADSHEET` View 和结构化工件。
4. **V2 问文链路重构**：以 KC Client 重写 `KnowledgeRetrievalSkillV2` 为 Discovery → Evidence → Citation Pack 流程；同步提供 V2 的 Doc Orchestrator、Root Agent 引用 DTO 与 SSE 输出，不适配 `TxtBaseSearchResult`。
5. **双轨切流与评测**：将本期知识重新入库或迁移为 Bundle，并按来源/App/Agent 将流量切至 V2；V1 File/Chunk 读写链路与旧表继续保留、独立运行。稳定 KM Asset 后，再接普通 KB、项目、工单等 Bundle Adapter。File Dataset/NL2SQL、多 Agent 改造仍留给后续版本。

## 迁移与运行规则

3.5 的 V2 不兼容 3.4 的 `KB → File → TxtChunk` 模型，但系统在过渡期并行保留 V1。需要升级的知识按来源重新投递或迁移为 Collection/Bundle/Document Version；`kbot_md_kb_files` 与 `KBOT_BIZ_TXT_EMBEDDING` 继续仅服务 V1。V2 的 KC 故障通过 Job 重试、租约过期重领和当前 Version 保持策略恢复，不能回写或查询旧表；业务路由是否切回 V1 必须由显式运维/产品决策执行，不得自动降级。

新服务加入 `start_kbot.sh` 的进程与端口检查，复用 `core` 的配置、日志、Oracle 异步连接和模型 HTTP Client。KC 不 import Agent、Skill、旧 `services/kb` 或 `services/search`。

## 验收标准

- 一个 Bundle 可原子接收主信息和多附件；重复投递幂等，新来源修订不覆盖旧版本。
- 已绑定 Agent 的 Collection 删除返回 `COLLECTION_IN_USE`；未绑定 Collection 删除后，其 Evidence 与对象存储内容均不可再访问。
- Parser 崩溃后任务可在租约过期后重领；局部附件失败正确呈现 `PARTIAL`。
- Discovery 能召回正确 Bundle/Document；Evidence 能定位附件、章节、页码/区域并保留版本链路。
- Excel 可定位子表并产生结构化工件；数值意图不会误以 VLM 文本计算。
- V2 问文访问只经过 KC V2 API 与重构后的 V2 Skill，不存在旧 File/Chunk 查询、`TxtBaseSearchResult` 兼容适配或请求内双后端回退；V1 问文链路独立可用。
- 建立并持续报告 Bundle/Document/Evidence Recall@K、页码定位准确率、跨附件覆盖率、解析/索引耗时、任务失败率与重试率。

## 上线检查

上线前完成来源修订与权限测试、错误重放测试、附件下载失败测试、不同解析视图去重测试、并发上传幂等测试，以及 V1/V2 路由隔离、Discovery → Evidence → Citation Pack 的端到端质量测试。每次解析器、Embedding 模型或分块策略升级均应以新 Parse View/Version 灰度验证，不覆盖既有可用结果。
