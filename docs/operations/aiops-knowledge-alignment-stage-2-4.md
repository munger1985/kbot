# AIOps 运维知识库对齐：检索、生命周期与验收交接

记录日期：2026-11-20。两仓同步。见[实施计划](aiops-knowledge-alignment-implementation-plan.md)与[阶段 1](aiops-knowledge-alignment-stage-1.md)。阶段 2/3 源码、合同和离线回归已落地；阶段 4 真实环境验收待授权。本文不授权数据处置、DDL 或上线。

## 阶段 2：检索与证据交付

- 先确定符合资格的发布版本，再检查其有效正文索引；已发布但索引未就绪、候选超预算、引擎/通道失败、读取预算及授权变化分别记录。只检索业务索引精确指向文档、发布解析和活动代次且就绪的记录；读取和最终组装复核当前 Agent 版本、目标、版本资格及精确索引绑定。
- 主命中由引擎 `primary_hits` 提供，`evidence` 仅作映射。证据分组分别给出主命中、关联项及关联缺口；原始分片内的 `primary.rank` 与最终的 `business_rank` 不混用。多个独立分片不比较局部名次，保留合法证据、标为 `LIMITED`。关联不进入主引用包，不占主结果数量。
- `EVIDENCE_FOUND` 有可用主证据；`NO_ELIGIBLE_RESOURCES` 是完整资格检查后无资料；`NO_MATCH` 是查询范围和通道充分检查后无正文匹配；`NOT_VERIFIED` 表示本次不能充分核验。结果可同时为 `EVIDENCE_FOUND` 和部分覆盖。目录查询不产生相关性结论。
- 主动诊断工作器只将主引用交给证据索引，标识为参考知识，不能证成现场状态。父子执行按当前 Agent 版本冻结文档和父覆盖；Runtime 仍在该文档子集内独立完成回答检索，子引用若未命中父级精确证据键不会被标为父级主命中，子执行成功不提升父覆盖。正文从 Runtime 的精确历史引用读取，同文档关联摘录按父回执精确读取；过期、撤权或跨出子范围则记录关联缺口。现有诊断抽屉展示这些区别，不公开 Scope、Query Ticket、检索回执或游标。
- AIOps 文本预览不传原始视觉载荷，仍保留可授权定位和不可读标记；现有 Runtime 视觉交付仍受管线、权限、实际输入和模型能力约束。实际图文召回质量和视觉交付需真实样本验证。

示例：`retrieval_conclusion: "EVIDENCE_FOUND"`，`coverage: {scope: "PARTIAL", ranking: "LIMITED", gaps: ["INDEX_NOT_READY"], channel_failures: []}`。返回主引用属于已核验部分，不说明其他资料不存在。

## 阶段 3：生命周期和并发

- 目录分别显示仍存资料、当前 Agent 版本和历史版本引用。删除入口保留，按原因禁用并链接已有资料库或 Agent 管理页；不自动删除历史或解除保护。停用不改索引、资料或历史。
- `DELETE` 要求例如 `If-Match: "rv-7"`。缺少为 428，格式错误为 400，版本过期为 412，保护引用为 409，成功为 204。后端核权后事务内锁库行、核验版本、重新统计依赖，再删除；上传登记与写入 Agent 新版本绑定使用同一库行锁，多库按稳定 ID 锁定。页面冲突后刷新一次，不自动重试。
- KBot4 Oracle 规范 `010_ops_knowledge.sql` 已有 `FK_OPS_KNOW_ASSET_BASE`、`FK_OPS_AGENT_VER_KNOW_BASE` 两条非级联外键，只有对应的 ORA-02292 映射为引用冲突。按用户后续要求，Ammolite 在规范实体补齐 `fk_aiops_knowledge_asset_base` 与 `fk_aiops_agent_ver_knowledge_base`，都指向 `aiops_knowledge_base.knowledge_base_id`、`ON DELETE NO ACTION`；只把对应的 PostgreSQL 23503 映射为引用冲突。全新 Schema 初始化会建立这两条外键，既有数据库是否已具备约束、是否有孤立记录仍待实库确认。

## 离线验证

- 定向 Python 回归：KBot4 **323 通过**，Ammolite **317 通过**，各 **75 子测试通过**。覆盖主命中和关联、部分失败、零匹配与未核验、权限变化、精确正文、删除条件及工作器/Runtime。分别使用 `kbot4`、`ammolite` Conda 环境运行两仓 `test_aiops_knowledge_*.py` 相关用例、`test_knowledge_base_status.py`、`test_aiops_operations_knowledge.py`、知识基础的子执行和资料用例、诊断和配置用例以及阶段 0 合同。
- 两仓 OpenAPI 快照全量比较、架构边界、Oracle 规范 DDL / PostgreSQL 基线均通过；Ammolite AIOps 实体元数据检查通过。KBot4 知识页面定向静态合同 2 项通过；Ammolite 前端信息架构初测有两项断言失败，后续已修复并全通过。
- Chromium 加载双端实际知识库卡片资源，模拟所有业务请求，共 **14 项通过**（`apps/portal-web/tests/aiops-knowledge-base-cards.spec.mjs`）。覆盖三类删除阻塞、引用冲突刷新且不重试、版本请求头和启停。诊断前端另做 KBot4 JavaScript 语法与 Ammolite Vue 模板检查，未执行完整诊断浏览器验收。
- 初测全量双仓对齐检查报告飞书通知文档、通知投递单测、日基线单测不一致，已修复并复验通过。Oracle 的 AIOps 实体对照入口会连接数据库读取目录；初测报告的是六张旧试行表遗留在实库，而不是现行 DDL 缺 ORM。本轮只对验收器做离线回归，未再次连接 Oracle。

### 全量差异修复记录

- 补修回归：KBot4 128 项通过（另 29 子测试），Ammolite 150 项通过（另 34 子测试，含 27 项前端合同）。双仓全量对齐、OpenAPI、架构检查通过；Oracle 规范 DDL 检查和 Ammolite PostgreSQL 基线／AIOps ORM 元数据检查通过。
- 双仓同步通知说明和测试。运行授权保持 KBot4 `aiops:proposal:approve`、Ammolite `aiops:change_manage`，测试同时断言另一平台的权限码不能替用；没有修改真实权限。
- 补齐 Ammolite 的巡检日基线测试夹具。前端断言改为验证工作项网格包含布局类，以及巡检卡片用正确数据呈现最近/下次执行时间和未执行状态，不改已经存在的产品布局或文案。
- KBot4 严格验收现区分规范外对象与已映射表的额外列，两者均报错。六个退役对象是 `KBOT_OPS_BACKUP_FIRE`、`KBOT_OPS_BACKUP_PLAN`、`KBOT_OPS_BACKUP_SCHEDULE`、`KBOT_OPS_PROTECTION_LOCATION`、`KBOT_OPS_PROTECTION_OBJECT`、`KBOT_OPS_PROTECTION_RESERVATION`。现行保护业务使用 `KBOT_OPS_PROTECTION_PLAN`、`KBOT_OPS_PROTECTION_RUN`，不改、不兼容旧表。见[本地 Schema 历史记录](local-schema-alignment-20261117.md)。

## 阶段 4：最后执行的真实验收

1. 确认两套非生产验收环境、Tenant/Domain、Agent、测试 Target、授权样本及允许写入的隔离范围。先做只读确认：实际外键状态、删除规则、两类引用的孤立及跨作用域记录数；只记录脱敏汇总。不自动纠正孤立记录。
2. PostgreSQL 若既有库未含两条新外键，须在确认孤立数据处置、锁窗口和获准的实际执行方式后才能补齐。现行初始化基于 metadata 创建新表，不会用重建或自动迁改处理已有表；无完整防线时不签字确认跨所有写入口的删除保护。Oracle 对两个已有知识库约束分别核验已启用且已验证；对六张规范外数据保护试行表另做行数和依赖只读盘点，用户确认目标与归档退役范围后才处理。不自动删除或级联。
3. Oracle、PostgreSQL 分别用两会话验证四组顺序：删除先提交/上传先提交、删除先提交/Agent 绑定先提交。删除先提交时使用方应发现库不存在并回滚；引用先提交时删除应报 409，保留资料和历史。另验空库、草稿、未发布、退役、仅历史引用、过期版本、停用与在途事务，确认没有悬挂记录。
4. 先完成阶段 1 部署前验收，再用获授权的多页中文手册、表格、扫描件、图文和已发布案例核对四种检索结论、局部通道失败、原始与业务排名、中文页段定位、关联可读性、案例前提和禁忌、Agent/Target 切换、禁用、撤权与索引换代。用同一场景比对预览、主动诊断和 Runtime 引用，不以模型回答成功代替覆盖检查。
5. 确认协同切换及回滚：Main API、AIOps API/Worker、Runtime、客户端与前端须配套验证。缺少新版本冻结字段的旧查询委托不续用，按原过期或释放机制处理，不回填。上线与回切前单独核查实际外键、在途任务和合同兼容性；不做破坏性重建。

补修轮次未连接真实数据库，未执行模型、真实资料、端到端验收、DDL、部署、重启、提交或推送。
