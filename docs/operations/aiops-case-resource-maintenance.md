# AIOps 案例资源自动维护

## 资源归属

诊断案例库是 AIOps 业务空间的系统资源：每个独立运维 Domain 一个案例 KC Collection，
不是全平台共用一个，也不是每次会话创建一个。非 AIOps 空间不创建案例库。
运维手册仍是独立知识体系，不借用其 Collection、模型或解析策略。

## 自动流程

1. AIOps 首次启用、默认 Domain 初始化、新建 AIOps Domain、重新启用时，
   在原业务事务内登记持久化资源事件；事务失败时事件一起回滚。
2. Main API 后台领取持久化记录，提交租约后，通过 KC 专用内部接口幂等创建或接管案例 Collection。
   KC 在可信父 Domain 行锁内重新检查 App、空间启用状态和归属，创建与绑定同事务完成。
3. Collection 身份、资源状态、配置状态和重试信息持久化。进程重启或网络中断后自动重试，
   不依赖请求进程中的一次性异步任务，也不在运行时调用部署脚本。
4. 周期业务对账每批最多扫描 100 个 Domain，使用键集游标遍历；已有空间没有事件时自动补登记。
   这是业务资源对账，不探测表、列、索引或 Schema 版本。
   每个历史空间独立提交，单个异常空间不会阻断同批其他空间，后续轮次仍会自动复查。
5. 空间或 App 停用后暂停使用，不删除 Collection 和案例历史；重新启用复用原资源。

Ammolite 由 Main API lifespan 中的持久化资源 Worker 执行；
KBot4 由现有 Main API notification_worker 进程中的独立循环执行。
部署时必须运行对应后台进程。KC 暂不可达不会回滚已经创建的 Domain。

## 稳定身份与冲突

- Ammolite 使用 Tenant + Domain、source_system=aiops、external_key=diagnosis-cases:<domain_id>。
- KBot4 使用 Domain 与固定资源元数据 owner_app_id=aiops / fixed_resource=true /
  content_kind=diagnosis_case；展示名不是唯一身份依据。
- 已有唯一且有效的资源保留 Collection ID、模型、解析策略和管理员元数据；
  Ammolite 只补完全缺失的所属空间 manage 绑定，不扩展冲突绑定权限。
- 成功结果保存已知 Collection ID。该 ID 对应资源丢失、被停用或身份改变时，
  后续核验报告错误，不创建替代资源、不重新激活、不合并重复库。
- 首次核验即发现身份/绑定冲突时，也持久关闭首次创建权限；后续超时或原库被删除
  不会重新打开该权限。资源纠正后可复核接管，但不会自动创建替代库。
- 通用 KC 物理删除入口拒绝删除系统案例资源；停用 Domain 是保留历史的暂停，
  案例退役是内容治理，不等于删除整个案例 Collection。
- 租约 60 秒，KC 请求最多 10 秒。过期租约可恢复；迟到结果不能覆盖新任务状态。
  暂时失败指数退避、最长 300 秒；身份冲突每 300 秒重新核验，但不自动改写冲突资源。
  已就绪资源每 60 秒复核。发现新旧空间每批间隔 30 秒，无须人工运行后补脚本。

## 资源状态与模型配置

案例库页面独立读取只读 resource 状态接口；状态读取不触发创建，也不阻断案例列表。

资源状态：UNINITIALIZED / QUEUED / RUNNING / READY / FAILED / PAUSED。
模型配置：UNKNOWN / MODELS_REQUIRED / CONFIGURED / TEXT_POLICY_REQUIRED。

新建资源固定为 TEXT 解析、空模型绑定。parser_llm、retrieval_llm、embedding 仍需管理员显式配置，
不得自动选择默认模型或借用 Agent、运维手册模型。CONFIGURED 仅表示绑定完整，
不代表模型实际调用或端到端检索通过。案例还须完成索引、来源核验和人工审核发布后才参与检索。

配置菜单七项及原表单不变。本次不新增配置页面或后台自动选择模型。

## 初始化与发布边界

规范 Schema 增加持久化案例资源记录：Ammolite platform_case_resource，
KBot4 KBOT_PLATFORM_CASE_RESOURCE。规范初始化只登记事件，Collection 创建统一经过 KC 用例。
原定向 provision_aiops_case_collection.py 已删除，不再存在两套创建逻辑。
已有部署在规范 Schema 对齐且后台进程运行后，由周期对账自动纳管，无须手工补建 KC。

内部接口 /internal/v1/knowledge/system/aiops-case-collections:ensure
仅允许 Main API 服务身份、正确 audience 和 knowledge.aiops_case.ensure scope 的双 JWT。
不接受浏览器、普通用户/API Client 或普通静态服务凭证，也不保存或重用用户令牌。

本轮仅修改规范代码、DDL、文档及离线测试，不连接服务器、不执行数据库变更、
不重启服务。真实环境验收由用户手动完成。
