# AIOps 固定案例资源初始化与模型维护

日期：2026-11-03。两平台的产品语义相同；本文区分 PostgreSQL／Oracle 的部署入口。
这些是供用户手动准备与验收的部署操作说明，不代表已对真实环境执行或验收。
按最新分工，代理只交付代码与本地回归，不代为运行下述资源初始化、模型配置或真实索引操作。

## 1. 资源边界

诊断案例使用固定 `diagnosis-cases` Collection，资料库使用各自独立 Collection。
案例复用通用 KC 的解析、索引、精确 Revision 和引用，不复制手册库，不建立第二套检索引擎。

- 新案例资源：显式 TEXT 解析策略、空模型映射；不从资料库、Agent 或模型目录自动选模型。
- 必需模型角色：`parser_llm`、`retrieval_llm`、`embedding`；不要求 OCR 或视觉模型。
- 资源存在不等于模型配置完成；CONFIGURED 不等于模型健康、索引就绪或案例发布。
- 生成任务只有在确切 KC Revision 索引就绪、Registry 对账和可审核草稿就绪后才成功。
- 草稿索引完成仍须人工审核发布，之后才能被 Agent 检索；退役／撤权版本不能再召回。

配置区菜单和已有表单不改版，不增加 Collection 选择页。固定资源模型通过已有管理 API 显式维护。

## 2. 安全的定向部署入口

两仓均提供 `scripts/deployment/provision_aiops_case_collection.py`。
由受信部署管理员在已经确认的环境执行，操作者仅用于部署审计归属，不代替业务 API 的授权。
先执行只读核验；资源缺失时再去掉 `--check-only` 定向补缺。
以下尖括号参数必须替换为已经核对的真实标识，不包含密码或服务密钥。

Ammolite：

```bash
conda run --no-capture-output -n cube python scripts/deployment/provision_aiops_case_collection.py \
  --tenant-id '<Tenant UUID>' --domain-id '<Domain UUID>' --actor-id '<操作者 UUID>' --check-only
```

Ammolite 要求活动 Tenant 安装及该 Tenant 下由 AIOps 拥有的活动空间。
身份来自 `source_system=aiops`、`external_key=diagnosis-cases:<Domain UUID>` 及所有权元数据，
不从可编辑显示名称推断。资源必须绑定该 Tenant／Domain 的 manage 权限；
缺失绑定只在明确补缺时插入，已有 consume／跨空间绑定不静默提权。

规范 `init_postgresql.py` 在建表和平台边界对齐后，补齐当时已启用的 AIOps 空间案例资源。
全新空库尚无 Tenant／Domain 时不会造空间或账号；随后在平台启用 App／创建 AIOps 空间后，
须把上面的定向操作纳入该空间的部署初始化清单。不要为补一个资源重新初始化全库。

KBot4：

```bash
conda run --no-capture-output -n kbot4 python scripts/deployment/provision_aiops_case_collection.py \
  --domain-id '<Domain 数字 ID>' --actor-id '<部署操作者>' --check-only
```

KBot 要求活动 AIOps App 和仅绑定 AIOps 的活动独立 Domain，不复用 KM Domain。
固定身份为该 Domain 内唯一 `diagnosis-cases` 加所有权元数据。
首次 `initialize_aiops.py` 的新案例资源同样使用 TEXT；已有案例资源只核验，不覆盖管理员策略、模型或重新激活。
该广域首次初始化仍会重置管理员初始密码，不得把它当作案例资源日常维护入口。

定向操作两平台均遵守：

1. 写模式锁定父 Domain；读模式无行锁、无 INSERT／UPDATE／DELETE，也不 commit。
2. 仅补缺失案例 Collection，Ammolite 同事务补所属空间资源绑定；不修改资料库、用户、角色、密码或模型。
3. 已有资源不改 ID、正文、模型、策略、元数据或状态；重复、身份冲突或停用均明确失败。
4. 事务由部署入口提交；异常回滚并关闭连接池。仅输出受控业务诊断，不打印原始驱动异常或连接串。
5. 返回资源 ID、必需／缺失角色及静态配置状态；VISUAL 等不适配策略报告 TEXT_POLICY_REQUIRED，不擅自改写。

本工具不校验或修复 Schema；缺表等错误按部署故障处理。不要在会话、Worker 或 readiness 中调用它。

## 3. 显式配置案例模型

沿用原有认证方式。公共 API 使用当前登录者或已授权 App 客户端，不接受浏览器自行声明 actor／Domain，
不把平台 API Key 转发给 KC。内部管理仅由可信服务使用受众绑定短时 AuthContext JWT。
下列是路径和请求体，不提供绕过认证的直接内网调用命令。

| 平台 | 公共 GET／PUT 路径 |
| --- | --- |
| Ammolite | /api/v1/apps/aiops/domains/{domain_code}/ops/operations-knowledge/case-collection/models |
| KBot4 | /api/v1/apps/aiops/operations-knowledge/case-collection/models |

公开调用要求 `aiops:knowledge_manage`；现有内部路径
`/internal/v1/aiops/operations-knowledge/case-collection/models` 要求 `aiops.manage`。

先 GET 确认固定资源和缺失角色，再 PUT 三个来自当前可用模型目录的确切 ID：

```json
{
  "models": {
    "parser_llm": "<已启用的 LLM UUID>",
    "retrieval_llm": "<已启用的 LLM UUID>",
    "embedding": "<已启用的文本向量模型 UUID>"
  }
}
```

不传 `knowledge_base_id`、Collection ID 或来源空间；后端依据可信身份定位固定案例资源，
校验模型类别与当前可用性。绑定变更遵循两平台已有 KC 的向量模型不可变／解析活动门禁及并发规则，
不是通过修改数据库 JSON 绕过 KC 管理。

若 GET 报 TEXT_POLICY_REQUIRED，部署补缺不会修复它，模型 PUT 也不会悄悄修改策略。
由现有受信 KC 管理能力针对该确切 Collection 修改解析设置：

- Ammolite：PUT /internal/v1/knowledge/tenants/{tenant_id}/collections/{collection_id}/parsing-settings。
- KBot4：PUT /internal/v1/knowledge/domains/{domain_id}/collections/{collection_id}/parsing-settings；
  请求还必须提供当前确切 `expected_row_version`，冲突先重新读取。

请求明确使用 `parse_policy={"parse_strategy":"TEXT","image_scale":2.0,"detect_table_structure":true}`，
并保留确切 `parser_llm_model_id`；不需要的 OCR／VLM 角色明确置 null。
该操作会按 KC 合同更新解析模型角色，不要把省略字段误当成保留已有角色。
如没有合规 KC 管理凭据，交由现有平台管理员处理，不开放内部接口或新增配置表单。

## 4. 验收与当前证据

离线测试覆盖新建、重复执行、只读、父空间隔离、元数据／状态冲突、管理员策略和模型保留、
部署事务提交／回滚及错误输出脱敏；Ammolite 额外覆盖空间资源绑定缺失和跨域／提权阻断。
这些测试使用连接替身，不能证明真实数据库锁、Oracle JSON 驱动、模型调用或 KC 索引已成功。

真实环境需独立记录：明确目标 → 定向核验／补缺 → 显式模型 PUT／GET → 生成草稿 →
确切 KC Revision READY／Registry 对账 → 通知及结果定位 → 人工发布 → 新诊断检索 →
撤权或退役后阻断。未完成最后一条之前，不把静态配置结果称为检索闭环验收。
