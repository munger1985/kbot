# KM Asset 统一搜索详细设计

> 状态：目标设计，尚未实现。
>
> 本文描述 KM Asset 新搜索方案的目标合同、执行语义和迁移边界，不代表当前生产行为。
> 当前实现仍以 `docs/architecture/`、OpenAPI 快照和代码为准。

## 1. 背景

当前 KM Asset 搜索把问题先路由为问文、问数或 Data Query 优先的混合链路。结构化
查询负责确定 Asset 集合，Knowledge Core（KC）随后只在已确定的 Bundle 范围内检索
正文证据。这种方案对精确统计和引用边界有效，但存在以下限制：

1. 主题候选主要依赖标题、产品和解决方案字段的字符串包含匹配，不是真正的正文语义召回；
2. 一个问题只能稳定表达一个主题筛选，不能完整表达多主题、多偏好及嵌套布尔条件；
3. 结构化结果过早按展示数量截断，正文检索无法找回截断范围外更符合偏好的 Asset；
4. 次级偏好只进入正文问句，没有独立的命中事实、排序语义和证据约束；
5. 多语言扩展、结果合并和最终排序之间缺少统一计划，可能覆盖用户明确排序；
6. `READY`、当前 Revision 和 Bundle 可检索性尚未成为所有搜索入口共同的不可绕过边界；
7. 主题相关性依赖检索和证据判定，不能形成可精确统计的封闭集合，当前回答却可能让用户误以为
   Top-K 结果就是全部数量。

新设计用一个统一的 Asset 搜索计划表达问题，再按所需能力组合 Data Query 与 KC。
问文、问数和混合查询不再是互斥的意图类型，而是同一计划的不同执行形态。

## 2. 目标与非目标

### 2.1 目标

1. 一个问题可同时包含多个元数据条件、多个语义条件、多个软偏好和排除条件；
2. 精确元数据过滤、计数、分组、排序保持可复现和可审计；
3. 语义检索覆盖标题、受控元数据、Discovery Profile 和正文 Evidence；
4. 候选召回与用户展示数量解耦，最终排序前不得按展示上限截断；
5. 每个最终 Asset 都能解释满足了哪些硬条件、命中了哪些偏好以及证据来自哪里；
6. 所有搜索入口统一排除不可搜索 Asset；
7. 纯问文、纯问数和混合问题共用同一规划、校验、追踪及引用协议；
8. 模型只负责受限语义规划和候选内判断，不生成 SQL、不扩大权限范围、不新增候选；
9. 在全文、向量或模型部分降级时保持结果边界准确，并明确披露能力降级；
10. 保留现有 `QUERY_RESULT`、`CITATION_PACK`、`GROUNDED_ANSWER` 和 Qn/Cn 展示协议。
11. 明确不实现语义主题总数；遇到此类请求时说明限制，并返回 3 至 5 个较新相关 Asset 参考。
12. 任何成功回答都必须包含 Asset 明细和与该 Asset 正确绑定的引用，不能只返回解释、数字或图表。

### 2.2 非目标

1. 不建立第二套 Knowledge Core、向量库或通用 Data Query 服务；
2. 不允许 Agent Runtime 直接访问业务数据库或拼接 SQL；
3. 不用产品名、行业名或具体案例词在 Python 中维护特例规则；
4. 不实现“关于某主题的 Asset 总共有多少个”这类语义计数，也不把近似向量 Top-K 的数量包装成
   完整 Asset 总数；
5. 不在本设计中改变 KM Asset 入库、审批或来源系统业务状态；
6. 不新增 3.x 兼容路由、双读双写或旧协议适配层；
7. 不修改 `integrations/apex/**`。

## 3. 设计原则

### 3.1 条件先于路径

规划器先识别用户要求，不先选择某条固定链路。执行路径由计划中的能力需求确定。

### 3.2 硬边界与排序分离

Domain、授权、可搜索状态和用户硬条件决定资格；相关度、软偏好和时间只对合格集合排序。
任何分数都不能补偿硬条件不满足。

### 3.3 召回与判定分离

全文、向量和元数据匹配用于扩大候选召回；条件判定器逐项确认候选是否满足要求。召回分数
不是最终事实，尤其不能直接证明多个语义条件同时成立。

### 3.4 展示上限与候选预算分离

用户默认最多看到 10 个 Asset；候选召回预算由服务配置和查询复杂度决定，不等于 10。
候选融合、硬条件判定和偏好排序全部完成后，才应用展示上限。

### 3.5 证据与结论同源

元数据事实由 QueryResult 证明，正文语义事实由 Citation Pack 证明。没有支持证据时，系统
不得宣称对应语义硬条件或软偏好已经命中。

### 3.6 准确失败优于错误成功

强制范围无法验证或精确过滤失败时，查询失败或返回证据不足，不得悄悄放宽条件。语义主题
总数请求不进入计数执行：系统明确说明不能准确统计，再提供少量较新且有证据的相关 Asset。

## 4. 总体架构

```text
用户问题与会话上下文
          │
          ▼
Context Rewrite
          │ 独立、无歧义的问题
          ▼
Asset Search Planner
          │ ASSET_SEARCH_PLAN.v1
          ▼
Plan Validator / Compiler
          │
          ├───────────────┬────────────────┬─────────────────┐
          ▼               ▼                ▼                 ▼
Metadata Executor   Content Discovery  Aggregate Executor  Evidence Executor
   Data Query        Knowledge Core       Data Query        Knowledge Core
          │               │                │                 │
          └───────────────┴───────┬────────┴─────────────────┘
                                  ▼
                         Asset Candidate Fusion
                                  │
                                  ▼
                       Requirement Support Judge
                                  │ ASSET_MATCH_RESULT.v1
                                  ▼
                     Deterministic Sort / Limit / Count
                                  │
                                  ▼
                         Response Composer
                                  │
                                  ▼
              GROUNDED_ANSWER + Qn/Cn + 运行诊断
```

### 4.1 服务职责

| 组件 | 职责 | 不允许承担的职责 |
|---|---|---|
| Agent Runtime | 上下文改写、统一规划、DAG 编译、候选融合、条件判定、排序、回答组合 | 直接 SQL、绕过 KC 读取正文 |
| Data Query | 执行受控元数据过滤、投影、统计、分组和精确排序 | 正文语义判断、生成自然语言答案 |
| Knowledge Core | Asset/Bundle 发现、全文与向量召回、Evidence 检索、正文支持证据 | 决定 Domain 权限、生成结构化统计 |
| KM Asset App | 投影当前 Asset 状态、维护 Asset 与 Bundle Revision 的一致映射 | 生成回答、自由解释用户意图 |
| Model Serving | 按功能模型配置执行规划、Embedding、语义判断和回答生成 | 保存查询权限或业务状态 |

### 4.2 新增与复用的 Artifact

| Artifact | 用途 | 是否公开 |
|---|---|---:|
| `ASSET_SEARCH_PLAN.v1` | 冻结统一搜索语义、解析日期和有效范围 | 否 |
| `ASSET_CANDIDATE_SET.v1` | 保存各召回通道候选、来源和原始分数 | 否 |
| `ASSET_MATCH_RESULT.v1` | 保存 Asset 条件命中矩阵、排序键和排除原因 | 否 |
| `QUERY_RESULT.v1` | 保存结构化查询事实，继续产生 Qn | 是，脱敏后 |
| `CITATION_PACK` | 保存 Bundle 正文证据，继续产生 Cn | 是，按现有权限 |
| `GROUNDED_ANSWER` | 最终回答、使用的 Qn/Cn 和披露信息 | 是 |

Artifact 在 Run 内不可变。重试产生新的 Task Attempt，但不得原地改写已发布 Artifact。

## 5. 可搜索 Asset 的系统不变量

### 5.1 定义

Asset 只有同时满足以下条件才属于搜索全集：

```text
SEARCHABLE(asset) =
    asset.domain_id = trusted_domain_id
    AND asset.ingestion_status = 'READY'
    AND asset 是来源对象的当前有效修订
    AND asset.kc_bundle_id IS NOT NULL
    AND asset.kc_bundle_revision_id IS NOT NULL
    AND 对应 Bundle Revision 可检索
    AND 用户对 Agent、Collection 和 Bundle 具有访问权限
```

`asset_status` 是来源系统业务状态，不替代 `ingestion_status`。FAILED、处理中、缺少 Bundle、
旧 Revision 或越权 Asset 均不进入搜索、列表和统计。

这里不保留“最新来源 Revision 失败时继续搜索上一成功 Revision”的降级路径。来源对象一旦出现
更新，只有新 Revision 完整入库并成为 READY 后才能重新进入搜索全集；处理中或失败期间该 Asset
不可搜索。这一选择以结果与当前来源事实一致为优先，不以旧内容可用性为优先。

### 5.2 数据投影

保留 `KBOT_V_KM_ASSET_CURRENT` 作为运维和同步状态投影；新增查询专用只读视图
`KBOT_V_KM_ASSET_SEARCHABLE`，只暴露满足数据侧可搜索条件的当前 Asset。KM Asset
受管语义模型改为绑定该视图，并不再允许模型规划 `ingestion_status` 条件。

KC 侧只检索 ACTIVE Parse View、当前 Bundle Revision 和授权 Collection。Agent Runtime
对 Data Query 与 KC 返回的 Asset/Bundle 映射再次取交集，作为跨服务一致性防线。

### 5.3 发布时序

KM Asset 状态转换遵循：

```text
元数据同步完成
  → Bundle Revision 建立
  → Parse / Index / Profile / Discovery Index 完成
  → 当前 Revision 映射持久化
  → ingestion_status 原子转换为 READY
  → 进入 KBOT_V_KM_ASSET_SEARCHABLE
```

任一步失败均保持非 READY。新来源 Revision 进入处理后，旧成功 Revision 不再作为当前内容
继续参与搜索；只有新 Revision 完整 READY 后才重新进入搜索视图。同一 Asset 在搜索视图中最多
只能出现一个 Revision。

## 6. 统一搜索合同

### 6.1 顶层合同

目标合同为 `AssetSearchPlan.v1`。`v1` 是新合同自身的首个版本，不沿用现有路由枚举版本。

```json
{
  "contract_version": "AssetSearchPlan.v1",
  "query_text": "帮我搜一下关于 OAC 的 Asset，最好是金融欺诈案例",
  "language": "zh-CN",
  "operation": "LIST",
  "target": "ASSET",
  "answer_detail": "BRIEF",
  "criteria": [],
  "eligibility_expression": {},
  "preferences": [],
  "measures": [],
  "group_by": [],
  "projection": [],
  "order_by": [],
  "include_total_count": false,
  "display_limit": 10,
  "result_assets": {
    "mode": "PRIMARY",
    "target_count": 10,
    "selection": "REQUESTED_ORDER"
  },
  "unsupported_requests": [],
  "evidence_policy": {},
  "ambiguities": [],
  "time_zone": "Asia/Shanghai"
}
```

顶层字段含义：

| 字段 | 约束 |
|---|---|
| `operation` | `ANSWER`、`LIST`、`COUNT`、`GROUP`、`COMPARE` |
| `target` | `ASSET`、`CONTENT`；COUNT/GROUP 的目标必须是 ASSET |
| `answer_detail` | `NONE`、`BRIEF`、`DETAILED`；控制主结果之外是否需要解释 |
| `criteria` | 原子条件定义，最多由配置限制，不在模型 Prompt 中硬编码技术上限 |
| `eligibility_expression` | 只引用硬条件，表达嵌套 `ALL/ANY/NOT` |
| `preferences` | 有序软偏好，不参与资格判断 |
| `measures` | 受控 measure，例如 `asset_count` |
| `group_by` | 受控结构化维度 |
| `projection` | 列表需要返回的业务字段 |
| `order_by` | 用户明确排序；字段必须可排序且被查询投影 |
| `include_total_count` | LIST 是否同时要求完整总数；只有用户明确要求时才开启 |
| `display_limit` | LIST 的最终展示上限，最大 10；其他 Operation 为 null；与内部候选预算无关 |
| `result_assets` | 必需；声明最终 Asset 是主结果还是支撑结果、目标数量和选择规则 |
| `unsupported_requests` | 规范化后的不支持能力，例如 `SEMANTIC_TOTAL_COUNT`；由 Validator 生成 |
| `evidence_policy` | 各任务要求的证据完整性 |
| `ambiguities` | 影响结果集合的未决歧义；非空时不得执行 |
| `time_zone` | 从可信用户/Domain 配置注入，模型不能自由改写 |

Domain、用户、Agent、Collection、系统可搜索状态和服务预算不由 LLM 输出，编译器从冻结的
Execution Spec、AuthContext 和配置注入 `effective_scope`。

### 6.2 原子条件

```json
{
  "criterion_id": "c1",
  "kind": "SEMANTIC_CONCEPT",
  "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
  "operator": "RELATED_TO",
  "values": ["OAC"],
  "occurrence": "MUST",
  "evidence_requirement": "METADATA_OR_CONTENT",
  "resolved_concept": null
}
```

字段定义：

| 字段 | 值域与含义 |
|---|---|
| `kind` | `METADATA`、`SEMANTIC_CONCEPT`、`EXACT_PHRASE`、`IDENTIFIER`、`CONTENT_TYPE` |
| `field_scope` | 允许搜索的逻辑字段或 `CONTENT`，不得出现物理列名 |
| `operator` | 由 kind 限制，例如 `EQ`、`IN`、`BETWEEN`、`CONTAINS`、`RELATED_TO` |
| `values` | 保留用户值；日期在规划阶段解析成 ISO 边界 |
| `occurrence` | `MUST`、`MUST_NOT`；软偏好单独进入 `preferences` |
| `evidence_requirement` | `QUERY_RESULT`、`CONTENT`、`METADATA_OR_CONTENT` |
| `resolved_concept` | 可选受控概念 ID、规范名称和别名版本，不保存模型推理文本 |

### 6.3 布尔表达式

```json
{
  "node_type": "ALL",
  "children": [
    {"node_type": "REF", "criterion_id": "c1"},
    {
      "node_type": "ANY",
      "children": [
        {"node_type": "REF", "criterion_id": "c2"},
        {"node_type": "REF", "criterion_id": "c3"}
      ]
    },
    {
      "node_type": "NOT",
      "child": {"node_type": "REF", "criterion_id": "c4"}
    }
  ]
}
```

验证规则：

1. 每个 `REF` 必须引用一个存在的硬条件；
2. `ALL` 和 `ANY` 至少包含两个子节点；
3. `NOT` 只能包含一个子节点；
4. 条件不得循环引用或出现不可达定义；
5. 用户没有表达 OR 时，多个硬条件默认组合为 ALL；该语义由 Planner 给出并由校验器审计；
6. 软偏好不得出现在资格表达式中；
7. `occurrence=MUST_NOT` 是用户强度的审计标签，对应 `REF` 在表达式中的负极性；Validator
   必须拒绝 MUST_NOT 出现在正极性位置或 MUST 出现在非用户要求的负极性位置，避免双重否定；
8. Compiler 只以规范化后的表达式求值，不再额外根据 occurrence 取反。

### 6.4 软偏好

```json
{
  "preference_id": "p1",
  "criterion": {
    "kind": "SEMANTIC_CONCEPT",
    "field_scope": ["TITLE", "SOLUTION", "CONTENT"],
    "operator": "RELATED_TO",
    "values": ["金融欺诈"]
  },
  "priority": 1,
  "evidence_requirement": "METADATA_OR_CONTENT"
}
```

软偏好按用户表达顺序形成确定性优先级，不由模型产生任意浮点权重。“最好 A，其次 B”对应
priority 1 和 2；“必须 A”必须进入硬条件，不能降级为偏好。

### 6.5 输出任务

| Operation | 必需字段 | 禁止行为 |
|---|---|---|
| `ANSWER` | 正文语义条件、CONTENT 证据策略、3 至 5 个支撑 Asset | 用 Top-K 数量冒充总数 |
| `LIST` | Asset 投影、稳定排序、展示上限 | 在融合前应用展示上限 |
| `COUNT` | `asset_count`、精确元数据条件、同集合 3 至 5 个支撑 Asset | 包含任意语义资格条件 |
| `GROUP` | measure、group_by、稳定排序、同集合 3 至 5 个支撑 Asset | 用正文 Chunk 直接分组 |
| `COMPARE` | 至少两个可解析比较对象或集合、3 至 5 个支撑 Asset | 混淆 Asset 与 Evidence 数量 |

一个问题可以同时产生 LIST 和解释性回答，但主 Operation 只能有一个。附加需求通过
`answer_detail`、`projection` 和 `evidence_policy` 表达，不并列生成两个互相冲突的主任务。

### 6.6 最终 Asset 输出合同

`result_assets` 在所有可执行计划中必需：

```json
{
  "mode": "SUPPORTING",
  "target_count": 5,
  "selection": "RECENT_WITHIN_RESULT"
}
```

规则如下：

1. LIST 使用 `mode=PRIMARY`，Asset 就是主结果，数量服从用户要求并且最多 10；
2. ANSWER、COUNT、GROUP 和 COMPARE 使用 `mode=SUPPORTING`，默认目标为 5 个，允许 3 至 5 个；
3. 支撑 Asset 必须来自主回答实际使用的同一资格集合，不能另做一个放宽条件的搜索；
4. 用户没有指定支撑 Asset 顺序时，ANSWER 优先直接证据覆盖，再按日期从新到旧；COUNT/GROUP
   在同一精确过滤集合中按日期从新到旧；
5. 实际合格结果少于 3 个时按实际返回，不补足、不跨 Domain、不放宽 MUST，也不伪造引用；
6. 零个合格 Asset 时可以返回无结果或证据不足，但不能生成一个没有 Asset 来源的成功业务答案；
7. 每个 Asset 至少绑定一个证明其入选原因的 Qn 或 Cn。需要元数据和正文共同满足时，两类引用
   都必须存在。

## 7. 规划与校验

### 7.1 上下文改写

Context Rewrite 只在真实指代或省略时继承会话条件：

- “其中上个月的呢”继承上一轮集合，并增加日期条件；
- “改成作者 Alice”替换作者条件；
- “搜索 OAC Asset”是完整新问题，不继承上一轮金融主题；
- 改写结果必须保留用户原始数量、排序、否定和偏好强度。

改写 Artifact 同时记录 `inherited_constraints`、`replaced_constraints` 和
`dropped_constraints`，供调试和回放使用。

### 7.2 LLM 规划

规划模型只能输出 `AssetSearchPlan.v1`。Prompt 提供：

1. 当前 KM Asset 逻辑字段和允许操作符；
2. 受控产品、方案、行业和内容类型词表的可用版本；
3. 当前日期、时区及相对日期解析规则；
4. MUST、SHOULD、NOT 和用户明确排序的语义说明；
5. 正反例，重点覆盖多条件组合而非单主题例句。

不在应用代码中使用“关于”“最好”“最新”等关键词列表决定路由。语言理解由规划模型输出，
确定性代码只验证合同和跨字段一致性。

### 7.3 确定性校验

Plan Validator 至少执行：

1. Schema 严格校验，拒绝额外字段；
2. 字段、measure、operator 和 value type 必须存在于受管目录；
3. 日期必须转换为明确的半开或闭区间，并记录解析来源；
4. 作者匹配使用标准化邮箱或用户名，不做不受控模糊匹配；
5. COUNT/GROUP 的 `display_limit` 必须为 null；
6. LIST 的 order field 必须出现在物理投影中，但内部 ID 不公开展示；
7. 用户明确数量取 `min(requested_limit, 10)`，未指定时使用 10；
8. 用户明确排序完整保留；只有未指定时才能应用默认排序；
9. REQUIRED 内容条件必须要求正文或受控元数据证据；
10. 系统范围不能出现在模型可修改的 criteria 中；
11. COUNT 或 `include_total_count=true` 的资格表达式必须可完全编译为精确元数据条件；
12. 语义主题总数请求必须规范化为参考 LIST，并标记 `SEMANTIC_TOTAL_COUNT`；
13. 每个计划必须包含 `result_assets`；SUPPORTING 的目标数量必须位于 3 至 5，PRIMARY 不得超过 10；
14. COUNT/GROUP 必须生成与聚合共享同一过滤条件和快照的 Asset 明细分支；
15. 影响结果集合的歧义必须澄清，不能通过默认值猜测。

模型计划失败时允许一次带校验错误的结构化重试。第二次失败返回明确规划错误或澄清问题，
不能退回旧的单 topic 逻辑。

### 7.4 能力编译

Validator 之后由确定性 Compiler 计算能力集合：

```text
needs_metadata_filter
needs_content_discovery
needs_evidence_retrieval
needs_aggregation
needs_asset_enumeration = true
needs_answer_generation
needs_semantic_count_fallback
```

Compiler 根据能力生成 Task DAG，不由 LLM 自由输出 Skill 名称或服务 URL。

## 8. 查询执行模式

### 8.1 纯结构化问数

适用于作者、时间、产品精确值、行业、分类、分组、数量和排序，且没有正文语义条件的问题。

```text
AssetSearchPlan
  → 编译 DataQueryPlan
  ├── 主查询：LIST / COUNT / GROUP
  └── 支撑查询：同条件下 3 至 5 个 Asset（LIST 主查询本身即明细）
  → Data Query 执行
  → QUERY_RESULT
  → 确定性列表/统计回答 + Asset 明细
```

Data Query 绑定 `KBOT_V_KM_ASSET_SEARCHABLE`，自动注入 Domain。它负责精确过滤和统计；
Answer Composer 不重新解释集合，也不能改变行顺序或数量。COUNT/GROUP 的支撑明细与聚合共享
完全相同的过滤条件和快照，默认按 `asset_date DESC` 返回最多 5 个，并为统计和明细分别产生 Qn。

### 8.2 纯正文问文

适用于“解释”“总结”“比较实现方式”等不要求 Asset 列表或统计的问题。

```text
AssetSearchPlan
  → KC 多条件 Discovery
  → Bundle 候选融合
  → Evidence 检索与扩展
  → 条件支持判断
  → CITATION_PACK
  → Grounded Answer + 3 至 5 个支撑 Asset
```

如果同时包含作者、时间等硬条件，先通过 Data Query 得到资格集合，再把 Bundle Scope 传给 KC；
如果没有用户元数据条件，KC 在授权的 KM Asset Collection 和可搜索 Revision 内发现候选，随后
仍通过搜索视图批量校验候选 Asset 的当前 READY 映射。该校验是系统范围验证，不产生用户可见 Qn。
最终回答必须列出 3 至 5 个实际支撑解释的 Asset，每个 Asset 至少绑定一个来自自身 Bundle 的 Cn。
如果只有 1 至 2 个 Asset 具有充分证据，则按实际返回并明确证据范围，不能用弱相关结果补足。

### 8.3 混合搜索

适用于既要精确元数据条件，又要正文语义判断、列表或解释的问题。

```text
                ┌─ Data Query：元数据资格集合 ────────────┐
AssetSearchPlan ┤                                        ├─ 取交集/并集
                └─ KC：各语义条件多路候选与证据 ─────────┘
                                                           │
                                                           ▼
                                                   条件命中矩阵
                                                           │
                                                           ▼
                                              排序 → 截断 → 回答
```

元数据硬条件与语义硬条件通常取交集；同一语义概念允许 `TITLE OR PRODUCT OR SOLUTION OR
CONTENT` 时，各字段召回结果取并集，再统一判定概念是否满足。

### 8.4 不支持语义主题计数

“关于 OAC 的 Asset 总共有多少个”“有多少 Asset 涉及金融欺诈”中的集合边界依赖全文、向量
召回和语义证据判定，不是一个可由关系条件精确封闭的集合。本方案明确不实现这种计数，也不实现
全库逐条模型分类来模拟精确总数。

Validator 发现 COUNT 或 `include_total_count=true` 的资格表达式包含任意以下条件时，进入
`SEMANTIC_TOTAL_COUNT` 回退：

- `SEMANTIC_CONCEPT`；
- 需要正文证明的 `EXACT_PHRASE`；
- 需要正文判断的 `CONTENT_TYPE`；
- 任何不能完整编译为 Data Query 精确谓词的布尔分支。

规范化后的执行计划为：

```text
operation = LIST
include_total_count = false
display_limit = 5
result_assets = {mode: PRIMARY, target_count: 5, selection: RECENT_RELEVANT}
unsupported_requests = [SEMANTIC_TOTAL_COUNT]
reference_mode = RECENT_RELEVANT
```

执行规则：

1. 保留原问题中的全部系统条件、元数据硬条件、语义硬条件和排除条件；
2. 正常执行多路召回和证据支持判断，不对候选或最终集合执行 COUNT；
3. 只有满足全部 MUST、未命中 MUST_NOT 且有必要证据的 Asset 才能进入参考结果；
4. 合格后按用户明确偏好排序；没有其他明确排序时按 `asset_date DESC NULLS LAST` 选择较新结果；
5. 默认返回最多 5 个。用户明确要求参考数量时只接受 3 至 5；不足 3 个有证据结果时返回实际
   找到的结果，但不得据此声称总数少于 3；
6. 回答先明确“主题相关性不是可精确统计的结构化字段，因此无法提供可靠总数”；
7. 随后使用“以下是本次检索到的较新相关 Asset，供参考”，不得使用“总共有”“全部”“仅有”或
   “共命中 N 个”等暗示完整集合的文案；
8. 每个参考 Asset 必须带有支持主题相关性的 Cn，受控元数据精确命中时可同时带 Qn。

以下问题仍属于支持的精确 COUNT：

```text
产品字段等于 OAC 的 READY Asset 有多少个？
Alice 上个月发布了多少个 Asset？
按行业统计上个月发布的 Asset 数量。
```

关键区别是条件能否完整转换为 Data Query 的受控元数据谓词，而不是根据“OAC”等具体词做代码
特判。“关于 OAC”默认是语义相关性；“产品字段等于 OAC”是精确元数据条件。

## 9. 多条件语义检索

### 9.1 概念解析

每个语义条件独立解析，不合并为一条含糊查询：

```text
c1: OAC，MUST
c2: 金融欺诈，SHOULD
c3: 案例，SHOULD
c4: 安装手册，MUST_NOT
```

概念解析器优先使用版本化受控词表，将产品缩写、全称和多语言别名映射到同一个 concept ID。
未命中词表时，保留原始短语，并由查询扩展模型生成少量语义等价表达。扩展结果必须记录来源和
版本，不能覆盖原文，也不能把上下位概念当成完全等价词。

### 9.2 多语言查询扩展

每个概念形成独立查询组：

```text
concept_id: oracle.analytics.cloud
original: OAC
equivalents:
  - Oracle Analytics Cloud
  - Oracle 分析云
related_terms: []
```

`equivalents` 用于同概念 OR 召回；`related_terms` 只能用于扩大候选，不能单独证明条件命中。
多个 MUST 概念之间按布尔表达式组合。翻译或扩展失败时仍执行原始词、全文和向量检索，并记录
降级，不得删除该条件。

### 9.3 候选通道

每个语义条件至少可使用以下通道：

1. 标题精确短语及全文索引；
2. 产品、方案、行业、分类等受控元数据匹配；
3. Discovery Profile 全文检索；
4. Discovery 向量检索；
5. Evidence 全文桥接到 Bundle；
6. 必要时 Evidence 向量桥接到 Bundle。

候选预算按检索配置和查询复杂度分配。技术 Top-K、RRF 常量和阈值属于服务配置或版本化检索
策略，不写入业务 Prompt，也不与用户展示上限绑定。

### 9.4 候选融合

融合单位是 Asset/Bundle Revision，不是 Chunk。每个通道先归一到：

```text
asset_id
bundle_id
bundle_revision_id
criterion_id
channel
rank
raw_score
matched_locator
```

融合步骤：

1. 按 Bundle Revision 去重 Evidence 命中；
2. 同一 Asset 的多通道结果按 criterion 汇总；
3. 使用 RRF 等确定性方法生成 criterion 级召回分；
4. 不同 criterion 的分数保持分离，不提前压缩成一个总分；
5. Data Query 与 KC 的 Asset/Bundle 映射不一致时排除候选并记录数据质量事件；
6. 候选集合完成后才进入支持判断。

### 9.5 条件支持判断

支持判断器只接收预分配的 Asset 标签、criterion 标签和证据标签，只能输出：

```text
DIRECT_SUPPORT
PARTIAL_SUPPORT
METADATA_SUPPORT
CONTEXT_ONLY
CONTRADICTS
NO_SUPPORT
```

MUST 条件通过规则：

- 受控元数据精确命中：`METADATA_SUPPORT` 或更强；
- 正文语义条件：`DIRECT_SUPPORT`；
- 配置允许时，多个互补 `PARTIAL_SUPPORT` 可形成 DIRECT，但必须保留组合证据；
- `CONTEXT_ONLY` 和纯向量相似不能通过；
- 任一 MUST_NOT 得到 DIRECT/METADATA 支持时排除 Asset。

模型不可返回未知 Asset、未知 criterion 或未知 citation label。缺失输出视为未证明，不视为通过。

## 10. 条件命中矩阵

每个 Asset 生成以下内部结构：

```json
{
  "asset_id": "...",
  "bundle_id": "...",
  "bundle_revision_id": "...",
  "eligible": true,
  "requirements": [
    {
      "criterion_id": "c1",
      "status": "DIRECT_SUPPORT",
      "query_reference_labels": ["Q1"],
      "citation_labels": ["C1"],
      "channel_ranks": {"metadata": 1, "text": 3, "vector": 2}
    }
  ],
  "preferences": [
    {
      "preference_id": "p1",
      "matched": true,
      "status": "DIRECT_SUPPORT",
      "citation_labels": ["C2"]
    }
  ],
  "exclusion_reasons": [],
  "stable_sort_key": []
}
```

该矩阵是筛选、排序、引用验证和线上诊断的共同事实源。最终回答不公开内部 ID、原始分数、
Prompt 或隐藏推理，只展示业务字段、命中说明和授权引用。

## 11. 排序与截断

### 11.1 资格门槛

排序前先删除：

1. 不满足系统范围的 Asset；
2. 不满足 eligibility expression 的 Asset；
3. 命中 MUST_NOT 的 Asset；
4. 必需正文证据不足的 Asset；
5. Asset/Bundle/Revision 映射不一致的 Asset。

### 11.2 用户明确排序

如果用户指定“最新”“最早”“按作者”“按产品”等排序：

1. 先完成所有硬条件语义判定；
2. 软偏好按用户声明的优先级形成分桶；
3. 在同一偏好桶内执行用户明确排序；
4. 相关度和 `asset_id` 只作为稳定 tie-breaker。

例如“最新 5 个关于 OAC 的 Asset”不能用相关度覆盖日期顺序。

### 11.3 默认排序

未指定排序时使用确定性字典序：

```text
preference_1_matched DESC
preference_2_matched DESC
...
minimum_must_relevance DESC
evidence_support_strength DESC
aggregate_retrieval_rank ASC
asset_date DESC NULLS LAST
asset_id ASC
```

`minimum_must_relevance` 使用最弱 MUST 条件的相关度，避免某一主题极强掩盖另一个必需主题
完全不相关。默认排序策略必须版本化，并在 Run provenance 中记录版本。

### 11.4 展示上限

最终合格集合完成排序后：

```text
effective_limit = min(user_requested_limit or 10, 10)
```

列表阶段保留第 `effective_limit + 1` 条用于判断截断，但不把探测行交给 Composer。回答必须明确：

- 纯元数据 LIST 已完整返回时说明“全部结果已展示”；或
- 已执行精确元数据 COUNT 时说明总数及当前展示数量；或
- 未执行完整计数，只能说明“结果超过 10 条，以下展示前 10 条”。

语义 LIST 和 `SEMANTIC_TOTAL_COUNT` 回退不得使用探测行推断总数，也不得声称“全部结果已展示”。
不得在没有精确元数据 COUNT 时编造总数。

## 12. 精确聚合与支撑 Asset 一致性

只有资格表达式可完全编译为精确元数据条件时，LIST 才能设置 `include_total_count=true`。同时需要
“总数 + 前 10 条”时，Compiler 生成共享条件快照：

```text
ASSET_SEARCH_PLAN
  ├── COUNT：完整资格集合
  └── LIST：同一集合排序后前 10 条
```

两条执行分支必须共享：

- 计划哈希；
- 语义模型版本；
- 可搜索视图版本；
- KC 索引/Revision 快照；
- 条件判定策略版本；
- 时间边界及时区。

如果两条分支跨越状态更新而无法获得一致快照，系统返回一致性警告或重试，不能把不同集合的
COUNT 和 LIST 拼成一个看似精确的回答。

包含语义资格条件的 LIST 强制 `include_total_count=false`，不生成 COUNT 分支。

纯 COUNT 或 GROUP 也必须生成支撑 Asset 明细分支：

```text
ASSET_SEARCH_PLAN
  ├── Q1：COUNT / GROUP 聚合结果
  └── Q2：相同过滤集合中较新的 3 至 5 个 Asset
```

Q2 不能从另一个过滤集合、缓存版本或历史 Revision 取样。每个展示 Asset 使用 Q2；数字和分组
结论使用 Q1。如果回答进一步描述 Asset 正文，则必须再检索对应 Bundle，并为该描述增加 Cn。

## 13. 标准示例

### 13.1 纯问文：详细解释技术

问题：

```text
关于 Oracle AI Vector Search，帮我详细解释一下它在这些方案里是怎么使用的。
```

关键计划：

```json
{
  "operation": "ANSWER",
  "target": "CONTENT",
  "criteria": [
    {
      "criterion_id": "c1",
      "kind": "SEMANTIC_CONCEPT",
      "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
      "operator": "RELATED_TO",
      "values": ["Oracle AI Vector Search"],
      "occurrence": "MUST",
      "evidence_requirement": "CONTENT"
    }
  ],
  "evidence_policy": {
    "coverage": "DEPTH",
    "required_support": "DIRECT_SUPPORT",
    "minimum_distinct_bundles": 1
  },
  "result_assets": {
    "mode": "SUPPORTING",
    "target_count": 5,
    "selection": "EVIDENCE_COVERAGE_THEN_RECENT"
  }
}
```

执行 KC Discovery 和 Evidence 检索。回答先详细解释技术，再列出 3 至 5 个实际支撑回答的 Asset；
技术结论和每个 Asset 都使用来自对应 Bundle 的 Cn。

### 13.2 纯问数：作者与时间组合

问题：

```text
Alice 和 Bob 上个月发布的 READY Asset 有哪些？按发布日期从新到旧。
```

关键计划：

```json
{
  "operation": "LIST",
  "target": "ASSET",
  "criteria": [
    {
      "criterion_id": "c1",
      "kind": "METADATA",
      "field_scope": ["author"],
      "operator": "IN",
      "values": ["alice", "bob"],
      "occurrence": "MUST",
      "evidence_requirement": "QUERY_RESULT"
    },
    {
      "criterion_id": "c2",
      "kind": "METADATA",
      "field_scope": ["asset_date"],
      "operator": "BETWEEN",
      "values": ["2026-07-01", "2026-07-31"],
      "occurrence": "MUST",
      "evidence_requirement": "QUERY_RESULT"
    }
  ],
  "eligibility_expression": {
    "node_type": "ALL",
    "children": [
      {"node_type": "REF", "criterion_id": "c1"},
      {"node_type": "REF", "criterion_id": "c2"}
    ]
  },
  "order_by": [{"field": "asset_date", "direction": "DESC"}],
  "display_limit": 10,
  "result_assets": {
    "mode": "PRIMARY",
    "target_count": 10,
    "selection": "REQUESTED_ORDER"
  }
}
```

`READY` 不作为用户条件进入计划，而由搜索视图强制保证。回答使用 Q1，不调用 KC。

### 13.3 混合：主主题与多个偏好

问题：

```text
帮我搜一下关于 OAC 的 Asset，最好是金融欺诈案例，优先最近一年发布的。
```

关键计划：

```json
{
  "operation": "LIST",
  "target": "ASSET",
  "criteria": [
    {
      "criterion_id": "c1",
      "kind": "SEMANTIC_CONCEPT",
      "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
      "operator": "RELATED_TO",
      "values": ["OAC"],
      "occurrence": "MUST",
      "evidence_requirement": "METADATA_OR_CONTENT"
    }
  ],
  "eligibility_expression": {"node_type": "REF", "criterion_id": "c1"},
  "preferences": [
    {
      "preference_id": "p1",
      "criterion": {
        "kind": "SEMANTIC_CONCEPT",
        "field_scope": ["TITLE", "SOLUTION", "CONTENT"],
        "operator": "RELATED_TO",
        "values": ["金融欺诈"]
      },
      "priority": 1,
      "evidence_requirement": "METADATA_OR_CONTENT"
    },
    {
      "preference_id": "p2",
      "criterion": {
        "kind": "CONTENT_TYPE",
        "field_scope": ["category", "CONTENT"],
        "operator": "EQ_OR_RELATED",
        "values": ["案例"]
      },
      "priority": 2,
      "evidence_requirement": "METADATA_OR_CONTENT"
    },
    {
      "preference_id": "p3",
      "criterion": {
        "kind": "METADATA",
        "field_scope": ["asset_date"],
        "operator": "GTE",
        "values": ["2025-08-21"]
      },
      "priority": 3,
      "evidence_requirement": "QUERY_RESULT"
    }
  ],
  "display_limit": 10,
  "result_assets": {
    "mode": "PRIMARY",
    "target_count": 10,
    "selection": "PREFERENCES_THEN_RELEVANCE"
  }
}
```

“最近一年”因“优先”而属于软偏好，不是硬过滤。所有满足 OAC 的 Asset 均有资格；金融欺诈、
案例和最近一年依次决定偏好桶。最终结果为每个语义命中提供 Cn 或受控元数据 Qn。

### 13.4 不支持的主题总数请求

问题：

```text
关于 OAC 的 Asset 总共有多少个？
```

规范化后的关键计划：

```json
{
  "operation": "LIST",
  "target": "ASSET",
  "answer_detail": "BRIEF",
  "criteria": [
    {
      "criterion_id": "c1",
      "kind": "SEMANTIC_CONCEPT",
      "field_scope": ["TITLE", "PRODUCT", "SOLUTION", "CONTENT"],
      "operator": "RELATED_TO",
      "values": ["OAC"],
      "occurrence": "MUST",
      "evidence_requirement": "METADATA_OR_CONTENT"
    }
  ],
  "eligibility_expression": {"node_type": "REF", "criterion_id": "c1"},
  "include_total_count": false,
  "display_limit": 5,
  "result_assets": {
    "mode": "PRIMARY",
    "target_count": 5,
    "selection": "RECENT_RELEVANT"
  },
  "unsupported_requests": ["SEMANTIC_TOTAL_COUNT"],
  "order_by": [{"field": "asset_date", "direction": "DESC"}]
}
```

回答结构固定为：

```text
“关于 OAC”属于语义相关性条件，不是可精确统计的结构化字段，因此我无法提供可靠的 Asset
总数。以下是本次检索到的较新相关 Asset，供参考：

1. ... [C1]
2. ... [C2]
3. ... [C3]
4. ... [C4]
5. ... [C5]
```

实际只有少于 5 个结果得到充分证据时，只返回这些结果，不补足、不放宽主题，也不使用“仅有”
等可能暗示完整总数的措辞。

## 14. 引用和回答组合

### 14.1 引用职责

| 结论 | 允许证据 |
|---|---|
| 作者、日期、产品、行业、分类、数量、排序 | Qn |
| 正文描述、技术细节、案例事实、业务问题 | Cn |
| 受控产品/分类元数据足以证明的概念 | Qn；需要解释时补 Cn |
| 语义偏好命中 | Cn，或明确的受控元数据 Qn |

每个最终 Asset 的最低引用要求：

| 搜索方式 | 每个 Asset 必须绑定 |
|---|---|
| 纯问文 ANSWER | 至少一个来自该 Asset Bundle 的 Cn |
| 纯元数据 LIST | 包含该行的 Qn |
| 精确 COUNT/GROUP | 支撑明细 QueryResult 的 Qn；聚合数字另有聚合 Qn |
| 元数据 + 正文混合 | 证明元数据资格的 Qn，以及证明语义条件的同 Bundle Cn |
| 语义总数回退 | 证明主题相关性的同 Bundle Cn，受控元数据命中时可补 Qn |

一个 Qn 可以证明同一 QueryResult 中的多行，但必须在每个 Asset 条目后明确显示引用标记；Cn 不得
跨 Bundle 复用。

### 14.2 最终结果约束

Composer 必须验证：

1. 最终 Asset 集合和顺序与 `ASSET_MATCH_RESULT` 一致；
2. 每个展示 Asset 的标题完整出现一次；
3. 每个 REQUIRED 语义条件都有同 Bundle 的有效 Cn，或合同允许的 Qn；
4. 声称命中的软偏好必须有证据；
5. 不得引用其他 Asset 的 Bundle；
6. 不得泄露 `asset_id`、`bundle_id`、`bundle_revision_id` 或原始模型 JSON；
7. 截断和总数文案与执行事实一致；
8. `used_citation_labels`、回答正文标签和 Reference Cards 完全一致。
9. 存在 `SEMANTIC_TOTAL_COUNT` 时，必须包含“不支持可靠语义总数”的系统说明，并禁止任何总数式
   表述；参考结果必须为 3 至 5 个，证据不足时允许少于 3 个。
10. ANSWER、COUNT、GROUP 和 COMPARE 必须包含 3 至 5 个支撑 Asset；不足时按实际返回，零个时
    不能产出无来源的成功答案。
11. 每个展示 Asset 的 Qn/Cn 必须满足上表的最低要求，引用缺失或 Bundle 映射错误时整个回答不通过。

模型输出连续两次验证失败时，使用确定性 Markdown 回退。回退按已冻结排序输出全部展示行，
并按 Bundle 映射 Qn/Cn；不能因为某条引用生成失败而悄悄删除 Asset。

### 14.3 纯问文回答约束

每个实质性技术结论必须由至少一个 Cn 支持。多个 Asset 出现相互冲突的内容时，回答应并列
陈述来源差异，不能由模型选择一个无说明的“正确版本”。回答末尾必须列出 3 至 5 个实际支撑
结论的 Asset 及其 Cn；证据覆盖不足时明确列出无法回答的方面。

## 15. 内部 API 与合同变化

本方案优先复用现有内部 API。实现阶段需要新增或扩展以下能力：

### 15.1 Platform Core

在 `packages/platform_core` 增加 Asset Search 版本化合同：

- `AssetSearchPlanV1`；
- `AssetSearchCriterion`；
- `AssetBooleanExpression`；
- `AssetPreference`；
- `AssetCandidateSetV1`；
- `AssetMatchResultV1`。

合同必须 `extra="forbid"`、冻结、限制递归深度和集合大小，并保持稳定英文协议值。

### 15.2 Data Query

继续使用 `DataQueryPlan.v1`，由 Asset Search Compiler 生成。需要支持：

1. 搜索专用受管数据集绑定 `KBOT_V_KM_ASSET_SEARCHABLE`；
2. 多条件结构化过滤；
3. 为候选校验按 Asset ID 批量查询；
4. COUNT 和 LIST 共用版本与计划追踪信息；
5. 返回物理排序字段但在公开 QueryResult 中隐藏内部字段。

不把递归语义布尔树直接塞进 DataQueryPlan；Compiler 只把其中可由结构化字段精确执行的子树
编译到 Data Query。无法保持语义等价的嵌套表达式由 Coordinator 在候选矩阵上执行。

### 15.3 Knowledge Core

扩展现有 Retrieval Query Plan，使一次请求可以携带多个带 label 的语义条件：

```text
criterion_id
semantic_query
exact_phrases
resolved_concept
field_scope
occurrence
candidate_budget_class
```

KC 返回 criterion 级 Bundle 候选与 Evidence Group，不直接决定最终 Asset 是否合格。内部 API
继续要求服务凭据和 audience-bound AuthContext JWT，不接受浏览器 API Key。

### 15.4 Agent Runtime

用统一 `asset_search` Specialist 收敛现有 KM Asset 特殊路由、topic 扩展和 hybrid scope 逻辑。
Root Planner 只需要判断当前 Agent 是否启用 Asset Search 能力；具体的问文、问数和混合执行图由
Asset Search Compiler 确定。

## 16. 安全与权限

1. Domain 从可信 Header/AuthContext 注入，用户文本中的 Domain 名称不能改变范围；
2. Public Main API 验证 Portal API Key，内部调用使用服务凭据和 audience-bound JWT；
3. 不向下游转发 Portal API Key，不缓存长生命周期内部 JWT；
4. Candidate Fusion 只接受本 Run 冻结范围内的 Asset/Bundle；
5. Data Query 的物理 schema、表、列和 SQL 不进入 LLM 上下文；
6. KC Citation Preview 继续执行 Bundle/Collection 授权；
7. 日志不记录正文全文、Token、数据库密码、原始 AuthContext 或敏感作者信息；
8. Qn 公开展示隐藏内部 ID，Cn 只展示现有允许的来源定位信息。

## 17. 降级和错误语义

| 故障 | 处理原则 |
|---|---|
| Data Query 连接失败 | 所有 Asset 搜索无法验证当前 READY 映射时失败；不得跳过系统或用户硬条件 |
| KC 全文失败、向量可用 | 保留向量候选，标记通道降级并加强 Evidence 支持验证 |
| KC 向量失败、全文可用 | 保留全文候选；语义覆盖不足时返回部分证据或无结果 |
| 全文和向量均失败 | 带正文条件的查询失败 |
| 查询扩展模型失败 | 使用原始概念和受控词表，不删除条件 |
| 条件支持模型失败 | 使用确定性支持规则；不能证明的 MUST 不通过 |
| Asset/Bundle 映射缺失 | 排除 Asset，记录数据质量错误，不生成无 Cn 的语义命中 |
| 用户请求语义主题总数 | 不执行 COUNT；返回限制说明和最多 5 个较新相关 Asset |
| COUNT/GROUP 聚合成功但支撑明细失败 | 不发布最终成功回答；重试共享条件的明细分支 |
| 支撑 Asset 引用缺失或跨 Bundle | Composer 验证失败，不删除条目后伪装成功 |
| Composer 校验失败 | 使用确定性 Markdown 回退 |
| COUNT/LIST 快照不一致 | 重试或披露不一致，不能合并为一个精确回答 |

稳定错误码建议：

```text
ASSET_SEARCH_PLAN_INVALID
ASSET_SEARCH_AMBIGUOUS
ASSET_SEARCH_SCOPE_UNAVAILABLE
ASSET_SEARCH_DATA_SOURCE_FAILED
ASSET_SEARCH_CONTENT_CHANNELS_FAILED
ASSET_SEARCH_MAPPING_INCONSISTENT
ASSET_SEARCH_EVIDENCE_INSUFFICIENT
ASSET_SEARCH_SNAPSHOT_MISMATCH
```

## 18. 可观测性

### 18.1 Run provenance

每次搜索至少记录：

- `asset_search_plan_id` 和 plan hash；
- Prompt、模型、词表、语义模型及检索策略版本；
- 解析后的时间范围和时区；
- criterion、MUST、MUST_NOT、preference 数量；
- 各通道候选数、去重后 Asset 数、硬条件淘汰数；
- 各偏好命中数；
- 最终数量、展示数量和是否截断；
- `result_assets` 目标数量、实际数量和选择规则；
- Qn/Cn 数量及引用覆盖率；
- 降级通道、重试和错误码；
- Data Query Run ID、KC retrieval request ID、Agent Run/Trace ID。

### 18.2 运行事件

建议增加可公开但不暴露隐藏推理的事件：

```text
asset_search.plan.completed
asset_search.metadata.completed
asset_search.discovery.completed
asset_search.fusion.completed
asset_search.requirements.completed
asset_search.count.completed
asset_search.completed
```

事件只描述事实，例如“合并 36 个候选、18 个通过硬条件、展示 10 个”，不输出模型 chain of
thought。开发环境日志继续通过统一 operations logs 页面观察，正式业务追踪通过 Run、Task、Artifact
和事件 API 完成。

### 18.3 指标

```text
asset_search_runs_total{operation,status}
asset_search_candidates_total{channel,criterion_kind}
asset_search_filtered_total{reason}
asset_search_channel_failures_total{channel}
asset_search_citation_coverage_ratio
asset_search_plan_validation_failures_total{reason}
asset_search_duration_seconds{stage,operation}
asset_search_unsupported_requests_total{request_type}
asset_search_snapshot_mismatch_total
```

指标标签禁止使用原始问题、作者、Asset ID 或任意高基数字段。

## 19. 性能与预算

### 19.1 预算来源

候选数量、并发、超时、语义判定批次和模型输出长度必须来自：

- Agent Execution Spec；
- Data Query Policy Binding；
- KC 检索策略配置；
- Model Serving 功能模型配置。

除了产品明确规定的“最终最多展示 10 个 Asset”，不得在业务代码中硬编码技术 Top-K、并发数或
模型 `max_tokens`。

### 19.2 自适应执行

1. 纯元数据查询不调用 Embedding 或正文模型；
2. 高选择性元数据硬条件先缩小 KC 范围；
3. 没有元数据条件的纯语义查询由 KC 先发现，再由 Data Query 批量校验当前 Asset；
4. Evidence 检索只对进入支持判断的 Bundle 执行；
5. 软偏好在候选较多时先用 Profile 判断，必要时才下钻 Evidence；
6. 语义总数回退只检索生成 3 至 5 个参考结果所需的候选，不启动全库分类或计数；
7. 相同 plan hash、权限快照和索引版本可短期复用内部判定，但权限和 READY 状态必须重新验证。

## 20. 测试设计

### 20.1 合同测试

- 递归 ALL/ANY/NOT、非法引用、循环和深度限制；
- metadata、semantic、preference 与 evidence policy 的合法组合；
- COUNT、GROUP、LIST、ANSWER 的字段约束；
- 所有 Operation 必须生成合法的 `result_assets`；
- 额外字段、物理字段、未知 operator 严格拒绝；
- 用户数量、排序、时区和日期边界保真。

### 20.2 Planner 评估集

评估集必须使用组合问题，不以单 topic 为主体：

1. 多作者 OR + 日期 AND；
2. 产品 OR 正文主题 + 行业 AND；
3. 两个 MUST 主题 + 两个 SHOULD 偏好；
4. MUST_NOT 排除内容类型；
5. 明确排序、明确数量和分组；
6. 中文问题检索英文 Asset，英文问题检索中文正文；
7. 多轮增加、替换、删除条件；
8. “最好”与“必须”的最小对照；
9. 相对日期和模糊日期；
10. 精确 metadata count 与语义主题总数回退的最小对照。

Planner 验收重点是字段、条件强度、布尔结构、日期、数量和排序，不只检查最终 route。

### 20.3 检索质量测试

- 每个 MUST 条件的候选召回率；
- 多 MUST 条件下最弱条件召回率；
- 软偏好排序的 NDCG/Pairwise Accuracy；
- 全文单通道、向量单通道和组合通道对比；
- 多语言概念等价召回；
- 标题强匹配但正文不支持的误召回；
- 向量高分但无直接证据的误判；
- 同 Asset 多 Revision 和多 Chunk 去重。

### 20.4 准确性不变量

自动化测试必须达到：

1. 系统范围、Domain、READY 和权限违规数为 0；
2. 元数据硬条件违规数为 0；
3. 用户明确数量和排序保真率为 100%；
4. 语义 MUST 的最终 Asset 证据覆盖率为 100%；
5. MUST_NOT 命中 Asset 返回数为 0；
6. 精确元数据 COUNT 与数据库真值完全一致；
7. 语义主题总数请求执行 COUNT 的次数为 0，且参考结果数量和提示文案符合合同；
8. 最终正文引用、`used_citation_labels` 和 Reference Cards 完全一致；
9. FAILED、缺 Bundle、非当前 Revision Asset 返回数为 0；
10. 无完整 COUNT 时总数虚构数为 0；
11. 所有成功回答的 Asset 明细存在率为 100%；
12. 每个 Asset 的最低 Qn/Cn 引用覆盖率为 100%，跨 Bundle 引用数为 0；
13. COUNT/GROUP 的支撑 Asset 与聚合过滤集合一致率为 100%。

检索 Precision/Recall、NDCG 和时延目标在建立真实标注基线后冻结到版本化评估配置，不能凭主观
示例宣布达标。

### 20.5 集成与故障测试

- Data Query + KC + Runtime 真实依赖链路；
- Data Query 超时、KC 全文失败、KC 向量失败、模型失败；
- READY 状态在搜索运行中的变更；
- Bundle Revision 切换和映射不一致；
- 精确 COUNT/LIST 快照竞争；
- COUNT/GROUP 聚合成功但支撑 Asset 明细失败；
- 语义主题总数请求不得产生 Data Query COUNT 或全库语义分类任务；
- SSE 断线重放不重复执行检索；
- Composer 两次失败后的确定性回退；
- 多 Domain、无 Agent Grant、无 Collection 权限；
- Oracle RAW(16) Bundle ID 的传输和反序列化。

## 21. 迁移方案

### 阶段 1：合同和基线

1. 冻结当前真实问题集、失败样本和人工标注结果；
2. 增加 `AssetSearchPlan.v1` 及合同测试；
3. 实现 Planner/Validator，但不改变线上执行；
4. 对同一问题影子生成旧路由与新计划，比较条件、数量和排序。

退出条件：系统范围、硬/软条件、日期、数量和排序的计划准确率达到评估门槛。

### 阶段 2：可搜索边界

1. 建立 `KBOT_V_KM_ASSET_SEARCHABLE`；
2. 将受管语义模型绑定搜索视图；
3. 校验 KC 当前 Revision 和 Bundle 可检索状态；
4. 增加 Data Query/KC 交叉映射检查和数据质量告警。

退出条件：所有测试与真实数据中不可搜索 Asset 返回数为 0。

### 阶段 3：统一候选与条件矩阵

1. 实现多 criterion KC 查询；
2. 实现 Asset Candidate Fusion；
3. 实现 `ASSET_MATCH_RESULT.v1`；
4. 先切换 LIST，不切换 COUNT 和 ANSWER；
5. 影子比较旧方案与新方案的 Recall、Precision、排序和引用覆盖。

退出条件：新方案在标注集上达到目标，且无硬条件或引用不变量违规。

### 阶段 4：纯问文与混合回答

1. 切换 ANSWER 和带元数据范围的正文检索；
2. Composer 改为消费条件矩阵；
3. 为 ANSWER、COUNT、GROUP 和 COMPARE 增加 3 至 5 个支撑 Asset 输出；
4. 验证 Qn/Cn、回退和多 Bundle 引用；
5. 实现 `SEMANTIC_TOTAL_COUNT` 限制说明和 3 至 5 个较新参考结果；
6. 删除旧的单 topic 扩展和 Data Query First 特殊分支。

退出条件：问文、问数、混合路径均由统一计划生成，旧路径不再被调用。

### 阶段 5：清理与文档切换

1. 删除旧 route enum 中仅服务单 topic 的分支、Prompt 和测试；
2. 删除不再使用的 topic 翻译合同和合并逻辑；
3. 更新 `docs/architecture/`、产品文档、OpenAPI 和运行手册；
4. 将本文状态改为已实施后，把稳定内容合并进当前架构文档并删除 proposal。

迁移期间不做双写。影子规划和影子检索只读，不影响线上回答；正式切换按能力分阶段完成，每个
阶段只有一个权威执行结果。

## 22. 实现影响范围

预期修改范围如下，精确文件以实施前代码审计为准：

```text
packages/platform_core/src/platform_core/contracts/
services/agent_runtime/src/agent_runtime/specialists/
services/agent_runtime/src/agent_runtime/application/
services/data_query/src/data_query/application/managed_datasets.py
services/knowledge_core/src/knowledge_core/application/
services/knowledge_core/src/knowledge_core/api/
services/km_asset_app/src/
database/oracle/
tests/unit/
tests/contract/
tests/integration/
tests/evaluation/
docs/architecture/
docs/openapi/
```

实施时遵守服务所有权：API 只做协议转换，规划和编排位于 application/specialist，数据库访问位于
repository，事务由 Unit of Work 管理。不得让 Agent Runtime 或 API Adapter 直接持有数据库 Session。

## 23. 完成定义

新搜索方案只有同时满足以下条件才算完成：

1. 三类标准问题和组合评估集全部由统一计划执行；
2. 一个问题可表达多个 MUST、MUST_NOT、ANY 分支和有序 SHOULD；
3. 纯问数不调用 KC，纯问文不依赖 topic 字符串预截断，混合搜索不在融合前截取 10 条；
4. 系统可搜索边界在 Data Query、KC 和 Runtime 三层一致；
5. 精确元数据 COUNT 正常执行，语义主题总数请求明确拒绝计数并返回 3 至 5 个较新参考结果；
6. 用户数量、排序、时区和布尔条件在计划、执行和回答间保真；
7. 每个语义硬条件都有同 Asset/Bundle 的有效证据；
8. 所有成功回答均包含主结果或支撑 Asset；ANSWER、COUNT、GROUP 和 COMPARE 默认包含 3 至 5 个；
9. 每个 Asset 均具有符合搜索方式的正确 Qn/Cn，且不存在跨 Bundle 引用；
10. Qn/Cn、回答正文和 Reference Cards 一致；
11. 旧单 topic 特殊路径已删除，没有兼容或双执行分支；
12. 合同、单元、集成、故障、权限和质量评估均通过；
13. 当前架构文档、OpenAPI、数据库脚本和部署说明同步更新；
14. 生产等价环境验证 Run、Task、Artifact、引用、日志和指标完整。
