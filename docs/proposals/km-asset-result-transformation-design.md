# KM Asset 复合筛选与结果转换设计

> 状态：讨论方案，尚未实现。
>
> 本文扩展 `AssetSearchPlan.v1`，解决“筛选 Asset 集合后，再对指定展示字段执行翻译”等复合请求。

## 1. 典型请求

```text
1. 把 Solution briefing 转为中文。
2. 只需要过去一年发布的 Asset。
```

该请求同时包含两种不同语义：

1. “过去一年发布”是决定 Asset 资格集合的精确元数据条件；
2. “Solution briefing 转为中文”是对最终结果的展示转换，不参与候选召回和资格判断。

系统不得把“翻译成中文”识别为主题条件，也不得先翻译全量候选后再过滤。

## 2. 设计目标

1. 精确区分发布日期、更新时间和通用 Asset 日期；
2. 先执行授权与元数据过滤，再读取最终 Asset 的 manifest 正文；
3. 只在用户明确要求时调用一次批量翻译；
4. 翻译不得改变 Asset 集合、排序、标题、日期或引用关系；
5. 每个展示的 Asset 必须继续绑定自身 Bundle 的 C 引用；
6. 多轮追问必须基于上一轮冻结搜索计划增量修改，不能从 Assistant 回答反向猜测条件。

## 3. 日期字段语义

受管语义模型应暴露三个不同日期：

| 逻辑字段 | 来源 | 用途 |
|---|---|---|
| `publish_date` | 仅 `publish_time` | “发布于、过去一年发布、某日前发布” |
| `last_update_date` | 仅 `last_update_time` | “最近更新、某日后更新” |
| `asset_date` | `publish_date`，为空时回退 `last_update_date` | 未明确日期类型时的通用排序 |

以 `Asia/Shanghai` 的当前日期 `2026-08-25` 为例，“过去一年发布”规范化为：

```json
{
  "criterion_id": "c1",
  "kind": "METADATA",
  "field_scope": ["publish_date"],
  "operator": "GTE",
  "values": ["2025-08-25"],
  "occurrence": "MUST",
  "evidence_requirement": "QUERY_RESULT"
}
```

“过去一年”表示滚动十二个月；“去年”表示上一自然年。只有会改变结果集合且无法从用户表达确定时才澄清。

## 4. 搜索合同扩展

在 `AssetSearchPlan.v1` 增加可选的 `presentation`，不把展示转换放入 `criteria`：

```json
{
  "presentation": {
    "field_transforms": [
      {
        "field": "solution_briefing",
        "operation": "TRANSLATE",
        "target_language": "zh-CN"
      }
    ]
  }
}
```

首期只开放 `solution_briefing + TRANSLATE`。目标语言必须来自用户明确要求或可信回答语言，模型不能生成任意字段名或执行动作。

## 5. 执行流程

```text
用户问题与上一轮冻结计划
  → Context Rewrite：识别新问题或计划增量修改
  → Asset Search Planner：生成日期条件和 presentation
  → Plan Validator：校验字段、日期、语言和转换白名单
  → Data Query：按 publish_date 过滤、排序并限制到最多 10 个 Asset
  → Knowledge Core：只读取入选 Bundle 的 manifest/C 证据
  → Manifest Projector：按 bundle_id 提取原始 solution_briefing
  → Composer LLM：一次批量生成结构化翻译
  → Deterministic Composer：按冻结顺序组装标题、日期、翻译和对应 C 引用
```

翻译模型输出必须是按 `bundle_id` 映射的结构化结果。运行时必须拒绝额外 Bundle、重复 Bundle、缺失翻译和空翻译；原文缺失时显示“未提供 Solution briefing”，不得补写内容。

## 6. 引用规则

1. C 引用指向包含原始 Solution briefing 的 Asset manifest；
2. 中文 briefing 是该原文的派生翻译，应标注为“Solution briefing（中文翻译）”；
3. 每个 Asset 只能使用自身 Bundle 的 C 引用；
4. 翻译失败不得影响已经冻结的 Asset 资格集合，可返回原文并说明该字段未完成翻译；
5. 不为翻译文本创建伪造的新 Evidence 或 Citation。

## 7. 多轮计划增量

当该请求是上一轮 Asset 清单的追问时，系统读取上一轮不可变的 `AssetSearchPlan` 和结果序号映射：

- 保留上一轮作者、主题、产品、排序等未被修改的条件；
- 新增 `publish_date >= 边界日期`；
- 新增 briefing 翻译展示要求；
- 不从 Assistant 自然语言答案重新提取 Asset 条件；
- 序号必须绑定上一轮展示结果的 `asset_id + bundle_id + bundle_revision_id`。

## 8. 改造范围与部署影响

预计涉及：

1. `AssetSearchPlan.v1` 合同、Planner Prompt 与规范化；
2. KM 日期列、查询视图、同步归一化和现有数据修复；
3. Data Query 受管语义模型及模型重新协调；
4. manifest briefing 投影、批量翻译合同和确定性 Composer；
5. 单轮复合请求、多轮计划增量、日期边界、翻译映射和逐 Asset 引用测试。

不需要修改 UI，不需要重新解析附件或重建 KC 向量。部署时需要应用 Oracle Schema/数据修复、重新协调 KM 受管语义模型，并重启相关 KM Asset、Data Query 和 Agent Runtime 服务。

## 9. 不采用的简化方案

不采用“只改 Prompt，让 Composer 自由翻译”的方案。该方案无法严格区分发布日期和更新时间，也无法保证翻译结果与 Bundle 一一对应，容易丢失 Asset、改变排序或错配 C 引用。
