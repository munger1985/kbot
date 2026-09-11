# 01 · Shell、导航、主题与页面可见性

## 设计目标

测试页面使用与 `ui/km`、`tools/dev_console` 相同的原生 HTML、CSS、JavaScript 方式：每个页面是可直接打开的静态入口，公共 Shell 负责认证上下文、导航、Dialog、Toast、空态和错误展示。它不是 SPA，也不引入前端框架。

建议目录为 `ui/assistant/`：

```text
ui/assistant/
  dashboard.html              js/assistant-dashboard.js
  knowledge.html              js/assistant-knowledge.js
  x-search.html               js/assistant-x-search.js
  image-generation.html       js/assistant-image-generation.js
  domains.html                js/assistant-domains.js
  knowledge-cores.html        js/assistant-knowledge-cores.js
  data-models.html            js/assistant-data-models.js
  agents.html                 js/assistant-agents.js
  model-bindings.html         js/assistant-model-bindings.js
  media-assets.html           js/assistant-media-assets.js
  usage-runs.html             js/assistant-usage-runs.js
  css/assistant.css
  js/assistant-shell.js
```

## 公共 Shell

沿用 KM 已验证的 226px 固定左栏、68px 顶部上下文条和 12 列主体网格。Shell 只从 `GET /api/v1/apps/assistant/access` 获取当前可信上下文；页面不能从 URL、localStorage 或输入框相信 `user_id`、`domain_id`、权限或模型身份。

```text
┌──────── 226px 侧栏 ───────┬────────── 顶栏：智能工作台 / Domain / 用户 ──────────┐
│ AS  Intelligent Desk       │                                                    │
│                            ├──────────────── 页面标题与上下文操作 ──────────────┤
│ 工作台                     │                                                    │
│ 知识问答                   │                 12 列内容网格                     │
│ X 实时搜索                 │                                                    │
│ 文生图                     │                                                    │
│ ─────────                  │                                                    │
│ 资源配置                   │                                                    │
│   Domains                  │                                                    │
│   Knowledge Cores          │                                                    │
│   问数模型                  │                                                    │
│   Agents                   │                                                    │
│ ─────────                  │                                                    │
│ 管理                        │                                                    │
│   模型绑定                  │                                                    │
│   图片资产                  │                                                    │
│   用量与运行记录            │                                                    │
└────────────────────────────┴────────────────────────────────────────────────────┘
```

侧栏按权限删除不可访问项目，不能仅置灰。当前入口使用 `aria-current="page"`；顶栏显示当前业务 Token 固定的 Domain，并提供“切换 Domain”入口（仅列出当前 App 授权范围）。

## 视觉语言

页面不使用渐变、装饰性图标墙或模拟 AI 思维链。采用与 KM 一致的纸张背景、清晰边界、衬线标题和高密度但可扫描的信息布局；本 App 使用独立深青主题。

```css
:root {
  --assistant-bg: #edf1ee;
  --assistant-paper: #fcfdfb;
  --assistant-ink: #1e2b28;
  --assistant-muted: #66736e;
  --assistant-line: #d2dad5;
  --assistant-sidebar: #20312d;
  --assistant-sidebar-muted: #afbbb5;
  --assistant-accent: #176b52;
  --assistant-accent-dark: #105440;
  --assistant-good: #256348;
  --assistant-warn: #9a6317;
  --assistant-bad: #a12c2c;
}
```

颜色是 App Shell 变量，业务页面不得自行再造一套主题。标题延续 `Georgia, "Songti SC", serif`；正文沿用中文无衬线字体。卡片半径不大于 6px，主要依靠分隔线、留白和排版建立层级。

## 页面可见性

| 导航项 | 最低权限 | 无资源时的表现 |
| --- | --- | --- |
| 工作台 | `assistant:access` | 展示三个入口和待配置项，不编造指标 |
| 知识问答 | `assistant:knowledge_chat` | 提示创建或获授可用 Agent |
| X 实时搜索 | `assistant:x_search` | 提示管理员未绑定可用模型 |
| 文生图 | `assistant:image_generate` | 提示管理员未绑定可用模型 |
| Domains | `assistant:domain_manage` | 可创建或管理当前 App 的 Domain |
| Knowledge Cores | `assistant:knowledge_core_manage` | 指引先选择/创建 Domain |
| 问数模型 | `assistant:data_model_manage` | 指引先创建数据源与 Schema Snapshot |
| Agents | `assistant:agent_manage` | 指引先创建 KC；未绑定 KC 的草稿不能启用 |
| 模型绑定 | `assistant:model_binding_manage` | 可见能力矩阵及验收状态 |
| 图片资产 | `assistant:media_read` | 展示可访问的个人/项目资产 |
| 用量与运行记录 | `assistant:run_read` | 显示真实 Run，无数据时解释为空的原因 |

## 工作台首页

首页不是数据大屏。首屏展示三个等权入口卡片、当前 Domain 和“开始前需要完成”的真实配置状态；下方才显示最近运行和需要处理的失败项。入口卡只陈述能力及配置状态，不能展示虚构的节省时间、命中率或用量数字。

当当前 Domain 没有 KC、问数模型或 Agent 时，知识问答卡给出按序操作：创建 KC → 发布问数模型（可选）→ 创建并启用 Agent。当 X 搜索或文生图模型未通过能力验收时，入口显示“管理员尚未启用”，不会发出请求。
