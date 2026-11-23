# 评审原型视觉来源

本项目是已有企业 Web App 的界面设计，不是对外品牌宣传。产品身份、已存在路由及交互从两仓的生产源码和整体设计验证，不主张线上版本、模型能力或发布日期。待选稿只使用评审夹具，不引用生产资料。

- Ammolite 的官方宿主标识为 `apps/portal-web/src/app/BusinessAppShell.vue` 中现有的 **A / Ammolite** 字标；KBot 宿主为 `ui/knowledge_retrieval/js/knowledge-shell.js` 中的 **K / KBot** 字标。原型使用这段现有文字标识，支持切换宿主壳以对照；不绘制新商标，不把两套身份并列为一个生产页面。
- 颜色源为 Portal `src/shared/ui/styles.css` 的 `.business-app-knowledge_retrieval`：accent `#087f74`、strong `#08635c`、soft `#e1f3f0`。共享层提供中性色、语义徽标、边框与排版；运维蓝 `#465f9e` 不是本页主题。常规正文至少 14px，必要注释至少 12px。
- 控件参考：Portal `AIOpsKnowledgeBasesView.vue`、KBot `ui/aiops/knowledge-bases.html`、`reference-knowledge-bases.css`；借用目录密度、搜索栏、右抽屉和确认手势，不复制运维资料语义或模型字段。
- 工具页无需摄影或装饰图。证据预览采用明确标注的夹具文件内容，语义图片功能不作可用性演示，无外链资源、模型请求或真实账号。

正式方向见[确认记录](direction-approved.md)。Agent 问答另外以 Ammolite `AIOpsDiagnosisView.vue` 和 KBot4 `ui/aiops/chat.html`、`ui/aiops/css/reference-chat.css` 为准，继承工作台布局、交互与控件规格；主题变量仍由知识检索 App 壳层提供。
