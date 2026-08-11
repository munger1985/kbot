# APEX 静态资源

本目录保存 KBot APEX 应用使用的版本化静态资源源码。修改后应在 APEX
Shared Components 的 Static Application Files 中上传同名文件，并同步更新页面引用的
版本查询参数，禁止只在某一台服务器上保留未提交的静态文件修改。

当前文件：

- `js/agent-dialog.js`：Agent 新建与编辑弹窗逻辑。
- `js/document-preview.js`：按 Run 引用授权加载源文档，并将 PDF 定位到引用页。

引用卡不要携带 Collection、Bundle 或文档主键，只输出服务端结果已有的
`run_id` 与 `citation_label`：

```html
<button type="button"
        data-kbot-run-id="..."
        data-kbot-citation-label="C1">打开来源文档</button>
```

页面初始化时绑定引用卡容器：

```javascript
KBotDocumentPreview.bind("#chat-references");
```

模块先调用引用预览描述接口，再读取受保护的 `content_url`。如果当前 APEX 的
`KBotApi` 认证只由 JavaScript 注入 `Authorization`，应通过
`KBotDocumentPreview.configure({ loadContent })` 提供与 `KBotApi` 相同认证上下文的
Blob 加载函数；不得把 Portal API Key 写入静态文件、URL 或 DOM。PDF 打开后使用
服务端返回的 `page_no` 设置 `#page=<页码>`，不再展示 Chunk 文本。
