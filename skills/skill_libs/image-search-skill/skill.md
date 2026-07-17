---
name: image-search-skill
description: 根据用户上传的截图或图片，在知识库中搜索视觉相似的文档页面和图片，同时反向检索关联的文本内容。用于图文并茂回答、设备面板识别、图表检索等场景。输入图片 base64 和可选文字描述，返回配对的页面截图和文本片段。
category: knowledge_retrieval
usage_example: 用户上传一张设备控制面板截图，搜索知识库中相关技术手册页面的图文内容。
---

# 输入参数约束
* image_base64 (string, 必填): base64 编码的图片数据（JPEG/PNG）
* query (string, 选填): 辅助文本查询，用于补充文本搜索
* kb_ids (array[string], 选填): 限定搜索的知识库 ID 列表，为空则搜索全部

# 输出说明
返回 list[VisualTextPair]，每个元素包含：
- file_id: 文件 ID
- page_no: 页码
- page_image_path: 页面截图路径
- image_description: 图片描述
- text_snippets: 该页关联的文本片段列表
- similarity: 相似度分数
- source: 来源标识 (visual/text/both)
