# Legacy File Processor

此目录保留 3.4/V1 `File → TxtChunk → embedding` 链路的实现，包含旧的
`services/file_processor.py`、旧 Docling 分块器和旧 ParserService。它不属于
Knowledge Core V2，不得被 KC API、KC Parser Worker 或 V2 Skill 新增引用。

KC Parser 已迁移到 `knowledge_core/workers/parser/`，PROFILE/INDEX/PURGE
迁移到 `knowledge_core/workers/projection/`。目录中的 `kc_*.py` 和
`visual_enricher.py` 仅为开发期导入 shim，待 V1 退役时一并删除。
