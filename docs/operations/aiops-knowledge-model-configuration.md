# 运维知识模型配置与上传错误

知识库必须保存完整的文档解析、检索和向量模型配置后才可上传。缺少配置时返回
`AIOPS_KNOWLEDGE_COLLECTION_MODELS_REQUIRED`，在写入源文件前拒绝新上传。
幂等重放不受后续模型配置变化影响；新登记仍在知识库行锁内重新检查冻结 Profile。

所选文本/视觉向量模型必须与服务端 `knowledge.aiops.representations` 中的
`model_ref` 一致。不匹配时返回 409 和 `AIOPS_KNOWLEDGE_PROFILE_MISMATCH`，
提示管理员补齐模型、维度、版本及检索方向，不再只返回 `INVALID_CONTRACT`。

每个表示空间必须来自实际模型与处理器配置，不能猜测维度、版本、归一化或跨模态能力。
模型目录中的 `model_revision`、`processor_revision`、`input_template` 必须与
实际推理回执和冻结表示空间相同。视觉模型还需要显式登记 `retrieval_directions`。
只验证配置可以保存，不代表解析、索引及发布全链路通过。

KBot 使用显式的 `knowledge.aiops.served_model_names` 映射，将表示空间中的业务模型
UUID 映射到模型目录的推理服务名；映射必须完整且不含多余模型。
Ammolite 直接使用其模型 ID 推理合同，不添加 KBot 专属服务名配置。

本修复不修改数据库结构，不清理数据。部署配置变更需按实际进程加载方式生效；
模型目录元数据变更需刷新相应模型实例。部署前先确认定向重启范围。
