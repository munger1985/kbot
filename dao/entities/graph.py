from datetime import datetime
from sqlalchemy import String, CLOB, Integer, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity


class GraphVertexEntity(BaseEntity):
    """知识图谱顶点表（实体表）"""
    __tablename__ = "kbot_graph_knowledge_vertices"

    vertex_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="顶点ID，推荐名称+类型MD5")
    vertex_name: Mapped[str] = mapped_column(String(255), comment="实体或概念的实际名称（如'RTX 5080'）")
    vertex_type: Mapped[str] = mapped_column(String(64), comment="实体的业务大类分类（如：技术、设备、指标）")
    description: Mapped[str | None] = mapped_column(CLOB, comment="LLM对该实体在上下文中提炼的简要文本定义")
    attributes: Mapped[dict | None] = mapped_column(JSON, comment="JSON格式动态扩展属性，存储非固定字段")
    name_vector: Mapped[list | None] = mapped_column(comment="基于实体名称/描述生成的原生向量嵌入，用于消歧")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), comment="实体的首次创建或抽取时间")
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), comment="最后一次更新时间")


class GraphEdgeEntity(BaseEntity):
    """知识图谱边表（关系表）"""
    __tablename__ = "kbot_graph_knowledge_edges"

    edge_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="边的唯一标识，源ID+目标ID+关系类型MD5")
    source_id: Mapped[str] = mapped_column(String(64), ForeignKey("kbot_graph_knowledge_vertices.vertex_id"), comment="源顶点ID（起点）")
    target_id: Mapped[str] = mapped_column(String(64), ForeignKey("kbot_graph_knowledge_vertices.vertex_id"), comment="目标顶点ID（终点）")
    relation_type: Mapped[str] = mapped_column(String(128), comment="关系的语义类型（如：属于、导致、测量）")
    weight: Mapped[int] = mapped_column(Integer, server_default="1", comment="关系权重值，重复抽取时递增")
    attributes: Mapped[dict | None] = mapped_column(JSON, comment="JSON格式动态扩展属性，存储边特有附加信息")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), comment="关系的首次创建或抽取时间")
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), comment="最后一次更新时间")


class GraphEdgeChunkMapEntity(BaseEntity):
    """图关系与文档切片映射表（中间关联表）"""
    __tablename__ = "kbot_graph_edge_chunk_map"

    edge_id: Mapped[str] = mapped_column(String(64), ForeignKey("kbot_graph_knowledge_edges.edge_id"), primary_key=True, comment="边ID")
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="提取出该关系的原始文档切片ID")
    file_id: Mapped[str | None] = mapped_column(String(64), comment="冗余存储的文档唯一标识，方便级联清理")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), comment="该条关系在当前切片下的提取时间")
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), comment="最后一次更新时间")