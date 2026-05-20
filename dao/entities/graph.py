from datetime import datetime
from sqlalchemy import String, CLOB, Integer, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, VectorField


class GraphVertexEntity(BaseEntity):
    """知识图谱顶点表（实体表）"""
    __tablename__ = "KBOT_GRAPH_KNOWLEDGE_VERTICES"

    vertex_id: Mapped[str] = mapped_column(String(64), name="VERTEX_ID", primary_key=True, comment="顶点ID，推荐名称+类型MD5")
    vertex_name: Mapped[str] = mapped_column(String(255), name="VERTEX_NAME", comment="实体或概念的实际名称（如'RTX 5080'）")
    vertex_type: Mapped[str] = mapped_column(String(64), name="VERTEX_TYPE", comment="实体的业务大类分类（如：技术、设备、指标）")
    description: Mapped[str | None] = mapped_column(CLOB, name="DESCRIPTION", comment="LLM对该实体在上下文中提炼的简要文本定义")
    attributes: Mapped[dict | None] = mapped_column(JSON, name="ATTRIBUTES", comment="JSON格式动态扩展属性，存储非固定字段")
    name_vector: Mapped[list | None] = mapped_column(VectorField(), name="NAME_VECTOR", comment="基于实体名称/描述生成的原生向量嵌入，用于消歧")
    created_at: Mapped[datetime] = mapped_column(name="CREATED_AT", server_default=func.now(), comment="实体的首次创建或抽取时间")
    updated_at: Mapped[datetime] = mapped_column(name="UPDATED_AT", server_default=func.now(), onupdate=func.now(), comment="最后一次更新时间")


class GraphEdgeEntity(BaseEntity):
    """知识图谱边表（关系表）"""
    __tablename__ = "KBOT_GRAPH_KNOWLEDGE_EDGES"

    edge_id: Mapped[str] = mapped_column(String(64), name="EDGE_ID", primary_key=True, comment="边的唯一标识，源ID+目标ID+关系类型MD5")
    source_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_GRAPH_KNOWLEDGE_VERTICES.VERTEX_ID"), name="SOURCE_ID", comment="源顶点ID（起点）")
    target_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_GRAPH_KNOWLEDGE_VERTICES.VERTEX_ID"), name="TARGET_ID", comment="目标顶点ID（终点）")
    relation_type: Mapped[str] = mapped_column(String(128), name="RELATION_TYPE", comment="关系的语义类型（如：属于、导致、测量）")
    weight: Mapped[int] = mapped_column(Integer, name="WEIGHT", server_default="1", comment="关系权重值，重复抽取时递增")
    attributes: Mapped[dict | None] = mapped_column(JSON, name="ATTRIBUTES", comment="JSON格式动态扩展属性，存储边特有附加信息")
    created_at: Mapped[datetime] = mapped_column(name="CREATED_AT", server_default=func.now(), comment="关系的首次创建或抽取时间")
    updated_at: Mapped[datetime] = mapped_column(name="UPDATED_AT", server_default=func.now(), onupdate=func.now(), comment="最后一次更新时间")


class GraphEdgeChunkMapEntity(BaseEntity):
    """图关系与文档切片映射表（中间关联表）"""
    __tablename__ = "KBOT_GRAPH_EDGE_CHUNK_MAP"

    edge_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_GRAPH_KNOWLEDGE_EDGES.EDGE_ID"), name="EDGE_ID", primary_key=True, comment="边ID")
    chunk_id: Mapped[str] = mapped_column(String(64), name="CHUNK_ID", primary_key=True, comment="提取出该关系的原始文档切片ID")
    file_id: Mapped[str | None] = mapped_column(String(64), name="FILE_ID", comment="冗余存储的文档唯一标识，方便级联清理")
    created_at: Mapped[datetime] = mapped_column(name="CREATED_AT", server_default=func.now(), comment="该条关系在当前切片下的提取时间")
    updated_at: Mapped[datetime] = mapped_column(name="UPDATED_AT", server_default=func.now(), onupdate=func.now(), comment="最后一次更新时间")