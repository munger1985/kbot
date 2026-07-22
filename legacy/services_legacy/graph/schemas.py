from pydantic import BaseModel, Field

class VertexSchema(BaseModel):
    """
    知识图谱顶点（实体）的抽取 Schema
    """
    vertex_name: str = Field(description="实体名称，应为具体、有现实含义的词，如 'Oracle 26ai' 或 'PostgreSQL'")
    vertex_type: str = Field(description="实体类型，大写英文标识，如 'DATABASE', 'PROJECT', 'TECH', 'PERSON'")
    vertex_desc: str | None = Field(default=None, description="关于该实体的简要描述、上下文解释或属性补充")

class EdgeSchema(BaseModel):
    """
    知识图谱边（关系）的抽取 Schema
    """
    source_name: str = Field(description="源实体的名称（对应某个顶点的 vertex_name）")
    target_name: str = Field(description="目标实体的名称（对应另一个顶点的 vertex_name）")
    relation_type: str = Field(description="两者之间的关系类型，大写英文下划线标识，如 'SUPPORT_INDEX', 'INTEGRATED_IN', 'DEVELOPED_BY'")

class GraphAnalysis(BaseModel):
    """
    大模型图谱抽取的根响应模型，直接对接 AIModelClient.get_llm_json 吐出的数据
    """
    vertices: list[VertexSchema] = Field(default_factory=list, description="从文本中抽取出的所有实体顶点列表")
    edges: list[EdgeSchema] = Field(default_factory=list, description="从文本中抽取出的所有关系边列表")
