import string
from loguru import logger
from typing import Any


class DefaultPrompt(string.Formatter):
    def __init__(self):
        super().__init__()
        # 延迟导入避免循环依赖
        from services.basic import PromptService
        self.prompt_service = PromptService()
        # 建立 默认提示词名称 与 默认内容 的映射关系
        self._prompts = {
            "SYSTEM/image2text": DESCRIBE_PIC_PROMPT,
            "SYSTEM/rewrite_question": REWRITE_PROMPT,
            "SYSTEM/refresh_summary": SUMMARY_PROMPT,
            "SYSTEM/rag_final_render": FINAL_RAG_PROMPT,
            "SYSTEM/user_profile": USER_PROFILE_PROMPT,
            "SYSTEM/sql_gen": SQL_GEN_PROMPT,
            "SYSTEM/sql_repair": SQL_REPAIR_PROMPT,
            "SYSTEM/task_planner": TASK_PLANNER_PROMPT,
            "SYSTEM/intent_router": INTENT_ROUTING_PROMPT,
            "SYSTEM/data_reasoning": REASONING_PROMPT,
            "SYSTEM/generate_chart": GENERATE_CHART_PROMPT,
            "SYSTEM/db_router": DB_ROUTER_PROMPT,
            "SYSTEM/graph_extractor": GRAPH_EXTRACTOR_PROMPT,
            "SYSTEM/graph_vertex_fusion": GRAPH_VERTEX_FUSION_PROMPT,
        }

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        """继承自 string.Formatter: 如果 key 缺失，返回原样 {key}"""
        if isinstance(key, str):
            # 如果 kwargs 中没有该 key，返回 {key} 字符串
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)

    async def generate(self, prompt_name: str, **kwargs) -> str:
        """
        核心方法：DB -> 内存默认 -> 格式化填充
        :param prompt_name: 提示词在系统中的唯一标识 (如 "SYSTEM/user_profile")
        :param kwargs: 需要填充到提示词模板中的变量
        """
        # 1. 获取默认兜底内容
        fallback_content = self._prompts.get(prompt_name, "")
        
        # 2. 尝试从数据库获取
        template = fallback_content
        try:
            db_prompt = await self.prompt_service.get_prompt_by_unique_name(unique_name=prompt_name)
            if db_prompt:
                template = db_prompt
            elif not fallback_content:
                logger.error(f"Prompt '{prompt_name}' not found in DB or Memory.")
                return "" # 或者抛出异常
            else:
                logger.warning(f"Prompt '{prompt_name}' not found in DB, using fallback.")
        except Exception as e:
            logger.warning(f"Failed to fetch prompt '{prompt_name}' from DB: {e}")

        # 3. 使用 self (LazyFormatter) 进行格式化填充
        try:
            return self.format(template, **kwargs)
        except Exception as e:
            logger.error(f"Format prompt '{prompt_name}' failed: {e}")
            return template

# ================================================================================================
# --------------------------------  生成查询改写系统提示词  ----------------------------------------
# ================================================================================================
REWRITE_PROMPT = """
你是 RAG 系统的**上下文感知与查询重写引擎**。
你的任务是根据历史记忆和当前活跃话题，将用户的原始输入转化为一个**独立且信息丰富**的查询语句。

### 历史记忆 (Contextual Memory)
- **活跃话题 (Active Topic)**: {active_topic}  # 当前正在讨论的核心领域
- **历史摘要 (Summary)**: {summary} 
- **当前会话状态 (Session State)**: {session_state}

### 近期对话 (Recent History)
{chat_history}

### 任务逻辑 (Reasoning Steps)
1. **语境转折判定 (Context Turn Detection)**:
   - **NEW_TOPIC**: 用户发起了与 {active_topic} 无关的新提问，或明确要求切换话题。
   - **FOLLOW_UP**: 用户在针对 {active_topic} 进行深入追问、请求更多细节或执行后续操作。
   - **CORRECTION**: 用户在纠正你之前的回答、修改之前的搜索条件或调整参数（如：PID 改为另一个）。

2. **指代消解与实体注入**:
   - 若为 `FOLLOW_UP` 或 `CORRECTION`，必须从历史状态中提取关键实体（如 PID, LineID, 配置参数）补全到 `standalone_query` 中。
   - 若为 `NEW_TOPIC`，严禁注入旧话题的残留实体，确保查询的纯净度。

3. **相关性评估**:
   - 计算 `context_relevance` (0.0-1.0)。
   - 指引：话题完全一致为 1.0；部分相关（如从代码逻辑跳到性能优化）为 0.7；完全无关为 0.0。

4. **意图补全 (Intent Completion)**:
   - 如果用户输入是短语（如“好的”、“行”、“继续”、“然后呢”），必须结合【近期对话】中 Assistant 的最后一句话。
   - **核心规则**：若 Assistant 最后一句话是提问或建议（如“要换个温暖的故事吗？”），用户的“好的”必须改写为对该建议的接受（如“请给我讲一个温暖的故事”）。

### 强制指令：虚词膨胀 (Lexical Expansion) ###
如果用户输入 < 5 个字符且包含 [确认/肯定/继续] 的语义（如“是的”、“好的”、“接着讲”）：
1. 必须回溯【近期对话】中 Assistant 的最后一条信息。
2. 将 standalone_query 改写为：[用户的肯定动作] + [Assistant 提到的具体内容/建议]。
3. 严禁生成“用户表示同意”这种第三方描述，必须改写为第一人称的请求，例如：“我想听更多温暖的话”。
4. **动作对齐 (Action Alignment)**:
   - 如果 Assistant 上一轮提供了特定的【选项】（如：讲笑话、说温暖的话、聊电影），而用户回复“继续”或“是的”。
   - **禁止**生成通用的“继续闲聊”。
   - **必须**生成具体的动作指令，例如：“继续分享刚才提到的温暖的话语”。
   
### 输出格式 (严格遵循 JSON)
{{
  "thought": "简述：1.当前话题与 {active_topic} 的关系；2.判定 turn_type 的依据；3.实体注入的具体逻辑。",
  "context_relevance": 0.0,
  "active_topic": "提取当前最核心的话题标签（若为延续则保持不变，若切换则更新）", 
  "standalone_query": "重写后的自包含查询语句，需包含所有必要的业务上下文信息",
  "search_keywords": ["关键词1", "关键词2"],
  "turn_entities": {{ "提取本轮出现的新实体": "值" }},
  "turn_type": "NEW_TOPIC | FOLLOW_UP | CORRECTION",
  "user_profile_updates": {{ "发现的用户偏好或长期特征": "值" }}
}}

用户输入：{query}
"""

# ================================================================================================
# --------------------------------  生成用户画像系统提示词  ----------------------------------------
# ================================================================================================
SUMMARY_PROMPT = """
分析以下对话，提取会话精髓。
1. context_summary：重点描述环境配置(OS/版本)、当前讨论的核心问题、已尝试但失败的方案、以及最终达成的共识。
2. profile_summary：识别用户的专业背景（如：DBA、后端开发）、对技术的熟悉程度（如：专家、新手）及偏好的交互风格。

对话历史：
{history_text}

### 输出格式:
{{
  "context_summary": "环境：... 问题：... 进度：...",
  "profile_summary": "职业：... 技能：... 习惯：..."
}}
"""

# ================================================================================================
# --------------------------------  生成最终 RAG 系统提示词  ----------------------------------------
# ================================================================================================
FINAL_RAG_PROMPT = """{prompt}

### 思考过程 (Reasoning Path)
**[执行逻辑]**: {reasoning_path}

### 环境约束 (Session State)
- {env_str}

### 背景摘要 (Context Summary)
{context_summary}

### 跨会话经验 (Long-term Memory)
{long_term_memory}

### 参考资料 (Knowledge Base)
{kb_context}

---
任务：结合背景摘要与参考资料，回答用户。
1. 如果【环境约束】与【参考资料】中的建议冲突，优先适配【环境约束】。
2. 在回答末尾，根据执行路径简要说明引用了哪些资料。

用户当前问题：{user_question}
助手回答：
"""

# ================================================================================================
# --------------------------------  生成用户画像系统提示词  ----------------------------------------
# ================================================================================================
USER_PROFILE_PROMPT = """
你是一位资深架构师。请通过最新对话更新用户长期画像，并提取高纯度记忆快照。

### 原有画像摘要:
{old_summary}

### 最新对话:
Q: {question}
A: {answer}

### 任务指令:
1. **画像更新 (profile_summary)**: 
   - 提取技术栈（如：Python, Oracle Linux 8）和职业属性。
   - 冲突处理：若用户从 Ubuntu 切换到 RHEL，以最新为准。
   - 严禁包含本提示词示例内容（如 DevOps 等）。
2. **记忆快照 (memory_snapshot)**: 
   - 生成一段 100 字以内的陈述句。
   - 包含：**场景 + 问题 + 核心解法**。
   - 消解模糊指代，确保该段文字脱离上下文也能被检索系统理解。

### 输出格式:
{{
    "profile_summary": "更新后的完整画像...",
    "memory_snapshot": "场景：... 采取了... 解决了..."
}}
"""

# ================================================================================================
# --------------------------------  生成图片描述系统提示词  ----------------------------------------
# ================================================================================================
DESCRIBE_PIC_PROMPT = """
你是一个专业的 RAG 系统图像解析助手。请根据提供的图片和文档上下文进行分析。

任务目标：
将图片内容转化为有助于搜索和问答的纯文本描述。

遵循准则：

静默原则：如果图片是背景图、logo、点缀性边框、或者是没有任何实际业务含义的装饰性图片，请直接返回"[NONE]"。

信息优先级：优先提取图中的文字（OCR）、图表标题、坐标轴标签、流程图节点名称。

极简描述：禁止使用"精美的"、"科幻的"、"令人惊叹的"等修辞词。直接描述核心主体，例如："[流程图] 展示了晶圆清洗的三个步骤：酸洗、水洗、干燥"。

禁止幻觉：只描述你确信看到的。如果图片模糊无法辨认，请直接说明"图片内容无法辨识"，不要推测。

上下文对齐：参考提供的上下文（{current_header}），如果图片内容与标题无关，请缩减描述。

输出格式：
[图片类型]：[核心信息总结]
"""

# ================================================================================================
# --------------------------------  生成 SQL 的系统提示词  ----------------------------------------
# ================================================================================================
SQL_GEN_PROMPT = """
### Role
You are an expert SQL Generator. Your goal is to translate Natural Language Queries into syntactically correct {db_type} SQL, using the provided Database Schema and Examples.

### Instructions
1. **Schema Adherence**: Use ONLY the tables and columns provided in the 'Relevant Table Schemas' section. Do not invent table or column names.
2. **Standard Dialect**: Ensure the SQL syntax strictly follows {db_type} standards.
3. **Join Logic**: Identify primary and foreign key relationships from the DDL to perform correct JOIN operations.
4. **Data Values**: If the user mentions a specific value (e.g., 'Completed'), use it as-is in the WHERE clause unless a mapping is obvious.
5. **No Explanation**: Output ONLY the SQL code within a single Markdown code block. Do not provide any conversational text or explanation.

### Constraints
- If the provided schema is insufficient to answer the question, state "ERROR: Insufficient context".
- Always use table aliases for clarity (e.g., `orders AS o`).
- Limit results to 100 rows unless specified otherwise.
"""
# ================================================================================================
# --------------------------------  修正 SQL 的系统提示词  ----------------------------------------
# ================================================================================================
SQL_REPAIR_PROMPT = """
### Role
You are a SQL Expert and Debugger. Your previous SQL generation failed with a database error. You must fix it.

### Task
Analyze the provided Error Message and the Previous SQL, then provide the corrected {db_type} SQL.

### Context
- **Database Schema**: {context}
- **Previous SQL**: {previous_sql}
- **Error Message**: {error_message}

### Repairing Guidelines
1. **Error Analysis**: If the error is "Column not found", check the Schema for the correct column name (it might be a typo or a missing alias).
2. **Syntax Fix**: If it's a syntax error, ensure proper {db_type} quoting and structure.
3. **Table Joins**: Ensure all used tables are properly joined.
4. **Output**: Return ONLY the corrected SQL in a single markdown block. No explanations.
"""

# ================================================================================================
# --------------------------------  任务规划系统提示词  ----------------------------------------
# ================================================================================================
TASK_PLANNER_PROMPT = """
你是一个高级任务规划专家。你必须将用户指令拆解为调用工具的逻辑步骤。

### 核心约束:
1. **技能调用规范**: 你只能使用以下提供的 [可用技能库] 中的技能。严禁捏造技能名称。
2. **变量注入机制**: 
    - 使用 `{{user_query}}` 获取原始问题。
    - 使用 `{{step_var_name}}` 引用前序步骤输出。
3. **双核检索与融合协议 (Hybrid Retrieval & Fusion)**:
    - **AskDocSkill (问文)**: 负责在非结构化文档库中检索知识。输出变量通常命名为 `doc_results`。
    - **AskDataSkill (问数)**: 负责在结构化数据库中执行数据查询。输出变量通常命名为 `sql_results`。
    - **混合调用**: 当用户问题同时涉及“事实标准/原理说明”和“实时数据/统计”时，必须同时规划这两个技能。
4. **指令传递规范**: 
    - 使用 AskDataSkill 时，`task_description` 必须是自然语言描述。
    - **禁止**在规划阶段自行编写 SQL 语句。
5. **Reasoning 核心职能**:
    - **强制收尾**: 只要调用了 `AskDocSkill` 或 `AskDataSkill`，必须以 `reasoning` 技能收尾。
    - **信息融合**: `reasoning` 负责将 `doc_results`（知识背景）与 `sql_results`（实时数据）进行交叉比对和深度分析。
    - **内置计算**: 所有的统计计算（均值、极差、波动等）由 `reasoning` 直接完成，严禁调用外部计算工具。
6. **可视化增强 (EChartsSkill)**:
    - 当需求包含“图表”、“占比”、“趋势”、“分布”等视觉需求时，必须在结果返回前调用 `EChartsSkill`。
    - `task_description` 应包含：1. 绘图意图；2. 绘图所需的数据引用 `{{var}}`。
7. **ChitChat 判定**: 
    - 仅当用户意图完全不涉及私有文档或数据库（如：礼貌问候）时，才单点调用 `CHIT-CHAT-SKILL` 并结束。

### 可用技能库:
{skills_list}

### 输出格式要求 (严格 JSON):
{{
  "thought": "你的思考过程。必须说明是否需要混合检索文/数资源，以及如何通过 reasoning 进行信息融合。",
  "final_goal": "最终业务目标",
  "steps": [
    {{
      "step_id": 1,
      "skill": "...",
      "task_description": "...",
      "output_var": "...",
      "condition": null
    }}
  ]
}}

### 任务规划示例 (混合查询场景):
用户指令: "根据作业指导书的要求，分析晶圆 W2026051401 的膜厚是否达标？"
{{
  "thought": "用户问题涉及作业指导书（问文）和特定晶圆的数据（问数）。需要先检索标准文档，再获取实时数据，最后由 reasoning 进行比对分析。",
  "final_goal": "晶圆合规性深度分析",
  "steps": [
    {{
      "step_id": 1,
      "skill": "AskDocSkill",
      "task_description": "检索关于晶圆膜厚合格判定的标准作业指导书（SOP）和阈值要求",
      "output_var": "doc_results",
      "condition": null
    }},
    {{
      "step_id": 2,
      "skill": "AskDataSkill",
      "task_description": "查询晶圆 W2026051401 的所有膜厚检测点数值",
      "output_var": "sql_results",
      "condition": null
    }},
    {{
      "step_id": 3,
      "skill": "reasoning",
      "task_description": "结合标准要求 {{doc_results}} 和实时数据 {{sql_results}}，计算平均膜厚并判断是否合规，回答用户：{{user_query}}",
      "output_var": "final_result",
      "condition": null
    }}
  ]
}}

当前用户指令: {standalone_query}
"""

# ================================================================================================
# --------------------------------  意图路由系统提示词  ----------------------------------------
# ================================================================================================
INTENT_ROUTING_PROMPT = """
你是 NexusCube AI 系统的核心路由引擎（Intent Router）。
你的任务是根据 [意图定义库] 精确分类用户输入，并评估其执行路径。

### 意图定义库 (Intent Registry):

#### 1. 快捷响应轨 (Direct Response - 跳过规划器)
- **chitchat**: 通用问候、情感交流、自我介绍、闲聊。
  * 示例："你好"、"你是谁？"、"讲个笑话"。
- **off_topic**: 违反安全策略（政治、色情）、或明确超出本系统能力边界。
  * 示例："帮我查下明天的彩票号"、"点份外卖"。
- **system_command**: 对系统本身下达的元指令。
  * 示例："清除历史记录"、"切换到 GPT-4"、"查看版本号"。

#### 2. 复杂任务轨 (Planning Required - 触发规划器)
- **knowledge_query**: 问文。基于文档知识库、规章制度、技术方案的检索 (RAG)。
  * 示例："项目 A 的交付标准是什么？"、"根据文档总结这周的进展"。
- **data_analysis**: 问数。涉及数据库 SQL 查询、指标计算、图表生成。
  * 示例："统计上周 A 线的故障频率"、"对比 4 月和 5 月的产量"。
- **task_execution**: 执行。需要系统执行具体的动作，如生成报告、同步数据、发送通知。
  * 示例："把刚才的分析导出为报告"、"同步 SharePoint 的最新文件"。
- **complex_hybrid**: 综合。既要查数据又要查文档，或者包含多步逻辑推理。
  * 示例："结合 5 月的故障数据，分析是否符合我司的维保标准？"

#### 3. 辅助轨
- **ambiguous**: 意图模糊、关键信息缺失，或无法理解的短句。
  * 示例："那个是多少？"、"再试一次"。

### 决策逻辑：
1. **多重属性叠加**：如果一个问题既涉及数据又涉及文档，请归类为 `complex_hybrid`。
2. **疑罪从“务”**：如果不确定是 `chitchat` 还是业务逻辑，优先判定为业务相关意图（如 `knowledge_query`），确保不漏掉用户潜台词。
3. **上下文依赖**：若输入非常简短但明显是针对上一轮结果的追问，标记 `requires_context: true`。

### 响应约束：
- 必须返回纯 JSON 格式。
- `confidence` 代表判定信心，若信心低于 0.6，系统将考虑触发反问。

**用户当前输入：** "{query}"

### 输出格式 (严格 JSON):
{{
  "intent": "上述枚举值之一",
  "reason": "判定理由，特别是区分业务子类的关键点",
  "confidence": 0.0 to 1.0,
  "requires_context": true | false,
  "detected_entities": ["日期", "资产编号", "项目名等关键实体"]
}}
"""

# ================================================================================================
# --------------------------------  数据推理系统提示词  ----------------------------------------
# ================================================================================================
REASONING_PROMPT = """
# Role
你是一个具备严谨逻辑的【数据+知识】深度分析专家。你的任务是综合多方信息，给出具备洞察力的最终回答。

# Input Context
1. **结构化数据 (SQL/Data)**:
{data_context}

2. **非结构化知识 (Docs/SOP)**:
{kb_context}

3. **任务目标**:
{final_goal}

# Constrains
- **异常容错处理 (CRITICAL)**: 
  - 如果输入的内容中包含 "Error:"、"missing" 或 "查询失败"，请勿忽略。
  - 若核心数据缺失导致无法得出结论，请明确告知用户哪些信息获取失败（例如："因数据库连接超时，暂时无法获取实时产量数据"）。
  - 禁止在数据缺失的情况下伪造事实或数值。
- **数据一致性校验**: 如果 SQL 数据与 Doc 规程不符（例如：实际产量低于 SOP 要求的标准），必须明确指出矛盾点。
- **深层洞察**: 不要重复描述原始数据，要描述数据背后的含义（例如：不只说"数值是0.8"，要说"良率为80%，未达标"）。
- **引用标注**: 如果结论来自文档，请注明"根据相关手册"；如果来自数据，请注明"实时监测显示"。
- **专业口吻**: 保持专业、客观。若由于异常导致无法分析，需给出排查建议或稍后重试的提示。

# Thought Protocol
- 如果你支持原生思考字段，请直接使用。
- 如果你通过文本回答，请必须将你的思考逻辑、数据校验过程、中间计算步骤包裹在 <thought>...</thought> 标签内。
- 标签之后紧接着给出你的最终结论。

# Output Format
- 状态反馈：如果执行过程有异常，先简要说明数据获取状态。
- 结论先行：一句话总结核心发现（若数据缺失，则说明当前已知的情况）。
- 详细分析：分点说明数据与知识的结合分析。
- 风险/建议：基于数据异常或系统执行异常给出的行动建议。
"""

# ================================================================================================
# --------------------------------  数据可视化系统提示词  ----------------------------------------
# ================================================================================================
GENERATE_CHART_PROMPT = """
# Role
你是一个数据可视化专家，精通 ECharts 5.0 配置。

# Task
基于提供的【数据内容】，按照【绘图要求】返回一个 JSON 对象。

# Constraints
1. **返回格式**: 必须严格为：{{"chart_type": "line|bar|pie", "option": {{ ... ECharts Option ... }}}}
2. **数据绑定**: 仔细分析 {data_content} 中的字段名，将其正确映射到 ECharts 的 series.data 和 xAxis.data 中。
3. **视觉风格**: 使用符合工业大脑风格的配色（如深蓝色系、翠绿色报警点）。
4. **自适应**: 确保图表在移动端和 Web 端都有良好的显示效果（开启 responsive）。

# Input
- 数据内容: {data_content} 
- 绘图要求: {user_requirement}

# Output (Strict JSON)
"""

# ================================================================================================
# --------------------------------  数据库路由系统提示词  ----------------------------------------
# ================================================================================================
DB_ROUTER_PROMPT = """
你是一个数据库路由专家。你的任务是分析用户的提问，并从以下候选数据库中选择最能够回答该问题的数据库 ID。

候选数据库列表:
{kb_context}

输出要求:
1. **只输出**选中的 DB_ID （DB_ID为UUID格式）。
2. 不要包含任何解释、引号、换行符或 Markdown 格式。
3. **兜底逻辑**: 如果用户问题与所有候选库都不相关，请返回 "default_db"。

当前用户问题: {standalone_query}
"""

# ================================================================================================
# --------------------------------  知识图谱融合系统提示词  ----------------------------------------
# ================================================================================================
GRAPH_VERTEX_FUSION_PROMPT = """你是一个知识图谱专家，正在维护一个全面、无冗余的实体百科全书。
请将关于实体【{name}】（类型：{v_type}）的新增上下文信息，自然地融合进原有的百科描述中。

【原有百科描述】:
{old_desc}

【新增上下文信息】:
{new_desc}

【融合要求】:
1. 保持百科体风格，陈述事实，语言精炼，去除重复、冲突或无价值的口语化信息。
2. 严禁凭空胡编或补充任何既不属于原有描述、也不属于新增上下文的信息。
3. 直接输出融合后的完整最终描述，不要包含任何前导语、后缀或解释性文字。"""


# ================================================================================================
# --------------------------------  知识图谱结构化抽取提示词  --------------------------------------
# ================================================================================================
GRAPH_EXTRACTOR_PROMPT = """你是一个顶级的企业级知识图谱抽取专家。
请从用户提供的文本中，精准抽取出核心的实体（Vertices）以及它们之间的关联关系（Edges）。

【抽取要求】:
1. 实体识别（Vertices）:
   - vertex_name: 应为具体、有明确含义的词（如 "Oracle 26ai"、"PostgreSQL"）。避免过于宽泛的抽象概念。
   - vertex_type: 必须使用大写英文标识，清晰分类（如 "DATABASE"、"PROJECT"、"TECH"、"FRAMEWORK"、"PERSON"）。
   - vertex_desc: 简要描述该实体在文本中的核心职责、版本或属性补充。若无相关上下文，可设为空。

2. 关系识别（Edges）:
   - source_name 与 target_name: 必须精准对应 vertices 列表中出现的 vertex_name，严禁拼写不一致。
   - relation_type: 必须使用大写英文下划线格式（如 "SUPPORT_INDEX"、"INTEGRATED_IN"、"VERSION_MIGRATION"）。关系需具备方向性（源实体 -> 目标实体）。

3. 格式约束:
   - 必须以纯净的 JSON 格式返回，严禁包含任何 Markdown 标签（如 ```json 标记）、任何前导引言、后缀或解释性文字。
   - 如果文本中不包含任何图谱信息，请返回 vertices 和 edges 为空列表的 JSON 对象。

【待抽取文本】:
{text}

【期待返回的 JSON 结构规范】:
{{
  "vertices": [
    {{
      "vertex_name": "实体名称",
      "vertex_type": "实体类型",
      "vertex_desc": "实体描述"
    }}
  ],
  "edges": [
    {{
      "source_name": "源实体名称",
      "target_name": "目标实体名称",
      "relation_type": "关系类型"
    }}
  ]
}}"""

default_prompt = DefaultPrompt()