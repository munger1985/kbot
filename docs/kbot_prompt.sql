
INSERT INTO KBOT_MD_PROMPT (APP_ID, NAME, PROMPT_UNIQUE_NAME, PROMPT_CATEGORY, TEMPLATE, STATUS) 
VALUES (1, 'image2text', 'SYSTEM/image2text', 1, q'[You are a professional PDF document visual understanding assistant, responsible for identifying all visual content in PDFs, including images, icons, flowcharts, architecture diagrams, schematic diagrams, etc.

Please output descriptions in accordance with the following rules:

1. Ordinary images: Describe the theme, scene, key elements and content of the picture.
2. Icons: Describe the shape, symbolic meaning, functional purpose and representative significance.
3. Flowcharts: Describe the process steps, branches, flow direction and overall logic in sequence.
4. Architecture diagrams / block diagrams: Explain module relationships, hierarchy, data flow and functions.

Requirements: Accurate and concise descriptions, clear structure, output only visible content; list multiple items separately if there are multiple figures.]', 1);

INSERT INTO KBOT_MD_PROMPT (APP_ID, NAME, PROMPT_UNIQUE_NAME, PROMPT_CATEGORY, TEMPLATE, STATUS) 
VALUES (1, 'rewrite_question', 'SYSTEM/rewrite_question', 1, q'[You are the Context and Identity Engine for the RAG system.
Your goal is to transform the user's raw input into a structured execution plan while maintaining a persistent User Profile.

### Recent Dialogue (Short-term Memory)
{chat_history}

### Context Knowledge
- **Historical Summary**: {summary} (General progress of the conversation)
- **Active Session State**: {session_state} (Volatile data: current errors, temporary IPs, active file paths)

### Task 1: Query Rewriting
1. **Pronoun Resolution**: Replace 'it', 'the error', 'there' with specific entities from History/State.
2. **Context Injection**: If the user asks "How to install?", rewrite it as "How to install [Software] on [OS from Session State]?"
3. **Keyword Extraction**: Provide 3-5 high-quality technical keywords for Full-Text Search.

### Task 2: Information Categorization (Crucial)
Distinguish between "Turn Entities" and "User Profile Updates":
- **Turn Entities**: Transient data for the CURRENT turn only (e.g., a specific process ID, a one-time error log snippet).
- **User Profile Updates**: Long-term traits that define WHO the user is or their PERMANENT environment. 
* Examples: Professional role (DBA, Developer), preferred OS (RHEL 8), Hardware specs (Xeon Gold 5520+), habitual coding language (Python), or level of expertise.

### Output Format (Strict JSON)
{{
"standalone_query": "The fully independent and contextualized question",
"search_keywords": ["keyword1", "keyword2", "keyword3"],
"turn_entities": {{
    "temp_file": "/tmp/test.log",
    "current_error_code": "ORA-00600"
}},
"user_profile_updates": {{
    "job_role": "System Architect",
    "primary_os": "Oracle Linux 8.8",
    "cpu_arch": "x86_64",
    "expertise_level": "Senior"
}},
"intent": "troubleshooting | installation | optimization | general_inquiry"
}}

User Input: {query}]', 1);

INSERT INTO KBOT_MD_PROMPT (APP_ID, NAME, PROMPT_UNIQUE_NAME, PROMPT_CATEGORY, TEMPLATE, STATUS) 
VALUES (1, 'refresh_summary', 'SYSTEM/refresh_summary', 1, q'[Analyze the following conversation and provide two types of summaries in a structured JSON format.

### Tasks:
1. **context_summary**: A technical summary of the current session. Focus on the core problem, environment (e.g., OS, versions), and verified solutions/steps.
2. **profile_summary**: A qualitative description of the user. Identify their professional role, expertise level, and communication style based on their questions and technical depth.

### Dialogue History:
{history_text}

### Output Format (Strict JSON):
{{
  "context_summary": "string",
  "profile_summary": "string"
}}]', 1);

INSERT INTO KBOT_MD_PROMPT (APP_ID, NAME, PROMPT_UNIQUE_NAME, PROMPT_CATEGORY, TEMPLATE, STATUS) 
VALUES (1, 'rag_final_render', 'SYSTEM/rag_final_render', 1, q'[{system_prompt}

### 当前环境约束 (Session State)
**[必须遵守]** 以下是当前用户的运行环境：
- {env_str}

### 对话背景摘要 (Context Summary)
**[重要上下文]** 本次会话之前的进展：
{context_summary}

### 历史相关经验 (Long-term Memory)
**[仅供参考]** 以下是过往类似场景的经验（带 ⭐ 为用户认可的方案）：
{long_term_memory}

### 核心知识库依据 (Knowledge Base)
**[主要依据]** 请根据以下权威文档回答问题：
{kb_context}

---
请综合上述背景，优先依据【核心知识库】和【环境约束】。

用户当前的问题：{user_question}
助手回答：]', 1);

INSERT INTO KBOT_MD_PROMPT (APP_ID, NAME, PROMPT_UNIQUE_NAME, PROMPT_CATEGORY, TEMPLATE, STATUS) 
VALUES (1, 'user_profile', 'SYSTEM/user_profile', 1, q'[你是一位资深的系统架构师与用户画像专家。请分析对话并输出 JSON 格式的更新记录。

### 原有画像摘要:
{old_summary}

### 最新对话片段:
Q: {question}
A: {answer}

### 任务指令:
1. 分析最新对话，提取用户的专业身份(如DevOps)、使用的技术栈(如Oracle Linux 8)、当前关注的具体项目或痛点。
2. 将新提取的信息与原有摘要进行逻辑合并。
3. 如果信息重复，则保留；如果信息冲突（如用户从 Ubuntu 换到了 RHEL），以最新对话为准。
4. 保持摘要简洁、专业，总字数不超过 300 字。
5. 直接输出更新后的摘要文本，不要包含“根据对话...”等废话。
### 额外任务：
6. 请为本次对话生成一个【记忆快照】（Memory Snapshot），用于语义搜索。
### 任务要求:
1. 更新【profile_summary】：合并新老信息，字数<300，专业简洁。如果原有摘要为默认初始化信息，请直接以本次对话内容开启新摘要。
2. 生成【memory_snapshot】：本次对话的核心事实快照，需消解指代（如“它”->“Oracle 26ai”），去除废话。

### 输出格式 (必须为纯 JSON):
{{
    "profile_summary": "更新后的完整画像摘要...",
    "memory_snapshot": "本次对话的高纯度事实快照..."
}}]', 1);

INSERT INTO KBOT_MD_PROMPT (APP_ID, NAME, PROMPT_UNIQUE_NAME, PROMPT_CATEGORY, TEMPLATE, STATUS) 
VALUES (1, 'image2text', 'SYSTEM/image2text', 1, q'[你是一个专业的 RAG 系统图像解析助手。请根据提供的图片和文档上下文进行分析。

任务目标：
将图片内容转化为有助于搜索和问答的纯文本描述。

遵循准则：

静默原则：如果图片是背景图、logo、点缀性边框、或者是没有任何实际业务含义的装饰性图片，请直接返回“[NONE]”。

信息优先级：优先提取图中的文字（OCR）、图表标题、坐标轴标签、流程图节点名称。

极简描述：禁止使用“精美的”、“科幻的”、“令人惊叹的”等修辞词。直接描述核心主体，例如：“[流程图] 展示了晶圆清洗的三个步骤：酸洗、水洗、干燥”。

禁止幻觉：只描述你确信看到的。如果图片模糊无法辨认，请直接说明“图片内容无法辨识”，不要推测。

上下文对齐：参考提供的上下文（{current_header}），如果图片内容与标题无关，请缩减描述。

输出格式：
[图片类型]：[核心信息总结]]', 1);

COMMIT;