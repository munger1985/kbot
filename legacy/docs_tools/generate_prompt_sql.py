#!/usr/bin/env python3
"""
从 agent/prompt/default_prompt.py 自动生成 init_prompt_data.sql (Oracle 语法)。

用法:
    cd /home/chris/KBot
    conda run -n cube python docs/database/generate_init_sys_data.py

每当 default_prompt.py 中新增或修改提示词后，运行此脚本更新 SQL 文件。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompt.default_prompt import DefaultPrompt

# ============================================================
# 配置变量 - 可根据需要修改
# ============================================================
APP_ID = 1001                 # 应用ID，默认1001
DOMAIN_ID = None              # 域ID，None表示全局提示词
PROMPT_CATEGORY = 0           # 提示词类别：0=系统提示词
STATUS = 0                    # 状态：0=启用
CREATED_BY = 'System'         # 创建人
UPDATED_BY = 'System'         # 更新人
DESCRIPTION = '系统自动生成，请勿删除'  # 描述信息
# ============================================================

HEADER = f"""-- ============================================================
-- KBot 系统提示词 — 种子数据
-- 自动生成自 agent/prompt/default_prompt.py
-- 生成时间: {{timestamp}}
-- 表: KBOT_MD_PROMPT
-- ============================================================
-- 配置信息:
--   APP_ID: {APP_ID}
--   DOMAIN_ID: {DOMAIN_ID if DOMAIN_ID is not None else 'NULL (全局)'}
--   PROMPT_CATEGORY: {PROMPT_CATEGORY}
--   STATUS: {STATUS}
-- ============================================================

SET DEFINE OFF;

BEGIN

"""

FOOTER = """
    COMMIT;
END;
/

SET DEFINE ON;

-- 验证插入结果
SELECT PROMPT_ID, PROMPT_UNIQUE_NAME, NAME, PROMPT_CATEGORY, STATUS 
FROM KBOT_MD_PROMPT 
WHERE CREATED_BY = '{CREATED_BY}';

"""


def escape_sql(text: str) -> str:
    """转义单引号，处理换行和特殊字符"""
    # 转义单引号
    text = text.replace("'", "''")
    # 处理 Oracle CLOB 中的换行符
    text = text.replace("\r\n", "\n")
    return text


def get_current_timestamp():
    """获取当前时间戳用于 SQL 注释"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_domain_id_sql():
    """获取 DOMAIN_ID 的 SQL 表示"""
    if DOMAIN_ID is None:
        return "NULL"
    return str(DOMAIN_ID)


def generate():
    dp = DefaultPrompt()
    prompts = dp._prompts

    lines = []
    lines.append(HEADER.format(timestamp=get_current_timestamp()))

    domain_id_value = get_domain_id_sql()

    for i, (prompt_unique_name, content) in enumerate(prompts.items()):
        safe_content = escape_sql(content)
        safe_prompt_unique_name = escape_sql(prompt_unique_name)
        safe_name = escape_sql(prompt_unique_name)  # NAME 也可以使用相同的唯一名称

        # Oracle 语法：使用 MERGE 实现 upsert
        lines.append(
            f"    -- 提示词: {safe_name}\n"
            f"    MERGE INTO KBOT_MD_PROMPT p\n"
            f"    USING (SELECT '{safe_prompt_unique_name}' AS PROMPT_UNIQUE_NAME FROM DUAL) src\n"
            f"    ON (p.PROMPT_UNIQUE_NAME = src.PROMPT_UNIQUE_NAME)\n"
            f"    WHEN MATCHED THEN\n"
            f"        UPDATE SET\n"
            f"            TEMPLATE = '{safe_content}',\n"
            f"            UPDATED_BY = '{UPDATED_BY}',\n"
            f"            UPDATED_TIME = SYSDATE\n"
            f"    WHEN NOT MATCHED THEN\n"
            f"        INSERT (\n"
            f"            APP_ID,\n"
            f"            DOMAIN_ID,\n"
            f"            NAME,\n"
            f"            PROMPT_UNIQUE_NAME,\n"
            f"            PROMPT_CATEGORY,\n"
            f"            TEMPLATE,\n"
            f"            STATUS,\n"
            f"            DESCS,\n"
            f"            CREATED_BY,\n"
            f"            CREATED_TIME\n"
            f"        ) VALUES (\n"
            f"            {APP_ID},\n"
            f"            {domain_id_value},\n"
            f"            '{safe_name}',\n"
            f"            '{safe_prompt_unique_name}',\n"
            f"            {PROMPT_CATEGORY},\n"
            f"            '{safe_content}',\n"
            f"            {STATUS},\n"
            f"            '{DESCRIPTION}',\n"
            f"            '{CREATED_BY}',\n"
            f"            SYSDATE\n"
            f"        );\n"
        )
        
        # 每100条提交一次（可选，避免事务过大）
        if (i + 1) % 100 == 0:
            lines.append(f"    COMMIT;\n")

    lines.append(FOOTER)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_prompt_data.sql")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"✅ 已生成 {output_path}")
    print(f"   共 {len(prompts)} 条系统提示词")
    print(f"\n📋 配置信息:")
    print(f"   - APP_ID: {APP_ID}")
    print(f"   - DOMAIN_ID: {DOMAIN_ID if DOMAIN_ID is not None else 'NULL (全局)'}")
    print(f"   - PROMPT_CATEGORY: {PROMPT_CATEGORY}")
    print(f"   - STATUS: {STATUS}")
    
    # 打印前几条记录的信息
    print(f"\n📋 生成的提示词列表:")
    for i, name in enumerate(list(prompts.keys())[:10]):
        print(f"   - {name}")
    if len(prompts) > 10:
        print(f"   ... 共 {len(prompts)} 条")


if __name__ == "__main__":
    generate()