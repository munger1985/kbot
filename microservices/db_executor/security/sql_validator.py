import re

class SQLValidator:
    # 禁止执行的 DDL 和 DML 关键字
    FORBIDDEN_PATTERN = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|GRANT|REVOKE|EXECUTE|EXEC)\b", 
        re.IGNORECASE
    )

    @classmethod
    def validate(cls, sql: str) -> tuple[bool, str]:
        clean_sql = sql.strip()
        
        # 1. 准入检查：必须是查询
        if not re.match(r"^\s*(SELECT|WITH)\b", clean_sql, re.IGNORECASE):
            return False, "拒绝执行: 只允许执行 SELECT 或 WITH 语句。"

        # 2. 危险关键字扫描
        if cls.FORBIDDEN_PATTERN.search(clean_sql):
            return False, "拒绝执行: 不能执行 DDL 或 DML 语句。"

        # 3. 语句注入检查（禁止多条语句）
        if ";" in clean_sql and clean_sql.rstrip(";").find(";") != -1:
            return False, "拒绝执行: 不能执行多条语句。"

        return True, ""

    @classmethod
    def inject_limit(cls, sql: str, db_type: str, limit: int = 100) -> str:
        """根据数据库类型动态注入限流语句"""
        sql = sql.rstrip().rstrip(";")
        upper_sql = sql.upper()
        
        if db_type.lower() == "oracle":
            # Oracle 12c+ 语法：FETCH FIRST N ROWS ONLY 或 OFFSET M ROWS FETCH NEXT N ROWS ONLY
            # 如果已经包含 FETCH 或 ROWNUM，不再注入
            if "FETCH" in upper_sql and "ROWS ONLY" in upper_sql:
                return sql  # 已有限制语句，直接返回
            if "ROWNUM" in upper_sql:
                return sql  # 已使用 ROWNUM 限制
            
            # 检查是否已有 OFFSET ... FETCH
            if "OFFSET" in upper_sql and "FETCH" in upper_sql:
                return sql  # 已完整的分页语句
                
            # 注入标准的分页语法
            return f"{sql} OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        elif db_type.lower() in ["postgresql", "postgres"]:
            # PostgreSQL 支持 LIMIT 和 OFFSET
            if "LIMIT" not in upper_sql:
                return f"{sql} LIMIT {limit}"
            return sql
        
        elif db_type.lower() == "mysql":
            # MySQL 支持 LIMIT
            if "LIMIT" not in upper_sql:
                return f"{sql} LIMIT {limit}"
            return sql
        
        elif db_type.lower() == "clickhouse":
            # ClickHouse 支持 LIMIT
            if "LIMIT" not in upper_sql:
                return f"{sql} LIMIT {limit}"
            return sql
        
        else:
            # 默认使用 LIMIT
            if "LIMIT" not in upper_sql:
                return f"{sql} LIMIT {limit}"
            return sql