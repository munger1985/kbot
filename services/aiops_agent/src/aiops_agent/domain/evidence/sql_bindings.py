"""按数据库词法识别绑定位置；不改动字符串、注释和引用标识符。"""

from dataclasses import dataclass
import re

from sqlglot import Dialect
from sqlglot.tokens import TokenType


_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*\Z")


class SqlBindingInvalid(ValueError):
    """仅传播稳定错误码，不携带 SQL 或解析器原始错误。"""


@dataclass(frozen=True, slots=True)
class SqlBindPosition:
    start: int
    end: int
    name: str
    positional: bool


def sql_bind_positions(sql: str, *, dialect: str) -> tuple[SqlBindPosition, ...]:
    """返回左闭右开原文区间；Oracle 命名绑定不猜测对应的参数序号。"""
    try:
        tokens = Dialect.get_or_raise(dialect).tokenize(sql)
    except Exception:
        raise SqlBindingInvalid("SQL_BIND_TOKENIZATION_FAILED") from None
    positions = []
    bracket_depth = 0
    for index, token in enumerate(tokens):
        if token.token_type == TokenType.L_BRACKET:
            bracket_depth += 1
        elif token.token_type == TokenType.R_BRACKET:
            bracket_depth -= 1
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if dialect == "mysql" and token.token_type == TokenType.PLACEHOLDER and token.text == "?":
            positions.append(SqlBindPosition(token.start, token.end + 1, str(len(positions) + 1), True))
        elif (dialect == "postgres" and token.token_type == TokenType.PARAMETER
            and following is not None and following.start == token.end + 1
            and following.token_type == TokenType.NUMBER):
            positions.append(SqlBindPosition(token.start, following.end + 1, following.text, True))
        elif (token.token_type == TokenType.COLON and following is not None
            and following.start == token.end + 1):
            if (dialect == "postgres" and bracket_depth > 0 and index > 0
                and tokens[index - 1].token_type in {TokenType.NUMBER, TokenType.VAR, TokenType.IDENTIFIER,
                    TokenType.R_PAREN, TokenType.R_BRACKET, TokenType.STRING}):
                # 数组上下界间的冒号是切片运算符，例如 arr[1:upper]，不是名为 upper 的绑定。
                continue
            # 关键字也可能用作绑定名；必须同时核对原文，不能把 :"name" 当作同一协议。
            raw = sql[following.start:following.end + 1]
            if _NAME.fullmatch(raw):
                positions.append(SqlBindPosition(token.start, following.end + 1, raw, False))
            elif dialect == "oracle" and re.fullmatch(r"[0-9]+", raw):
                positions.append(SqlBindPosition(token.start, following.end + 1, raw, True))
    return tuple(positions)


def postgresql_named_bindings(sql: str) -> tuple[str, tuple[str, ...]]:
    """普通诊断通道只接受命名绑定；原生 $n 另走专用协议，不能与之混用。"""
    positions = sql_bind_positions(sql, dialect="postgres")
    if any(item.positional for item in positions):
        raise SqlBindingInvalid("SQL_BIND_STYLE_UNSUPPORTED")
    names = []
    pieces = []
    offset = 0
    for item in positions:
        if item.name not in names:
            names.append(item.name)
        pieces.extend((sql[offset:item.start], f"${names.index(item.name) + 1}"))
        offset = item.end
    pieces.append(sql[offset:])
    return "".join(pieces), tuple(names)
