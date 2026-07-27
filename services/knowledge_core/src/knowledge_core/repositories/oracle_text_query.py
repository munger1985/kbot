"""将自然语言问题转换为安全、宽松的 Oracle Text 查询。"""

from __future__ import annotations

import re

import jieba


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_.-]*|[0-9]+(?:\.[0-9]+)?|[\u3400-\u9fff]+"
)
_STOP_WORDS = {
    "请",
    "请问",
    "帮",
    "帮我",
    "一下",
    "有",
    "没有",
    "哪些",
    "什么",
    "怎么",
    "怎样",
    "如何",
    "是否",
    "可以",
    "能够",
    "相关",
    "关于",
    "介绍",
    "告诉",
    "列出",
    "说明",
    "问题",
}


def build_oracle_text_query(query: str, *, max_terms: int = 12) -> str:
    """提取有检索价值的词项，并使用 ACCUM 保持高召回。"""
    terms: list[str] = []
    normalized = " ".join(query.strip().split())
    for segment in jieba.cut(normalized, cut_all=False):
        for match in _TOKEN_PATTERN.findall(segment):
            term = match.strip()
            if not term or term.lower() in _STOP_WORDS:
                continue
            if _is_chinese(term) and len(term) == 1:
                continue
            if term not in terms:
                terms.append(term)
            if len(terms) >= max_terms:
                break
        if len(terms) >= max_terms:
            break
    if not terms:
        fallback = _TOKEN_PATTERN.findall(normalized)
        if fallback:
            terms.append(fallback[0][:64])
    return " ACCUM ".join(f"{{{term[:64]}}}" for term in terms)


def _is_chinese(value: str) -> bool:
    return all("\u3400" <= char <= "\u9fff" for char in value)
