"""案例可共享内容的确定性脱敏；自动处理不替代发布前人工核对。"""

import re

from aiops_agent.contracts.knowledge import DiagnosisCaseProfile


class CaseRedactor:
    version = "case-redaction.v1"

    def __init__(self):
        self.mapping = {}
        self.counts = {}

    def _placeholder(self, category, value):
        key = (category, value)
        if key not in self.mapping:
            self.counts[category] = self.counts.get(category, 0) + 1
            self.mapping[key] = f"<{category}_{self.counts[category]}>"
        return self.mapping[key]

    def text(self, value):
        patterns = (
            ("secret", r"(?i)\b(?:password|passwd|pwd|token|api[_-]?key|secret)\s*[:=]\s*[^\s;,]+"),
            ("address", r"https?://[^\s<>]+"),
            ("email", r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
            ("host", r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            ("path", r"(?<!\w)(?:[A-Za-z]:\\|/(?:home|var|opt|etc|mnt|data|u[0-9]+|tmp)/)[^\s;,<>]+"),
        )
        for category, pattern in patterns:
            value = re.sub(pattern, lambda match: self._placeholder(category, match.group(0)), value)
        return value

    def content(self, value, key=""):
        if isinstance(value, dict):
            return {name: self.content(item, str(name).lower()) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.content(item, key) for item in value]
        if not isinstance(value, str):
            return value
        if key in {"password", "token", "secret", "api_key", "username", "user_name", "hostname", "host_name", "customer", "customer_name", "business_identifier"}:
            return self._placeholder("identifier", value)
        value = self.text(value)
        if key in {"sql", "sql_text", "draft_sql", "statement", "command_text"}:
            value = re.sub(r"'(?:''|[^'])*'", lambda match: self._placeholder("literal", match.group(0)), value)
        return value

    def profile(self, profile):
        data = profile.model_dump(mode="json")
        # 来源标识只用于受控血缘，不改写引用或复制原始证据正文。
        source = data.pop("source")
        data.pop("redaction", None)
        data = self.content(data)
        data["source"] = source
        data["redaction"] = {
            "policy_version": self.version, "status": "REVIEW_REQUIRED",
            "replacement_counts": dict(self.counts),
        }
        return DiagnosisCaseProfile.model_validate(data)
