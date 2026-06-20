import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from detector import AnomalyResult


@dataclass
class LogTemplate:
    template: str = ""
    count: int = 0
    services: Dict[str, int] = field(default_factory=dict)
    levels: Dict[str, int] = field(default_factory=dict)
    example_indices: List[int] = field(default_factory=list)
    example_messages: List[str] = field(default_factory=list)
    mean_score: float = 0.0
    min_score: float = 0.0
    severity_rank: float = 0.0

    @property
    def services_list(self) -> List[str]:
        return sorted(self.services.keys(), key=lambda x: -self.services[x])


VARIABLE_PATTERNS: List[Tuple[str, str, Any, int]] = [
    ("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b", "<IP>", 0),
    ("ipv6", r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", "<IP>", 0),
    ("uuid", r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "<UUID>", 0),
    ("timestamp_iso", r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?\b", "<TS>", 0),
    ("timestamp_alt", r"\b\d{4}/\d{2}/\d{2}[T ]\d{2}:\d{2}:\d{2}\b", "<TS>", 0),
    ("date_yyyy_mm_dd", r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", 0),
    ("time_hh_mm_ss", r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<TIME>", 0),
    ("hex", r"\b0x[0-9a-fA-F]+\b", "<HEX>", 0),
    ("number_int", r"\b\d{6,}\b", "<NUM>", 0),
    ("number_cost", r"\b\$\d+(?:\.\d{1,2})?\b", "<AMOUNT>", 0),
    ("number_ver", r"\bv\d+\.\d+(?:\.\d+)*\b", "<VER>", 0),
    ("url", r"\bhttps?://[^\s<>\"']+", "<URL>", 0),
    ("email", r"\b[\w\.\-]+@[\w\.\-]+\b", "<EMAIL>", 0),
    ("path_unix", r"(?:/[\w\-\.]+){2,}", "<PATH>", 0),
    ("path_win", r"\b[a-zA-Z]:\\(?:[\w\-\.]+\\)+[\w\-\.]*", "<PATH>", 0),
    ("stacktrace_line", r"\s+at\s+[\w\.\$]+\([\w\.\-]+:\d+\)", "<STACK_LINE>", 0),
    ("error_code", r"\b(?:error|err|code|status)[_\- ]?\s*[=:]\s*\d+", "error_code=<CODE>", re.IGNORECASE),
    ("order_id", r"\border[_\- ]?id[_\- ]?\s*[=:]\s*#?\w+", "order_id=<ORDER_ID>", re.IGNORECASE),
    ("user_id", r"\buser[_\- ]?id[_\- ]?\s*[=:]\s*\w+", "user_id=<USER_ID>", re.IGNORECASE),
    ("request_id", r"\brequest[_\- ]?id[_\- ]?\s*[=:]\s*[\w\-]+", "request_id=<REQ_ID>", re.IGNORECASE),
    ("cart_id", r"\bcart[_\- ]?id[_\- ]?\s*[=:]\s*[\w\-]+", "cart_id=<CART_ID>", re.IGNORECASE),
    ("tracking_id", r"\btracking[_\- ]?id[_\- ]?\s*[=:]\s*[\w\-]+", "tracking_id=<TRK_ID>", re.IGNORECASE),
    ("sku", r"\bSKU[_\- ]?\s*[=:]\s*[\w\-]+", "SKU=<SKU>", 0),
    ("addr_id", r"\baddr(?:ess)?[_\- ]?id[_\- ]?\s*[=:]\s*[\w\-]+", "addr_id=<ADDR_ID>", re.IGNORECASE),
    ("device_id", r"\bdevice[_\- ]?id[_\- ]?\s*[=:]\s*[\w\-]+", "device_id=<DEV_ID>", re.IGNORECASE),
    ("session_id", r"\bsession[_\- ]?id[_\- ]?\s*[=:]\s*[\w\-]+", "session_id=<SESS_ID>", re.IGNORECASE),
    ("card_ending", r"\bcard[_\- ]?ending[_\- ]?\s*[=:]\s*\d+", "card_ending=<CARD>", re.IGNORECASE),
    ("latency_ms", r"\b(?:latency_ms|response_time|elapsed)[_\- ]?\s*[=:]\s*\d+", "latency=<LATENCY>", re.IGNORECASE),
    ("retry_count", r"\bretry[_\- ]?count[_\- ]?\s*[=:]\s*\d+", "retry_count=<RETRY>", re.IGNORECASE),
    ("http_status", r"\b\d{3}\b.*\b(?:status|code)\b", "<HTTP_STATUS>", re.IGNORECASE),
    ("java_error_line", r"\.java:\d+\)", ".java:<LINE>)", 0),
    ("python_file_line", r'File "[^"]+", line \d+', r'File "<FILE>", line <LINE>', 0),
]


TOKEN_PATTERN = re.compile(r'(\b[A-Z][A-Z0-9_]+\b|"[^"]+"|\'[^\']+\'|\d+[a-zA-Z]+|[a-zA-Z]+\d+|[a-zA-Z]+|\S)')


def _apply_variable_replacement(message: str) -> str:
    result = message
    for name, pattern, repl, flags in VARIABLE_PATTERNS:
        try:
            compiled = re.compile(pattern, flags) if flags else re.compile(pattern)
            result = compiled.sub(repl, result)
        except Exception as e:
            print(f"[template] pattern '{name}' failed: {e}")
    return result


def _tokenize(message: str) -> List[str]:
    return [t for t in TOKEN_PATTERN.findall(message) if t]


def _jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def _lcs_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    return lcs_len / max(m, n)


def _merge_templates(t1: str, t2: str) -> str:
    toks1 = _tokenize(t1)
    toks2 = _tokenize(t2)
    merged: List[str] = []
    i = j = 0
    while i < len(toks1) and j < len(toks2):
        t1_tok = toks1[i]
        t2_tok = toks2[j]
        if t1_tok == t2_tok:
            merged.append(t1_tok)
            i += 1
            j += 1
        elif t1_tok.startswith("<") and t1_tok.endswith(">"):
            merged.append(t1_tok)
            i += 1
        elif t2_tok.startswith("<") and t2_tok.endswith(">"):
            merged.append(t2_tok)
            j += 1
        else:
            merged.append("<*>")
            i += 1
            j += 1
    while i < len(toks1):
        merged.append(toks1[i])
        i += 1
    while j < len(toks2):
        merged.append(toks2[j])
        j += 1
    return " ".join(merged)


class TemplateExtractor:
    def __init__(self, similarity_threshold: float = 0.85, use_lcs: bool = True):
        self.similarity_threshold = similarity_threshold
        self.use_lcs = use_lcs
        self.templates: List[LogTemplate] = []
        self._template_tokens: List[List[str]] = []

    def normalize_message(self, message: str) -> str:
        msg = message.strip()
        msg = _apply_variable_replacement(msg)
        return msg

    def _find_template_index(self, normalized_msg: str) -> Optional[int]:
        tokens = _tokenize(normalized_msg)
        for i, t_tokens in enumerate(self._template_tokens):
            if self.use_lcs:
                sim = _lcs_similarity(tokens, t_tokens)
            else:
                sim = _jaccard_similarity(tokens, t_tokens)
            if sim >= self.similarity_threshold:
                return i
        return None

    def add_message(
        self,
        message: str,
        service: str = "",
        level: str = "",
        score: float = 0.0,
        original_index: int = 0,
    ) -> int:
        normalized = self.normalize_message(message)
        idx = self._find_template_index(normalized)
        if idx is None:
            template = LogTemplate(
                template=normalized,
                count=1,
                mean_score=score,
                min_score=score,
                example_indices=[original_index],
                example_messages=[message[:500]],
            )
            if service:
                template.services[service] = 1
            if level:
                template.levels[level] = 1
            self.templates.append(template)
            self._template_tokens.append(_tokenize(normalized))
            return len(self.templates) - 1
        else:
            t = self.templates[idx]
            t.count += 1
            old_norm = t.template
            new_norm = normalized
            merged = _merge_templates(old_norm, new_norm)
            if merged != old_norm:
                t.template = merged
                self._template_tokens[idx] = _tokenize(merged)
            t.mean_score = (t.mean_score * (t.count - 1) + score) / t.count
            t.min_score = min(t.min_score, score)
            if service:
                t.services[service] = t.services.get(service, 0) + 1
            if level:
                t.levels[level] = t.levels.get(level, 0) + 1
            if len(t.example_messages) < 5:
                t.example_indices.append(original_index)
                t.example_messages.append(message[:500])
            return idx

    def fit(self, anomaly_results: List[AnomalyResult]) -> List[LogTemplate]:
        for i, a in enumerate(anomaly_results):
            if not a.is_anomaly:
                continue
            self.add_message(
                message=a.entry.message or "",
                service=a.entry.service or "",
                level=a.entry.level or "",
                score=a.score,
                original_index=i,
            )
        return self.rank_templates()

    def rank_templates(self) -> List[LogTemplate]:
        if not self.templates:
            return []
        max_count = max(t.count for t in self.templates) or 1
        max_score = max(abs(t.min_score) for t in self.templates) or 1
        max_services = max(len(t.services) for t in self.templates) or 1
        for t in self.templates:
            count_rank = t.count / max_count
            score_rank = abs(t.min_score) / max_score
            service_rank = len(t.services) / max_services
            t.severity_rank = 0.4 * count_rank + 0.4 * score_rank + 0.2 * service_rank
        self.templates.sort(key=lambda x: -x.severity_rank)
        return self.templates

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "template": t.template,
                "count": t.count,
                "services": t.services,
                "levels": t.levels,
                "mean_score": round(t.mean_score, 4),
                "min_score": round(t.min_score, 4),
                "severity_rank": round(t.severity_rank, 4),
                "examples": t.example_messages,
                "example_indices": t.example_indices,
            }
            for t in self.templates
        ]
