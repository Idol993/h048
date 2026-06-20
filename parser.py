import json
import re
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from config import AppConfig, LogFormatConfig


LEVEL_MAP = {
    "DEBUG": 10,
    "TRACE": 10,
    "INFO": 20,
    "NOTICE": 25,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
    "ERR": 40,
    "FATAL": 50,
    "CRITICAL": 50,
    "CRIT": 50,
    "ALERT": 50,
    "EMERGENCY": 60,
}


@dataclass
class LogEntry:
    timestamp: Optional[datetime] = None
    level: str = "INFO"
    message: str = ""
    service: str = ""
    source_file: str = ""
    line_number: int = 0
    raw: str = ""
    is_valid: bool = True
    level_value: int = 20


@dataclass
class ParseResult:
    entries: List[LogEntry] = field(default_factory=list)
    format_detected: str = "unknown"
    format_confidence: float = 0.0


@dataclass
class FeatureResult:
    entries: List[LogEntry] = field(default_factory=list)
    tfidf_matrix: Optional[csr_matrix] = None
    vectorizer: Optional[TfidfVectorizer] = None
    feature_names: List[str] = field(default_factory=list)
    numeric_features: Optional[np.ndarray] = None


class LogParser:
    def __init__(self, config: AppConfig):
        self.config = config
        self.lf: LogFormatConfig = config.log_formats
        self._compiled_patterns: List[Tuple[str, re.Pattern]] = []
        self._compile_patterns()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted_vectorizer: bool = False

    def _compile_patterns(self) -> None:
        for p in self.lf.plain_patterns:
            self._compiled_patterns.append((p.get("name", "unknown"), re.compile(p["pattern"])))

    @staticmethod
    def _extract_service_from_path(filepath: str) -> str:
        base = os.path.basename(filepath)
        name, _ = os.path.splitext(base)
        parts = re.split(r"[_\-.]", name)
        for p in parts:
            if p and not p.isdigit() and len(p) > 1:
                return p
        return name or "unknown"

    @staticmethod
    def _parse_timestamp(ts_str: str, default_fmt: Optional[str] = None) -> Optional[datetime]:
        ts_str = ts_str.strip().strip("[]")
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%d/%b/%Y:%H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%d_%H:%M:%S",
            "%b %d %H:%M:%S",
            "%b %d %Y %H:%M:%S",
        ]
        if default_fmt:
            formats.insert(0, default_fmt)
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except (ValueError, TypeError):
                continue
        try:
            ts_float = float(ts_str)
            return datetime.fromtimestamp(ts_float)
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _normalize_level(level: str) -> Tuple[str, int]:
        if level is None:
            return "INFO", 20
        level_up = str(level).upper().strip()
        if level_up in LEVEL_MAP:
            return level_up, LEVEL_MAP[level_up]
        for key in LEVEL_MAP:
            if key in level_up or level_up in key:
                return key, LEVEL_MAP[key]
        return "INFO", 20

    def _parse_plain_line(self, line: str) -> Optional[Dict[str, Any]]:
        for name, pattern in self._compiled_patterns:
            m = pattern.match(line)
            if m:
                gd = m.groupdict()
                return {
                    "format": name,
                    "timestamp": gd.get("timestamp"),
                    "level": gd.get("level"),
                    "message": gd.get("message", ""),
                    "service": gd.get("service"),
                }
        return None

    def _parse_json_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                return None
            level_val = None
            for k in self.lf.json_level_fields:
                if k in obj and obj[k]:
                    level_val = obj[k]
                    break
            msg_val = ""
            for k in self.lf.json_message_fields:
                if k in obj and obj[k]:
                    msg_val = str(obj[k])
                    break
            ts_val = None
            for k in self.lf.json_timestamp_fields:
                if k in obj and obj[k]:
                    ts_val = obj[k]
                    break
            svc_val = None
            for k in self.lf.json_service_fields:
                if k in obj and obj[k]:
                    svc_val = str(obj[k])
                    break
            return {
                "format": "json",
                "timestamp": ts_val,
                "level": level_val,
                "message": msg_val,
                "service": svc_val,
                "extra_fields": {k: v for k, v in obj.items() if not isinstance(v, (dict, list))},
            }
        except (json.JSONDecodeError, ValueError):
            return None

    def _parse_logfmt_line(self, line: str) -> Optional[Dict[str, Any]]:
        if "=" not in line:
            return None
        if line.startswith("{") or "[" in line[:25] and "]" in line[:25]:
            return None
        pattern = re.compile(self.lf.logfmt_pattern) if self.lf.logfmt_pattern else re.compile(r'([\w\-]+)=("[^"]*"|\S+)')
        matches = pattern.findall(line)
        if not matches or len(matches) < 3:
            return None
        total_covered = 0
        for k, v in matches:
            total_covered += len(k) + 1 + len(v)
        if total_covered < len(line) * 0.5:
            return None
        fields = {}
        for k, v in matches:
            fields[k] = v.strip('"')
        required_keys = set(self.lf.logfmt_level_keys + self.lf.logfmt_message_keys + self.lf.logfmt_timestamp_keys)
        if not any(k in fields for k in required_keys):
            return None
        level_val = None
        for k in self.lf.logfmt_level_keys:
            if k in fields:
                level_val = fields[k]
                break
        msg_val = ""
        for k in self.lf.logfmt_message_keys:
            if k in fields:
                msg_val = fields[k]
                break
        ts_val = None
        for k in self.lf.logfmt_timestamp_keys:
            if k in fields:
                ts_val = fields[k]
                break
        if level_val is None and not msg_val and ts_val is None:
            return None
        return {
            "format": "logfmt",
            "timestamp": ts_val,
            "level": level_val,
            "message": msg_val,
            "service": fields.get("service"),
            "extra_fields": fields,
        }

    def detect_format(self, sample_lines: List[str]) -> Tuple[str, float]:
        counts = {"json": 0, "logfmt": 0, "plain": 0}
        for line in sample_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                if self._parse_json_line(stripped):
                    counts["json"] += 1
                    continue
            if "=" in stripped:
                if self._parse_logfmt_line(stripped):
                    counts["logfmt"] += 1
                    continue
            if self._parse_plain_line(stripped):
                counts["plain"] += 1
                continue
        total = sum(counts.values())
        if total == 0:
            return "unknown", 0.0
        best = max(counts, key=counts.get)
        return best, counts[best] / total

    def parse_line(self, line: str, source_file: str = "", line_number: int = 0) -> LogEntry:
        stripped = line.strip()
        if not stripped:
            return LogEntry(raw=line, source_file=source_file, line_number=line_number, is_valid=False)

        entry = LogEntry(raw=line, source_file=source_file, line_number=line_number)
        parsed = None

        if stripped.startswith("{"):
            parsed = self._parse_json_line(stripped)
        if parsed is None:
            parsed = self._parse_plain_line(stripped)
        if parsed is None and "=" in stripped:
            parsed = self._parse_logfmt_line(stripped)
        if parsed is None:
            parsed = self._parse_plain_line(stripped)

        if parsed is None:
            entry.message = stripped
            entry.is_valid = False
            return entry

        if parsed.get("timestamp"):
            entry.timestamp = self._parse_timestamp(str(parsed["timestamp"]))
        level_str, level_val = self._normalize_level(parsed.get("level"))
        entry.level = level_str
        entry.level_value = level_val
        entry.message = str(parsed.get("message", "")) or stripped
        svc = parsed.get("service")
        entry.service = str(svc) if svc else self._extract_service_from_path(source_file)
        return entry

    def parse_file_stream(self, filepath: str) -> Iterator[LogEntry]:
        sample = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i < 50:
                    sample.append(line)
                yield self.parse_line(line, filepath, i + 1)

    def parse_files(self, file_patterns: List[str]) -> List[LogEntry]:
        files = []
        for pattern in file_patterns:
            matched = glob.glob(pattern, recursive=True)
            files.extend(matched)
        if not files:
            return []
        entries: List[LogEntry] = []
        for fp in files:
            entries.extend(list(self.parse_file_stream(fp)))
        return entries

    def parse_files_streaming(self, file_patterns: List[str]) -> Iterator[List[LogEntry]]:
        files = []
        for pattern in file_patterns:
            matched = glob.glob(pattern, recursive=True)
            files.extend(matched)
        if not files:
            return
        chunk: List[LogEntry] = []
        chunksize = self.config.streaming.chunksize
        for fp in files:
            for entry in self.parse_file_stream(fp):
                chunk.append(entry)
                if len(chunk) >= chunksize:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    def create_vectorizer(self, fit_messages: Optional[List[str]] = None) -> TfidfVectorizer:
        cfg = self.config.feature
        min_df = cfg.tfidf_min_df
        if fit_messages:
            valid_msgs = [m for m in fit_messages if m and m.strip()]
            if len(valid_msgs) < min_df * 10:
                min_df = max(1, min(len(valid_msgs), 2))
        self._vectorizer = TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            min_df=min_df,
            max_df=cfg.tfidf_max_df,
            stop_words=cfg.stop_words if cfg.stop_words != "none" else None,
        )
        if fit_messages:
            valid_msgs = [m for m in fit_messages if m and m.strip()]
            try:
                if len(valid_msgs) >= min_df:
                    self._vectorizer.fit(valid_msgs)
                    self._is_fitted_vectorizer = True
            except Exception:
                if len(valid_msgs) >= 1:
                    self._vectorizer.min_df = 1
                    try:
                        self._vectorizer.fit(valid_msgs)
                        self._is_fitted_vectorizer = True
                    except Exception:
                        self._is_fitted_vectorizer = False
        return self._vectorizer

    def extract_numeric_features(self, entries: List[LogEntry]) -> np.ndarray:
        n = len(entries)
        if n == 0:
            return np.array([]).reshape(0, 8)
        features = np.zeros((n, 8), dtype=np.float32)
        ts_seconds: List[float] = []
        base_ts = None
        for e in entries:
            if e.timestamp is not None:
                base_ts = e.timestamp
                break
        prev_ts = base_ts
        for i, e in enumerate(entries):
            features[i, 0] = float(e.level_value) / 60.0
            features[i, 1] = float(len(e.message)) / 500.0 if e.message else 0.0
            num_tokens = len(e.message.split()) if e.message else 0
            features[i, 2] = float(num_tokens) / 100.0
            if e.timestamp is not None and prev_ts is not None:
                ts_diff = (e.timestamp - prev_ts).total_seconds()
                features[i, 5] = min(float(abs(ts_diff)) / 60.0, 1.0)
                prev_ts = e.timestamp
            if e.timestamp is not None and base_ts is not None:
                ts_elapsed = (e.timestamp - base_ts).total_seconds()
                ts_seconds.append(ts_elapsed)
                max_ts_day = 24 * 3600
                features[i, 6] = min(float(ts_elapsed) / max_ts_day, 1.0)
            else:
                ts_seconds.append(0.0)
            error_keywords = ["error", "fail", "exception", "traceback", "fatal", "critical", "panic"]
            msg_lower = e.message.lower() if e.message else ""
            features[i, 3] = float(sum(1 for kw in error_keywords if kw in msg_lower)) / len(error_keywords)
            has_stack = 1.0 if ("traceback" in msg_lower or ("at " in msg_lower and ".java:" in msg_lower) or "File \"" in e.message) else 0.0
            features[i, 4] = has_stack
            special_chars = sum(1 for c in e.message if c in "!@#$%^&*()[]{}<>?\\/") if e.message else 0
            features[i, 7] = min(float(special_chars) / 20.0, 1.0)
        return features

    def extract_features(self, entries: List[LogEntry], vectorizer: Optional[TfidfVectorizer] = None) -> FeatureResult:
        result = FeatureResult(entries=entries)
        if not entries:
            return result
        messages = [e.message or "" for e in entries]
        if vectorizer is not None:
            self._vectorizer = vectorizer
            self._is_fitted_vectorizer = True
        if self._vectorizer is None:
            self.create_vectorizer(fit_messages=messages)
        try:
            valid_msgs = [m if m.strip() else " " for m in messages]
            if not self._is_fitted_vectorizer:
                self._vectorizer.fit(valid_msgs)
                self._is_fitted_vectorizer = True
            tfidf = self._vectorizer.transform(valid_msgs)
            result.tfidf_matrix = csr_matrix(tfidf)
            result.vectorizer = self._vectorizer
            result.feature_names = list(self._vectorizer.get_feature_names_out())
        except Exception:
            self.create_vectorizer(fit_messages=messages)
            try:
                valid_msgs = [m if m.strip() else " " for m in messages]
                tfidf = self._vectorizer.transform(valid_msgs)
                result.tfidf_matrix = csr_matrix(tfidf)
                result.vectorizer = self._vectorizer
                result.feature_names = list(self._vectorizer.get_feature_names_out())
            except Exception:
                pass
        result.numeric_features = self.extract_numeric_features(entries)
        return result

    def combine_features(self, tfidf_matrix: Optional[csr_matrix], numeric_features: Optional[np.ndarray]) -> Optional[csr_matrix]:
        if tfidf_matrix is None and numeric_features is None:
            return None
        if tfidf_matrix is None:
            return csr_matrix(numeric_features)
        if numeric_features is None or len(numeric_features.shape) == 0:
            return tfidf_matrix
        if numeric_features.shape[0] != tfidf_matrix.shape[0]:
            return tfidf_matrix
        try:
            from scipy.sparse import hstack
            return hstack([tfidf_matrix, csr_matrix(numeric_features)]).tocsr()
        except Exception:
            return tfidf_matrix

    def entries_to_dataframe(self, entries: List[LogEntry]) -> pd.DataFrame:
        rows = []
        for e in entries:
            rows.append({
                "timestamp": e.timestamp,
                "level": e.level,
                "level_value": e.level_value,
                "message": e.message,
                "service": e.service,
                "source_file": e.source_file,
                "line_number": e.line_number,
                "is_valid": e.is_valid,
            })
        return pd.DataFrame(rows)
