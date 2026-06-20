import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from config import AppConfig
from detector import AnomalyResult, DetectionResult
from parser import LogEntry, LogParser
from template import TemplateExtractor, LogTemplate


@dataclass
class KeywordInfo:
    keyword: str
    score: float
    frequency: int


@dataclass
class TimeBucket:
    bucket: str
    count: int
    anomaly_count: int


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    keywords: List[KeywordInfo] = field(default_factory=list)
    time_distribution: List[TimeBucket] = field(default_factory=list)
    services: Dict[str, int] = field(default_factory=dict)
    levels: Dict[str, int] = field(default_factory=dict)
    representative_messages: List[str] = field(default_factory=list)
    mean_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    severity_score: float = 0.0
    example_indices: List[int] = field(default_factory=list)


@dataclass
class RootCauseResult:
    clusters: List[ClusterInfo] = field(default_factory=list)
    global_keywords: List[KeywordInfo] = field(default_factory=list)
    top_services: List[Tuple[str, int]] = field(default_factory=list)
    overall_severity_score: float = 0.0
    noise_count: int = 0
    summary: str = ""
    templates: List[Any] = field(default_factory=list)


ERROR_PATTERNS = [
    r"(?:Connection|Timeout|Refused|Reset|Aborted|Failed|Failure|Error|Exception|Panic|Fatal|Crash|Unreachable|Unavailable|Not found|NotFound)",
    r"(?:NullPointer|NullReference|IndexOutOf|OutOfBounds|OutOfMemory|StackOverflow|Segmentation|SegFault)",
    r"(?:Permission|Denied|Unauthorized|Forbidden|Authentication|Authorization|Invalid|Corrupt|Corrupted)",
    r"(?:Disk|IO|Read|Write|Network|DNS|Host|Port|Socket|SSL|TLS|Certificate)",
]


class RootCauseExtractor:
    def __init__(self, config: AppConfig, parser: Optional[LogParser] = None):
        self.config = config
        self.cc = config.clustering
        self.parser = parser or LogParser(config)

    def _extract_error_keywords(self, messages: List[str]) -> Counter:
        counter = Counter()
        all_pattern = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)
        for msg in messages:
            if not msg:
                continue
            matches = all_pattern.findall(msg)
            for m in matches:
                counter[m.lower()] += 1
            words = re.findall(r"[A-Z][a-zA-Z]{2,}(?:[A-Z][a-zA-Z]*)*", msg)
            for w in words:
                if len(w) >= 4:
                    counter[w] += 1
            caps = re.findall(r"\b[A-Z_]{3,}\b", msg)
            for c in caps:
                counter[c] += 1
        return counter

    def _tfidf_keywords(self, messages: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        if len(messages) < 2:
            return []
        try:
            vec = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words=self.config.feature.stop_words if self.config.feature.stop_words != "none" else None,
                min_df=1,
            )
            tfidf_matrix = vec.fit_transform(messages)
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            feature_names = list(vec.get_feature_names_out())
            top_indices = scores.argsort()[-top_n * 2:][::-1]
            results = []
            for idx in top_indices:
                word = feature_names[idx]
                if re.search(r"[a-zA-Z]", word) and len(word) >= 2:
                    results.append((word, float(scores[idx])))
                if len(results) >= top_n:
                    break
            return results
        except Exception:
            return []

    def _cluster_messages(self, messages: List[str], feature_matrix: Optional[csr_matrix] = None) -> np.ndarray:
        n = len(messages)
        if n == 0:
            return np.array([])
        if feature_matrix is None or feature_matrix.shape[0] != n:
            try:
                vec = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words=self.config.feature.stop_words if self.config.feature.stop_words != "none" else None,
                )
                feature_matrix = vec.fit_transform([m or " " for m in messages])
            except Exception:
                return np.full(n, -1, dtype=int)
        min_samples = min(self.cc.min_samples, max(2, n // 10))
        min_samples = max(2, min_samples)
        try:
            db = DBSCAN(
                eps=self.cc.eps,
                min_samples=min_samples,
                metric=self.cc.metric,
            )
            labels = db.fit_predict(feature_matrix)
            return labels
        except Exception:
            try:
                db = DBSCAN(eps=0.8, min_samples=2, metric="cosine")
                labels = db.fit_predict(feature_matrix)
                return labels
            except Exception:
                return np.full(n, -1, dtype=int)

    def _time_distribution(self, timestamps: List[Optional[datetime]], bucket_type: str = "minute") -> List[TimeBucket]:
        buckets: Dict[str, Dict[str, int]] = {}
        for ts in timestamps:
            if ts is None:
                key = "unknown"
            elif bucket_type == "minute":
                key = ts.strftime("%Y-%m-%d %H:%M")
            elif bucket_type == "hour":
                key = ts.strftime("%Y-%m-%d %H:00")
            else:
                key = ts.strftime("%Y-%m-%d")
            if key not in buckets:
                buckets[key] = {"count": 0, "anomaly": 0}
            buckets[key]["count"] += 1
            buckets[key]["anomaly"] += 1
        sorted_keys = sorted(buckets.keys())
        result = []
        for k in sorted_keys:
            result.append(TimeBucket(
                bucket=k,
                count=buckets[k]["count"],
                anomaly_count=buckets[k]["anomaly"],
            ))
        return result

    def _extract_services(self, entries: List[LogEntry]) -> Dict[str, int]:
        counter = Counter()
        for e in entries:
            if e.service:
                counter[e.service] += 1
        return dict(counter.most_common())

    def _extract_levels(self, entries: List[LogEntry]) -> Dict[str, int]:
        counter = Counter()
        for e in entries:
            counter[e.level] += 1
        return dict(counter.most_common())

    def _select_representatives(self, messages: List[str], labels: np.ndarray, cluster_id: int, n: int = 5) -> List[int]:
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        if len(indices) <= n:
            return indices
        cluster_msgs = [messages[i] for i in indices]
        lengths = [len(m) for m in cluster_msgs]
        sorted_idx = sorted(range(len(lengths)), key=lambda x: -lengths[x])
        return [indices[i] for i in sorted_idx[:n]]

    def _compute_severity(self, cluster_size: int, mean_score: float, min_score: float, level_counts: Dict[str, int]) -> float:
        size_weight = min(cluster_size / 50.0, 1.0) * 0.3
        score_weight = abs(min_score) * 0.4
        level_weight = 0.0
        for lvl, cnt in level_counts.items():
            lvl_norm = {
                "ERROR": 1.0, "FATAL": 1.0, "CRITICAL": 1.0, "ERR": 1.0, "CRIT": 1.0, "ALERT": 1.0,
                "WARN": 0.6, "WARNING": 0.6,
                "INFO": 0.2, "NOTICE": 0.25,
                "DEBUG": 0.1, "TRACE": 0.1,
            }.get(lvl, 0.3)
            level_weight = max(level_weight, lvl_norm * (cnt / max(cluster_size, 1)))
        level_weight *= 0.3
        return size_weight + score_weight + level_weight

    def extract(self, detection_result: DetectionResult, feature_matrix: Optional[csr_matrix] = None) -> RootCauseResult:
        result = RootCauseResult()
        anomalies = [a for a in detection_result.anomaly_results if a.is_anomaly]
        if not anomalies:
            result.summary = "No anomalies detected in the log entries."
            return result

        anomaly_entries = [a.entry for a in anomalies]
        anomaly_messages = [e.message or "" for e in anomaly_entries]
        anomaly_scores = np.array([a.score for a in anomalies])

        labels = self._cluster_messages(anomaly_messages, feature_matrix)

        cluster_map: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            cluster_map[int(lbl)].append(i)

        noise_indices = cluster_map.get(-1, [])
        result.noise_count = len(noise_indices)

        tfidf_kw = self._tfidf_keywords(anomaly_messages, top_n=20)
        error_kw = self._extract_error_keywords(anomaly_messages)
        combined_scores: Dict[str, float] = {}
        for word, score in tfidf_kw:
            combined_scores[word] = score * 0.6
        for word, freq in error_kw.items():
            combined_scores[word] = combined_scores.get(word, 0) + freq * 0.4
        global_keywords_list = sorted(combined_scores.items(), key=lambda x: -x[1])[:15]
        for word, score in global_keywords_list:
            freq = error_kw.get(word, 0) or sum(1 for m in anomaly_messages if word.lower() in m.lower())
            result.global_keywords.append(KeywordInfo(keyword=word, score=round(score, 4), frequency=freq))

        result.top_services = list(Counter(e.service for e in anomaly_entries if e.service).most_common(10))

        all_severity = 0.0
        cluster_infos: List[ClusterInfo] = []

        for cluster_id, indices in cluster_map.items():
            if cluster_id == -1:
                continue
            c_entries = [anomaly_entries[i] for i in indices]
            c_messages = [anomaly_messages[i] for i in indices]
            c_scores = anomaly_scores[indices]

            c_keywords_list = []
            c_tfidf = self._tfidf_keywords(c_messages, top_n=10)
            c_error = self._extract_error_keywords(c_messages)
            c_combined: Dict[str, float] = {}
            for w, s in c_tfidf:
                c_combined[w] = s * 0.6
            for w, f in c_error.items():
                c_combined[w] = c_combined.get(w, 0) + f * 0.4
            c_sorted = sorted(c_combined.items(), key=lambda x: -x[1])[:10]
            for w, s in c_sorted:
                freq = c_error.get(w, 0) or sum(1 for m in c_messages if w.lower() in m.lower())
                c_keywords_list.append(KeywordInfo(keyword=w, score=round(s, 4), frequency=freq))

            c_timestamps = [e.timestamp for e in c_entries]
            c_time_dist = self._time_distribution(c_timestamps, bucket_type="minute")
            c_services = self._extract_services(c_entries)
            c_levels = self._extract_levels(c_entries)
            c_rep_indices = self._select_representatives(anomaly_messages, labels, cluster_id, n=self.config.output.terminal_max_examples)
            c_rep_messages = [anomaly_messages[i][:300] for i in c_rep_indices]

            mean_s = float(np.mean(c_scores))
            min_s = float(np.min(c_scores))
            max_s = float(np.max(c_scores))
            severity = self._compute_severity(len(indices), mean_s, min_s, c_levels)
            all_severity += severity * len(indices)

            info = ClusterInfo(
                cluster_id=cluster_id,
                size=len(indices),
                keywords=c_keywords_list,
                time_distribution=c_time_dist[:20],
                services=c_services,
                levels=c_levels,
                representative_messages=c_rep_messages,
                mean_score=round(mean_s, 4),
                min_score=round(min_s, 4),
                max_score=round(max_s, 4),
                severity_score=round(severity, 4),
                example_indices=[int(i) for i in c_rep_indices],
            )
            cluster_infos.append(info)

        cluster_infos.sort(key=lambda c: (-c.severity_score, -c.size))
        result.clusters = cluster_infos
        result.overall_severity_score = round(all_severity / max(len(anomalies), 1), 4)

        top3 = cluster_infos[:3]
        parts = []
        for c in top3:
            kws = ", ".join(kw.keyword for kw in c.keywords[:3])
            parts.append(f"Cluster#{c.cluster_id}(n={c.size}, sev={c.severity_score:.2f})[{kws}]")
        summary_parts = "; ".join(parts) if parts else "N/A"
        result.summary = (
            f"Detected {len(anomalies)} anomalies ({detection_result.anomaly_rate:.2%}) in {detection_result.total_entries} lines. "
            f"Found {len(cluster_infos)} clusters (noise={result.noise_count}). "
            f"Top issues: {summary_parts}."
        )

        extractor = TemplateExtractor()
        extractor.fit(detection_result.anomaly_results)
        result.templates = extractor.templates

        return result

    def get_cluster_messages(self, root_cause: RootCauseResult, cluster_id: int) -> List[str]:
        for c in root_cause.clusters:
            if c.cluster_id == cluster_id:
                return c.representative_messages
        return []
