import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LogFormatConfig:
    plain_patterns: List[Dict[str, str]] = field(default_factory=list)
    json_level_fields: List[str] = field(default_factory=list)
    json_message_fields: List[str] = field(default_factory=list)
    json_timestamp_fields: List[str] = field(default_factory=list)
    json_service_fields: List[str] = field(default_factory=list)
    logfmt_pattern: str = ""
    logfmt_level_keys: List[str] = field(default_factory=list)
    logfmt_message_keys: List[str] = field(default_factory=list)
    logfmt_timestamp_keys: List[str] = field(default_factory=list)


@dataclass
class AnomalyConfig:
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    default_threshold: float = -0.5
    n_estimators: int = 100
    max_samples: str = "auto"
    random_state: int = 42


@dataclass
class ClusteringConfig:
    algorithm: str = "dbscan"
    eps: float = 0.5
    min_samples: int = 3
    metric: str = "cosine"


@dataclass
class FeatureConfig:
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    stop_words: str = "english"


@dataclass
class TimeWindowConfig:
    default_window_minutes: int = 5
    burst_threshold_std: float = 2.0


@dataclass
class OutputConfig:
    terminal_max_keywords: int = 10
    terminal_max_clusters: int = 10
    terminal_max_examples: int = 5
    json_include_examples: bool = True
    json_include_features: bool = False
    json_pretty: bool = True


@dataclass
class StreamingConfig:
    chunksize: int = 10000
    n_jobs: int = -1


@dataclass
class AppConfig:
    log_formats: LogFormatConfig = field(default_factory=LogFormatConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    time_window: TimeWindowConfig = field(default_factory=TimeWindowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    raw = _load_yaml(config_path)

    lf = raw.get("log_formats", {}) or {}
    json_cfg = lf.get("json", {}) or {}
    logfmt_cfg = lf.get("logfmt", {}) or {}

    ngram_raw = (raw.get("feature_extraction", {}) or {}).get("tfidf_ngram_range", [1, 2])
    ngram_range = (int(ngram_raw[0]), int(ngram_raw[1])) if isinstance(ngram_raw, list) and len(ngram_raw) >= 2 else (1, 2)

    log_formats = LogFormatConfig(
        plain_patterns=lf.get("plain", []) or [],
        json_level_fields=json_cfg.get("level_fields", []),
        json_message_fields=json_cfg.get("message_fields", []),
        json_timestamp_fields=json_cfg.get("timestamp_fields", []),
        json_service_fields=json_cfg.get("service_fields", []),
        logfmt_pattern=logfmt_cfg.get("pattern", ""),
        logfmt_level_keys=logfmt_cfg.get("level_keys", []),
        logfmt_message_keys=logfmt_cfg.get("message_keys", []),
        logfmt_timestamp_keys=logfmt_cfg.get("timestamp_keys", []),
    )

    anom = raw.get("anomaly_detection", {}) or {}
    anomaly = AnomalyConfig(
        algorithm=anom.get("algorithm", "isolation_forest"),
        contamination=float(anom.get("contamination", 0.1)),
        default_threshold=float(anom.get("default_threshold", -0.5)),
        n_estimators=int(anom.get("n_estimators", 100)),
        max_samples=str(anom.get("max_samples", "auto")),
        random_state=int(anom.get("random_state", 42)),
    )

    clust = raw.get("clustering", {}) or {}
    clustering = ClusteringConfig(
        algorithm=clust.get("algorithm", "dbscan"),
        eps=float(clust.get("eps", 0.5)),
        min_samples=int(clust.get("min_samples", 3)),
        metric=clust.get("metric", "cosine"),
    )

    feat = raw.get("feature_extraction", {}) or {}
    feature = FeatureConfig(
        tfidf_max_features=int(feat.get("tfidf_max_features", 5000)),
        tfidf_ngram_range=ngram_range,
        tfidf_min_df=int(feat.get("tfidf_min_df", 2)),
        tfidf_max_df=float(feat.get("tfidf_max_df", 0.95)),
        stop_words=feat.get("stop_words", "english"),
    )

    tw = raw.get("time_window", {}) or {}
    time_window = TimeWindowConfig(
        default_window_minutes=int(tw.get("default_window_minutes", 5)),
        burst_threshold_std=float(tw.get("burst_threshold_std", 2.0)),
    )

    out = raw.get("output", {}) or {}
    term_out = out.get("terminal", {}) or {}
    json_out = out.get("json", {}) or {}
    output = OutputConfig(
        terminal_max_keywords=int(term_out.get("max_keywords", 10)),
        terminal_max_clusters=int(term_out.get("max_clusters", 10)),
        terminal_max_examples=int(term_out.get("max_examples", 5)),
        json_include_examples=bool(json_out.get("include_examples", True)),
        json_include_features=bool(json_out.get("include_features", False)),
        json_pretty=bool(json_out.get("pretty", True)),
    )

    stream = raw.get("streaming", {}) or {}
    streaming = StreamingConfig(
        chunksize=int(stream.get("chunksize", 10000)),
        n_jobs=int(stream.get("n_jobs", -1)),
    )

    return AppConfig(
        log_formats=log_formats,
        anomaly=anomaly,
        clustering=clustering,
        feature=feature,
        time_window=time_window,
        output=output,
        streaming=streaming,
        raw=raw,
    )
