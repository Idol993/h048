from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from joblib import Parallel, delayed, dump as joblib_dump, load as joblib_load

from config import AppConfig
from parser import LogEntry, FeatureResult, LogParser

MODEL_VERSION = "1.0.0"


@dataclass
class TrainedModel:
    model: Any
    vectorizer: Any
    config: Dict[str, Any]
    metadata: Dict[str, Any]

    def save(self, path: str) -> None:
        package = {
            "model_version": MODEL_VERSION,
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "vectorizer": self.vectorizer,
            "config": self.config,
            "metadata": self.metadata,
        }
        joblib_dump(package, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "TrainedModel":
        package = joblib_load(path)
        return cls(
            model=package["model"],
            vectorizer=package["vectorizer"],
            config=package["config"],
            metadata=package.get("metadata", {}),
        )


@dataclass
class AnomalyResult:
    entry: LogEntry
    score: float
    is_anomaly: bool
    cluster_id: Optional[int] = None


@dataclass
class TimeWindowBurst:
    window_start: datetime
    window_end: datetime
    anomaly_count: int
    total_count: int
    anomaly_rate: float
    is_burst: bool
    z_score: float


@dataclass
class DetectionResult:
    anomaly_results: List[AnomalyResult] = field(default_factory=list)
    model: Optional[Any] = None
    threshold: float = -0.5
    threshold_type: str = "auto"
    burst_windows: List[TimeWindowBurst] = field(default_factory=list)
    total_entries: int = 0
    anomaly_count: int = 0
    anomaly_rate: float = 0.0
    model_info: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ac = config.anomaly
        self.tw = config.time_window
        self.model: Optional[Any] = None
        self.model_info: Dict[str, Any] = {}
        self.vectorizer: Optional[Any] = None

    def load_model(self, trained_model: TrainedModel) -> None:
        self.model = trained_model.model
        self.vectorizer = trained_model.vectorizer
        self.model_info = {
            "model_version": trained_model.metadata.get("model_version", MODEL_VERSION),
            "created_at": trained_model.metadata.get("created_at", ""),
            "training_entries": trained_model.metadata.get("total_entries", 0),
            "feature_count": trained_model.metadata.get("feature_count", 0),
            "algorithm": trained_model.metadata.get("algorithm", ""),
        }

    def save_model(self, vectorizer: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        meta = {
            "model_version": MODEL_VERSION,
            "created_at": datetime.now().isoformat(),
            "algorithm": self.ac.algorithm,
            "contamination": self.ac.contamination,
            "n_estimators": self.ac.n_estimators if self.ac.algorithm == "isolation_forest" else None,
            **(metadata or {}),
        }
        if vectorizer is not None:
            meta["feature_count"] = len(vectorizer.get_feature_names_out())
        cfg = {
            "algorithm": self.ac.algorithm,
            "contamination": self.ac.contamination,
            "default_threshold": self.ac.default_threshold,
        }
        trained = TrainedModel(model=self.model, vectorizer=vectorizer, config=cfg, metadata=meta)
        trained.save(path)

    def _create_model(self):
        if self.ac.algorithm.lower() == "one_class_svm":
            return OneClassSVM(
                nu=self.ac.contamination,
                kernel="rbf",
                gamma="scale",
            )
        max_samples = self.ac.max_samples
        if isinstance(max_samples, str) and max_samples.lower() == "auto":
            max_samples_val = "auto"
        else:
            try:
                max_samples_val = int(max_samples)
            except (ValueError, TypeError):
                max_samples_val = "auto"
        return IsolationForest(
            n_estimators=self.ac.n_estimators,
            max_samples=max_samples_val,
            contamination=self.ac.contamination,
            random_state=self.ac.random_state,
            n_jobs=self.config.streaming.n_jobs if self.config.streaming.n_jobs > 0 else None,
        )

    def fit(self, feature_matrix: csr_matrix):
        self.model = self._create_model()
        if feature_matrix.shape[0] == 0:
            return self.model
        dense = feature_matrix.toarray() if hasattr(feature_matrix, "toarray") else np.asarray(feature_matrix)
        self.model.fit(dense)
        return self.model

    def decision_function(self, feature_matrix: csr_matrix) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        dense = feature_matrix.toarray() if hasattr(feature_matrix, "toarray") else np.asarray(feature_matrix)
        if dense.shape[0] == 0:
            return np.array([])
        return self.model.decision_function(dense)

    def predict_scores(self, feature_matrix: csr_matrix, fit_on_data: bool = True) -> np.ndarray:
        if fit_on_data:
            self.fit(feature_matrix)
        return self.decision_function(feature_matrix)

    def detect(
        self,
        entries: List[LogEntry],
        feature_matrix: csr_matrix,
        threshold: Optional[float] = None,
        manual_threshold: bool = False,
        fit_on_data: bool = True,
    ) -> DetectionResult:
        result = DetectionResult()
        result.total_entries = len(entries)
        if not entries or feature_matrix.shape[0] == 0:
            return result

        if manual_threshold and threshold is not None:
            result.threshold = threshold
            result.threshold_type = "manual"
        else:
            result.threshold = threshold if threshold is not None else self.ac.default_threshold
            result.threshold_type = "auto"

        scores = self.predict_scores(feature_matrix, fit_on_data=fit_on_data)
        result.model = self.model

        if result.threshold_type == "manual":
            anomaly_flags = scores < result.threshold
        else:
            anomaly_flags = scores < result.threshold
            if np.sum(anomaly_flags) == 0 and len(scores) > 0:
                auto_threshold = np.percentile(scores, self.ac.contamination * 100)
                result.threshold = float(auto_threshold)
                anomaly_flags = scores <= result.threshold

        for i, entry in enumerate(entries):
            is_anom = bool(anomaly_flags[i]) if i < len(anomaly_flags) else False
            score = float(scores[i]) if i < len(scores) else 0.0
            result.anomaly_results.append(
                AnomalyResult(entry=entry, score=score, is_anomaly=is_anom)
            )
        result.anomaly_count = int(np.sum(anomaly_flags)) if len(anomaly_flags) > 0 else 0
        result.anomaly_rate = result.anomaly_count / result.total_entries if result.total_entries > 0 else 0.0
        result.burst_windows = self._detect_time_bursts(result.anomaly_results)
        return result

    def detect_streaming(
        self,
        parser: LogParser,
        file_patterns: List[str],
        threshold: Optional[float] = None,
        manual_threshold: bool = False,
        window_minutes: Optional[int] = None,
        progress_callback: Optional[Any] = None,
        vectorizer: Optional[Any] = None,
        fit_model: bool = True,
    ) -> DetectionResult:
        all_results: List[AnomalyResult] = []
        total = 0
        total_anomalies = 0
        is_first_chunk = True
        for chunk_entries in parser.parse_files_streaming(file_patterns):
            if not chunk_entries:
                continue
            total += len(chunk_entries)
            feat = parser.extract_features(chunk_entries, vectorizer=vectorizer)
            combined = parser.combine_features(feat.tfidf_matrix, feat.numeric_features)
            if combined is None or combined.shape[0] == 0:
                for e in chunk_entries:
                    all_results.append(AnomalyResult(entry=e, score=0.0, is_anomaly=False))
                continue
            if is_first_chunk and fit_model:
                chunk_scores = self.predict_scores(combined, fit_on_data=(self.model is None))
                is_first_chunk = False
            else:
                chunk_scores = self.predict_scores(combined, fit_on_data=False)

            if manual_threshold and threshold is not None:
                th = threshold
                flags = chunk_scores < th
            else:
                th = threshold if threshold is not None else self.ac.default_threshold
                flags = chunk_scores < th

            chunk_anom = int(np.sum(flags))
            total_anomalies += chunk_anom

            for i, e in enumerate(chunk_entries):
                s = float(chunk_scores[i]) if i < len(chunk_scores) else 0.0
                is_a = bool(flags[i]) if i < len(flags) else False
                all_results.append(AnomalyResult(entry=e, score=s, is_anomaly=is_a))

            if progress_callback is not None:
                try:
                    progress_callback(total, total_anomalies)
                except Exception:
                    pass

        detection = DetectionResult()
        detection.anomaly_results = all_results
        detection.total_entries = total
        detection.anomaly_count = total_anomalies
        detection.anomaly_rate = total_anomalies / total if total > 0 else 0.0
        detection.model = self.model

        if manual_threshold and threshold is not None:
            detection.threshold = threshold
            detection.threshold_type = "manual"
        else:
            if total_anomalies == 0 and len(all_results) > 0:
                all_scores = np.array([a.score for a in all_results])
                if len(all_scores) > 0:
                    auto_th = float(np.percentile(all_scores, self.ac.contamination * 100))
                    detection.threshold = auto_th
                    for a in detection.anomaly_results:
                        a.is_anomaly = a.score <= auto_th
                    detection.anomaly_count = sum(1 for a in detection.anomaly_results if a.is_anomaly)
                    detection.anomaly_rate = detection.anomaly_count / total if total > 0 else 0.0
                    detection.threshold_type = "auto"
                else:
                    detection.threshold = threshold if threshold is not None else self.ac.default_threshold
                    detection.threshold_type = "auto"
            else:
                detection.threshold = threshold if threshold is not None else self.ac.default_threshold
                detection.threshold_type = "manual" if manual_threshold else "auto"

        detection.burst_windows = self._detect_time_bursts(all_results, window_minutes=window_minutes)
        return detection

    @staticmethod
    def _round_time(dt: datetime, minutes: int) -> datetime:
        total_minutes = dt.hour * 60 + dt.minute
        rounded = (total_minutes // minutes) * minutes
        return dt.replace(hour=rounded // 60, minute=rounded % 60, second=0, microsecond=0)

    def _detect_time_bursts(
        self,
        anomaly_results: List[AnomalyResult],
        window_minutes: Optional[int] = None,
    ) -> List[TimeWindowBurst]:
        if not anomaly_results:
            return []
        wm = window_minutes or self.tw.default_window_minutes
        entries_with_ts = [
            (a.entry.timestamp, a.is_anomaly)
            for a in anomaly_results
            if a.entry.timestamp is not None
        ]
        if not entries_with_ts:
            return []
        entries_with_ts.sort(key=lambda x: x[0])
        windows: Dict[datetime, Dict[str, int]] = {}
        for ts, is_anom in entries_with_ts:
            key = self._round_time(ts, wm)
            if key not in windows:
                windows[key] = {"total": 0, "anomaly": 0}
            windows[key]["total"] += 1
            if is_anom:
                windows[key]["anomaly"] += 1
        if not windows:
            return []
        sorted_keys = sorted(windows.keys())
        counts = np.array([windows[k]["anomaly"] for k in sorted_keys], dtype=float)
        if len(counts) < 2:
            mean_c, std_c = counts.mean(), 1.0
        else:
            mean_c = counts.mean()
            std_c = counts.std() or 1.0
        burst_list = []
        for k in sorted_keys:
            start = k
            end = start + timedelta(minutes=wm)
            total = windows[k]["total"]
            anom = windows[k]["anomaly"]
            rate = anom / total if total > 0 else 0.0
            z = (anom - mean_c) / std_c
            burst_list.append(TimeWindowBurst(
                window_start=start,
                window_end=end,
                anomaly_count=anom,
                total_count=total,
                anomaly_rate=rate,
                is_burst=z >= self.tw.burst_threshold_std,
                z_score=float(z),
            ))
        return burst_list

    def get_anomaly_entries(self, result: DetectionResult) -> List[AnomalyResult]:
        return [a for a in result.anomaly_results if a.is_anomaly]

    def get_entries_dataframe(self, result: DetectionResult) -> pd.DataFrame:
        rows = []
        for a in result.anomaly_results:
            e = a.entry
            rows.append({
                "timestamp": e.timestamp,
                "level": e.level,
                "level_value": e.level_value,
                "message": e.message,
                "service": e.service,
                "source_file": e.source_file,
                "line_number": e.line_number,
                "score": a.score,
                "is_anomaly": a.is_anomaly,
                "cluster_id": a.cluster_id,
            })
        return pd.DataFrame(rows)
