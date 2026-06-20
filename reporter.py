import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from config import AppConfig
from detector import DetectionResult, AnomalyResult, TimeWindowBurst
from rootcause import RootCauseResult, ClusterInfo, KeywordInfo, TimeBucket
from parser import LogEntry


class Reporter:
    def __init__(self, config: AppConfig, console: Optional[Console] = None):
        self.config = config
        self.console = console or Console()

    @staticmethod
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def to_json(
        self,
        detection_result: DetectionResult,
        root_cause: RootCauseResult,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_entries": detection_result.total_entries,
                "anomaly_count": detection_result.anomaly_count,
                "anomaly_rate": round(detection_result.anomaly_rate, 6),
                "threshold": detection_result.threshold,
                "algorithm": self.config.anomaly.algorithm,
            },
            "summary": {
                "clusters_count": len(root_cause.clusters),
                "noise_count": root_cause.noise_count,
                "overall_severity_score": root_cause.overall_severity_score,
                "summary_text": root_cause.summary,
                "global_keywords": [
                    {"keyword": k.keyword, "score": k.score, "frequency": k.frequency}
                    for k in root_cause.global_keywords
                ],
                "top_services": [{"service": s, "count": c} for s, c in root_cause.top_services],
            },
            "clusters": [],
            "burst_windows": [],
        }
        for c in root_cause.clusters:
            cluster_payload = {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "severity_score": c.severity_score,
                "mean_score": c.mean_score,
                "min_score": c.min_score,
                "max_score": c.max_score,
                "keywords": [
                    {"keyword": k.keyword, "score": k.score, "frequency": k.frequency}
                    for k in c.keywords
                ],
                "time_distribution": [
                    {"bucket": t.bucket, "count": t.count, "anomaly_count": t.anomaly_count}
                    for t in c.time_distribution
                ],
                "services": c.services,
                "levels": c.levels,
            }
            if self.config.output.json_include_examples:
                cluster_payload["representative_messages"] = c.representative_messages
                cluster_payload["example_indices"] = c.example_indices
            payload["clusters"].append(cluster_payload)
        for bw in detection_result.burst_windows:
            payload["burst_windows"].append({
                "window_start": bw.window_start.isoformat() if bw.window_start else None,
                "window_end": bw.window_end.isoformat() if bw.window_end else None,
                "anomaly_count": bw.anomaly_count,
                "total_count": bw.total_count,
                "anomaly_rate": round(bw.anomaly_rate, 6),
                "is_burst": bw.is_burst,
                "z_score": round(bw.z_score, 4),
            })
        if self.config.output.json_include_examples:
            anomalies = [a for a in detection_result.anomaly_results if a.is_anomaly]
            payload["anomaly_samples"] = [
                {
                    "timestamp": a.entry.timestamp.isoformat() if a.entry.timestamp else None,
                    "level": a.entry.level,
                    "message": a.entry.message,
                    "service": a.entry.service,
                    "score": round(a.score, 6),
                    "source_file": a.entry.source_file,
                    "line_number": a.entry.line_number,
                }
                for a in anomalies[:500]
            ]
        if output_path:
            indent = 2 if self.config.output.json_pretty else None
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=indent, default=self._serialize)
        return payload

    def print_summary(self, detection_result: DetectionResult, root_cause: RootCauseResult) -> None:
        dr = detection_result
        rc = root_cause
        title = Text("=== LOG ANOMALY ANALYSIS REPORT ===", style="bold cyan")
        self.console.print(Panel(title, border_style="cyan"))

        meta_table = Table(title="Overview", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        meta_table.add_column("Metric", style="bold")
        meta_table.add_column("Value", justify="right")
        meta_table.add_row("Total log entries", f"{dr.total_entries:,}")
        meta_table.add_row("Anomalies detected", f"[red]{dr.anomaly_count:,}[/red]")
        meta_table.add_row("Anomaly rate", f"[red]{dr.anomaly_rate:.2%}[/red]")
        meta_table.add_row("Threshold", f"{dr.threshold}")
        meta_table.add_row("Clusters found", f"[yellow]{len(rc.clusters)}[/yellow]")
        meta_table.add_row("Noise entries", f"{rc.noise_count}")
        meta_table.add_row("Overall severity", f"[bold]{rc.overall_severity_score:.4f}[/bold]")
        meta_table.add_row("Burst windows", f"[orange1]{sum(1 for b in dr.burst_windows if b.is_burst)}[/orange1]")
        self.console.print(meta_table)

        self.console.print()
        self.console.print(Panel(Text(rc.summary, style="white"), title="Summary", border_style="green"))

    def print_global_keywords(self, root_cause: RootCauseResult) -> None:
        if not root_cause.global_keywords:
            return
        table = Table(
            title=f"Top Global Keywords (Top {self.config.output.terminal_max_keywords})",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Keyword", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Frequency", justify="right")
        kw_list = root_cause.global_keywords[: self.config.output.terminal_max_keywords]
        for i, kw in enumerate(kw_list, 1):
            color = "red" if kw.score >= 2.0 else "yellow" if kw.score >= 1.0 else "white"
            table.add_row(
                str(i),
                f"[{color}]{kw.keyword}[/{color}]",
                f"{kw.score:.4f}",
                str(kw.frequency),
            )
        self.console.print(table)

    def print_top_services(self, root_cause: RootCauseResult) -> None:
        if not root_cause.top_services:
            return
        table = Table(title="Affected Services", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Service", style="bold")
        table.add_column("Anomaly Count", justify="right")
        table.add_column("Share", justify="right")
        total = sum(c for _, c in root_cause.top_services) or 1
        for i, (svc, count) in enumerate(root_cause.top_services[:10], 1):
            pct = count / total * 100
            color = "red" if pct >= 30 else "yellow" if pct >= 15 else "white"
            table.add_row(str(i), f"[{color}]{svc}[/{color}]", str(count), f"{pct:.1f}%")
        self.console.print(table)

    def print_burst_windows(self, detection_result: DetectionResult) -> None:
        bursts = [b for b in detection_result.burst_windows if b.is_burst]
        if not bursts:
            return
        table = Table(title="Burst Windows (Anomaly Spikes)", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Time Window", style="bold")
        table.add_column("Anomalies", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Rate", justify="right")
        table.add_column("Z-Score", justify="right")
        for b in bursts[:10]:
            start = b.window_start.strftime("%H:%M")
            end = b.window_end.strftime("%H:%M")
            rate_color = "red" if b.anomaly_rate >= 0.3 else "yellow"
            table.add_row(
                f"{start} → {end}",
                f"[orange1]{b.anomaly_count}[/orange1]",
                str(b.total_count),
                f"[{rate_color}]{b.anomaly_rate:.1%}[/{rate_color}]",
                f"[bold]{b.z_score:.2f}σ[/bold]",
            )
        self.console.print(table)

    def print_clusters(self, root_cause: RootCauseResult) -> None:
        if not root_cause.clusters:
            self.console.print("[yellow]No clusters found (all anomalies are noise).[/yellow]")
            return
        max_c = self.config.output.terminal_max_clusters
        clusters = root_cause.clusters[:max_c]

        summary_table = Table(
            title=f"Anomaly Clusters (Top {len(clusters)})",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        summary_table.add_column("#", justify="right", style="dim")
        summary_table.add_column("ID", justify="right")
        summary_table.add_column("Size", justify="right")
        summary_table.add_column("Severity", justify="right")
        summary_table.add_column("Scores (min/mean/max)")
        summary_table.add_column("Top Keywords")
        for i, c in enumerate(clusters, 1):
            sev_color = "red" if c.severity_score >= 0.7 else "yellow" if c.severity_score >= 0.4 else "white"
            top_kw = ", ".join(kw.keyword for kw in c.keywords[:3])
            summary_table.add_row(
                str(i),
                f"[bold]#{c.cluster_id}[/bold]",
                f"[cyan]{c.size}[/cyan]",
                f"[{sev_color}]{c.severity_score:.3f}[/{sev_color}]",
                f"{c.min_score:.2f}/{c.mean_score:.2f}/{c.max_score:.2f}",
                top_kw,
            )
        self.console.print(summary_table)

        max_ex = self.config.output.terminal_max_examples
        for i, c in enumerate(clusters, 1):
            self.console.print()
            sev_color = "red" if c.severity_score >= 0.7 else "yellow" if c.severity_score >= 0.4 else "green"
            header = Text.assemble(
                (f"Cluster #{c.cluster_id} ", "bold"),
                (f"[size={c.size}, severity={c.severity_score:.3f}] ", sev_color),
            )
            panel_content_parts = []

            if c.keywords:
                kw_line = ", ".join(f"[bold]{kw.keyword}[/bold]({kw.score:.2f})" for kw in c.keywords[:8])
                panel_content_parts.extend([("Keywords: ", "bold"), (kw_line, "")])

            if c.services:
                svc_line = ", ".join(f"{s}({n})" for s, n in list(c.services.items())[:5])
                panel_content_parts.extend([("\nServices: ", "bold"), (svc_line, "")])

            if c.levels:
                lvl_line = ", ".join(f"[{'red' if l in ('ERROR','FATAL','CRITICAL') else 'yellow' if l in ('WARN','WARNING') else 'blue'}]{l}[/]({n})" for l, n in c.levels.items())
                panel_content_parts.extend([("\nLevels: ", "bold"), (lvl_line, "")])

            if c.time_distribution:
                top_times = c.time_distribution[:5]
                time_line = ", ".join(f"{t.bucket.split(' ')[-1]}({t.count})" for t in top_times)
                panel_content_parts.extend([("\nTime buckets: ", "bold"), (time_line, "")])

            if c.representative_messages:
                panel_content_parts.append(("\n\nExample messages:", "bold underline"))
                for j, msg in enumerate(c.representative_messages[:max_ex]):
                    short_msg = msg[:200] + ("..." if len(msg) > 200 else "")
                    panel_content_parts.extend([(f"\n  {j+1}. ", "dim"), (short_msg, "white")])

            content = Text.assemble(*panel_content_parts)
            self.console.print(Panel(content, title=f"Cluster #{c.cluster_id} — Details", border_style=sev_color))

    def print_full_report(self, detection_result: DetectionResult, root_cause: RootCauseResult) -> None:
        self.print_summary(detection_result, root_cause)
        self.console.print()
        self.print_global_keywords(root_cause)
        self.console.print()
        self.print_top_services(root_cause)
        self.console.print()
        self.print_burst_windows(detection_result)
        self.console.print()
        self.print_clusters(root_cause)
        self.console.print()
        self.console.print(Text("Tip: check the generated JSON file for complete raw data.", style="dim italic"))
