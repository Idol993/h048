import glob
import os
import sys
import time
from datetime import timedelta
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from config import load_config, AppConfig
from parser import LogParser, LogEntry, FeatureResult
from detector import AnomalyDetector, DetectionResult
from rootcause import RootCauseExtractor, RootCauseResult
from reporter import Reporter


console = Console()


def _parse_window(value: str) -> Optional[int]:
    if not value:
        return None
    value = value.strip().lower()
    try:
        if value.endswith("m"):
            return int(float(value[:-1]))
        if value.endswith("h"):
            return int(float(value[:-1]) * 60)
        if value.endswith("s"):
            v = int(float(value[:-1]) / 60)
            return max(1, v)
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _expand_patterns(patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        matched = glob.glob(p, recursive=True)
        if matched:
            files.extend(matched)
        elif os.path.exists(p):
            files.append(p)
    return files


@click.group()
@click.version_option(version="1.0.0", prog_name="log-anomaly-detector")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False), help="Path to config.yaml")
@click.pass_context
def cli(ctx: click.Context, config_path: Optional[str]) -> None:
    """Log Anomaly Detector — AI-powered log analysis CLI for DevOps & SRE teams."""
    try:
        cfg = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise SystemExit(1)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["reporter"] = Reporter(cfg, console)


@cli.command("analyze")
@click.argument("log_paths", nargs=-1, required=True, type=click.Path())
@click.option("--output", "-o", "output_json", type=click.Path(dir_okay=False), help="Write JSON report to file")
@click.option("--threshold", "-t", type=float, default=None, help="Anomaly score threshold (default from config)")
@click.option("--window", "-w", "window_str", type=str, default=None, help="Time window size (e.g., 5m, 1h)")
@click.option("--streaming/--no-streaming", default=False, help="Force streaming mode for large files")
@click.option("--no-terminal", is_flag=True, help="Disable terminal output (JSON only)")
@click.option("--algorithm", "-a", type=click.Choice(["isolation_forest", "one_class_svm"]), default=None, help="Override detection algorithm")
@click.option("--top", type=int, default=None, help="Override max clusters shown in terminal")
@click.pass_context
def analyze_cmd(
    ctx: click.Context,
    log_paths: List[str],
    output_json: Optional[str],
    threshold: Optional[float],
    window_str: Optional[str],
    streaming: bool,
    no_terminal: bool,
    algorithm: Optional[str],
    top: Optional[int],
) -> None:
    """Analyze log files, detect anomalies, and extract root cause clues.

    LOG_PATHS supports glob patterns, e.g. /var/log/app/*.log or ./logs/**/*.json
    """
    cfg: AppConfig = ctx.obj["config"]
    reporter: Reporter = ctx.obj["reporter"]

    if algorithm:
        cfg.anomaly.algorithm = algorithm
    if threshold is None:
        threshold = cfg.anomaly.default_threshold
    if top:
        cfg.output.terminal_max_clusters = top

    window_minutes = _parse_window(window_str) if window_str else None

    files = _expand_patterns(list(log_paths))
    if not files:
        console.print(f"[yellow]No files matched patterns: {log_paths}[/yellow]")
        raise SystemExit(2)

    total_bytes = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    total_size_mb = total_bytes / (1024 * 1024)
    force_streaming = streaming or total_size_mb > 500

    console.print(Panel.fit(
        f"[bold cyan]Log Anomaly Detector[/bold cyan]\n"
        f"Files: [green]{len(files)}[/green]  |  "
        f"Total size: [green]{total_size_mb:.1f} MB[/green]  |  "
        f"Mode: [green]{'streaming' if force_streaming else 'in-memory'}[/green]  |  "
        f"Algorithm: [green]{cfg.anomaly.algorithm}[/green]  |  "
        f"Threshold: [green]{threshold}[/green]",
        border_style="cyan",
    ))

    t_start = time.time()

    parser = LogParser(cfg)
    detector = AnomalyDetector(cfg)
    extractor = RootCauseExtractor(cfg, parser)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Parsing log files", total=None)
        detection: Optional[DetectionResult] = None
        feature_matrix = None

        if force_streaming:
            progress.update(task, description="[1/3] Streaming & detecting anomalies")
            detection = detector.detect_streaming(parser, list(log_paths), threshold=threshold, window_minutes=window_minutes)
        else:
            all_entries: List[LogEntry] = []
            sample_lines: List[str] = []
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh):
                            if i < 20:
                                sample_lines.append(line)
                            all_entries.append(parser.parse_line(line, f, i + 1))
                except Exception as e:
                    console.print(f"[yellow]Warning: could not read {f}: {e}[/yellow]")
            total = len(all_entries)
            progress.update(task, completed=0, total=total, description=f"[1/3] Parsed {total} entries")

            if total == 0:
                console.print("[red]No log entries could be parsed.[/red]")
                raise SystemExit(3)

            fmt_name, fmt_conf = parser.detect_format(sample_lines)
            console.print(f"  Detected format: [bold]{fmt_name}[/bold] (confidence {fmt_conf:.0%})")

            progress.update(task, description="[2/3] Extracting TF-IDF features")
            feat_result: FeatureResult = parser.extract_features(all_entries)
            feature_matrix = parser.combine_features(feat_result.tfidf_matrix, feat_result.numeric_features)
            if feature_matrix is None:
                console.print("[red]Failed to extract feature matrix.[/red]")
                raise SystemExit(4)

            progress.update(task, description="[3/3] Running anomaly detection")
            detection = detector.detect(all_entries, feature_matrix, threshold=threshold, fit_on_data=True)

        if detection is None:
            console.print("[red]Detection failed.[/red]")
            raise SystemExit(5)

        progress.update(task, description="Extracting root cause clues")
        root_cause: RootCauseResult = extractor.extract(detection, feature_matrix=feature_matrix)

        elapsed = time.time() - t_start
        progress.update(task, description=f"Done in {elapsed:.1f}s ✓", completed=1, total=1)

    if not no_terminal:
        reporter.print_full_report(detection, root_cause)

    if output_json:
        try:
            reporter.to_json(detection, root_cause, output_json)
            console.print(f"[green]✓ JSON report written to: {output_json}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to write JSON report: {e}[/red]")

    anomalies = detection.anomaly_count
    total = detection.total_entries
    rate_text = f"{detection.anomaly_rate:.2%}" if total > 0 else "N/A"
    console.print()
    console.print(Panel.fit(
        f"[bold]Summary[/bold]: "
        f"[white]{total}[/white] entries, "
        f"[red]{anomalies}[/red] anomalies ({rate_text}), "
        f"[yellow]{len(root_cause.clusters)}[/yellow] clusters, "
        f"took [bold]{elapsed:.2f}s[/bold]",
        border_style="green",
    ))


@cli.command("detect-format")
@click.argument("log_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.pass_context
def detect_format_cmd(ctx: click.Context, log_paths: List[str]) -> None:
    """Auto-detect log file format(s) and print stats."""
    cfg: AppConfig = ctx.obj["config"]
    parser = LogParser(cfg)
    files = _expand_patterns(list(log_paths))
    table = Table(title="Log Format Detection", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("File", style="bold")
    table.add_column("Format", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Lines", justify="right")
    for fp in files:
        sample = []
        count = 0
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i < 50:
                        sample.append(line)
                    count += 1
        except Exception as e:
            console.print(f"[yellow]Warning reading {fp}: {e}[/yellow]")
            continue
        fmt, conf = parser.detect_format(sample)
        color = {"json": "cyan", "logfmt": "magenta", "plain": "green", "unknown": "yellow"}.get(fmt, "white")
        table.add_row(fp, f"[{color}]{fmt}[/{color}]", f"{conf:.0%}", f"{count:,}")
    console.print(table)


@cli.command("inspect")
@click.argument("log_path", nargs=1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--limit", "-n", type=int, default=20, help="First N lines to parse")
@click.option("--raw/--no-raw", default=False, help="Also show raw line")
@click.pass_context
def inspect_cmd(ctx: click.Context, log_path: str, limit: int, raw: bool) -> None:
    """Parse and display the first N log lines with parsed fields."""
    cfg: AppConfig = ctx.obj["config"]
    parser = LogParser(cfg)
    entries = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                entries.append((i + 1, line.rstrip("\n"), parser.parse_line(line, log_path, i + 1)))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    table = Table(title=f"Parsed Log Entries ({len(entries)} lines)", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Level", justify="center")
    table.add_column("Service", style="yellow")
    table.add_column("Message", style="white", overflow="fold")
    if raw:
        table.add_column("Raw", style="dim", overflow="fold")
    for ln, raw_line, e in entries:
        ts_str = e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else "-"
        lvl_colors = {
            "ERROR": "red", "FATAL": "red", "CRITICAL": "red",
            "WARN": "yellow", "WARNING": "yellow",
            "INFO": "blue", "NOTICE": "blue",
            "DEBUG": "dim", "TRACE": "dim",
        }
        lvl_color = lvl_colors.get(e.level, "white")
        lvl_cell = f"[{lvl_color}]{e.level}[/{lvl_color}]"
        msg_short = (e.message[:120] + "...") if len(e.message) > 120 else e.message
        row = [str(ln), ts_str, lvl_cell, e.service or "-", msg_short]
        if raw:
            row.append((raw_line[:80] + "...") if len(raw_line) > 80 else raw_line)
        table.add_row(*row)
    console.print(table)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
