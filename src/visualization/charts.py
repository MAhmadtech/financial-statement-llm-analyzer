"""Phase 7: visualization & analytics.

Generates per-strategy comparison charts (bar, line, box) for accuracy,
completeness, hallucination, response time, and user rating, and exports
PNG + PDF + CSV into outputs/charts/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: charts are files, not windows
import matplotlib.pyplot as plt
import pandas as pd

from config import settings
from config.constants import STRATEGY_LABELS

METRICS = [
    ("fact_accuracy", "Fact Extraction Accuracy", True),
    ("completeness", "Completeness", True),
    ("hallucination_rate", "Hallucination Rate", False),  # lower is better
    ("response_time", "Response Time (s)", False),
    ("user_rating", "User Rating (1-5)", True),
]

STRATEGY_ORDER = list(STRATEGY_LABELS.keys())


def _labels(strategies: list[str]) -> list[str]:
    return [STRATEGY_LABELS.get(s, s) for s in strategies]


def _save(fig: plt.Figure, name: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def results_dataframe(results: list[dict]) -> pd.DataFrame:
    """Successful experiments only, with a stable strategy ordering."""
    df = pd.DataFrame(results)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["strategy"] = pd.Categorical(
        df["strategy"], categories=STRATEGY_ORDER, ordered=True
    )
    return df.sort_values(["strategy", "statement_id"])


def bar_chart_means(df: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """One bar chart per metric: mean value per strategy."""
    out_dir = out_dir or settings.CHARTS_DIR
    saved = []
    for metric, title, _higher_better in METRICS:
        if metric not in df or df[metric].dropna().empty:
            continue
        means = df.groupby("strategy", observed=False)[metric].mean()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(_labels(list(means.index)), means.values, color="#4472C4")
        ax.set_title(f"{title} by Prompt Strategy (mean)")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(means.values):
            if pd.notna(v):
                ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom")
        saved += _save(fig, f"bar_{metric}", out_dir)
    return saved


def line_chart_by_statement(
    df: pd.DataFrame, metric: str = "fact_accuracy", out_dir: Path | None = None
) -> list[Path]:
    """Metric per statement, one line per strategy."""
    out_dir = out_dir or settings.CHARTS_DIR
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for strategy, group in df.groupby("strategy", observed=False):
        group = group.sort_values("statement_id")
        if group[metric].dropna().empty:
            continue
        ax.plot(
            group["statement_id"], group[metric],
            marker="o", label=STRATEGY_LABELS.get(str(strategy), str(strategy)),
        )
    ax.set_title(f"{metric} across statements")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=75)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, f"line_{metric}", out_dir)


def box_plot(df: pd.DataFrame, metric: str = "response_time",
             out_dir: Path | None = None) -> list[Path]:
    """Distribution of a metric per strategy."""
    out_dir = out_dir or settings.CHARTS_DIR
    groups, labels = [], []
    for strategy in STRATEGY_ORDER:
        values = df[df["strategy"] == strategy][metric].dropna()
        if len(values):
            groups.append(values)
            labels.append(STRATEGY_LABELS[strategy])
    if not groups:
        return []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_title(f"{metric} distribution by strategy")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, f"box_{metric}", out_dir)


def export_summary_csv(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    """Mean of every metric per strategy -> CSV (the report's main table)."""
    out_dir = out_dir or settings.CHARTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [m for m, _, _ in METRICS if m in df]
    summary = df.groupby("strategy", observed=False)[metric_cols].agg(
        ["mean", "std", "count"]
    ).round(4)
    path = out_dir / "strategy_summary.csv"
    summary.to_csv(path, encoding="utf-8")
    return path


def generate_all_charts(results: list[dict], out_dir: Path | None = None) -> list[Path]:
    """Everything Phase 7 promises: bar/line/box + CSV export."""
    out_dir = out_dir or settings.CHARTS_DIR
    df = results_dataframe(results)
    saved = bar_chart_means(df, out_dir)
    saved += line_chart_by_statement(df, "fact_accuracy", out_dir)
    saved += line_chart_by_statement(df, "completeness", out_dir)
    saved += box_plot(df, "response_time", out_dir)
    saved.append(export_summary_csv(df, out_dir))
    return saved
