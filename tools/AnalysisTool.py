from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from langchain_core.tools import tool

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "output" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_repo_path(path: str) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (REPO_ROOT / raw)
    candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT)
    except Exception as e:
        raise ValueError("access denied: path outside repository root") from e

    return candidate


def _save_fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _write_fig_file(fig, filename: Optional[str] = None) -> Path:
    if filename is None:
        filename = f"plot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = OUT_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


@tool(response_format="content")
def summarize_csv(path: str, max_rows: int = 5) -> str:
    """Read a CSV file and return a short summary: head and describe()."""
    if pd is None:
        return "Error: pandas is not installed in the environment."

    try:
        csv_path = _resolve_repo_path(path)
    except Exception as e:
        return json.dumps({"error": str(e)})

    if not csv_path.exists():
        return json.dumps({"error": f"file not found: {path}"})

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return json.dumps({"error": f"failed to read CSV: {e}"})

    head = df.head(max(1, max_rows)).to_dict(orient="records")
    try:
        desc = df.describe(include="all").fillna("").to_dict()
    except Exception:
        desc = {}

    return json.dumps(
        {
            "path": str(csv_path.relative_to(REPO_ROOT)),
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns.tolist()],
            "head": head,
            "describe": desc,
        },
        default=str,
        ensure_ascii=False,
    )


@tool(response_format="content")
def describe_stats(path: str, column: Optional[str] = None) -> str:
    """Return descriptive statistics for a CSV or a specific column."""
    if pd is None:
        return "Error: pandas is not installed in the environment."

    try:
        csv_path = _resolve_repo_path(path)
    except Exception as e:
        return json.dumps({"error": str(e)})

    if not csv_path.exists():
        return json.dumps({"error": f"file not found: {path}"})

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return json.dumps({"error": f"failed to read CSV: {e}"})

    if column:
        if column not in df.columns:
            return json.dumps(
                {
                    "error": f"column not found: {column}",
                    "available_columns": [str(c) for c in df.columns.tolist()],
                },
                ensure_ascii=False,
            )

        series = df[column]
        stats = series.describe().to_dict()
        stats.update({"missing": int(series.isna().sum()), "dtype": str(series.dtype)})
        return json.dumps(
            {"path": str(csv_path.relative_to(REPO_ROOT)), "column": column, "stats": stats},
            default=str,
            ensure_ascii=False,
        )

    try:
        desc = df.describe(include="all").fillna("").to_dict()
    except Exception:
        desc = {}

    return json.dumps(
        {"path": str(csv_path.relative_to(REPO_ROOT)), "describe": desc},
        default=str,
        ensure_ascii=False,
    )


@tool(response_format="content_and_artifact")
def plot_histogram(
    path: str,
    column: str,
    bins: int = 10,
    output_filename: Optional[str] = None,
) -> Tuple[str, str]:
    """Plot a histogram for one numeric column from CSV."""
    if pd is None:
        return "Error: pandas is not installed.", ""
    if plt is None:
        return "Error: matplotlib is not installed.", ""

    try:
        csv_path = _resolve_repo_path(path)
    except Exception as e:
        return f"Error: {e}", ""

    if not csv_path.exists():
        return f"Error: file not found: {path}", ""

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}", ""

    if column not in df.columns:
        return f"Error: column not found: {column}", ""

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = pd.to_numeric(df[column], errors="coerce").dropna()
        if data.empty:
            return f"Error: no numeric values in column: {column}", ""

        ax.hist(data, bins=max(1, bins), color="#4C72B0", edgecolor="white")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.set_title(f"Histogram: {column}")

        out_path = _write_fig_file(fig, filename=output_filename)
        b64 = _save_fig_to_base64(fig)
        return (
            json.dumps(
                {
                    "message": "histogram created",
                    "file": str(out_path.relative_to(REPO_ROOT)),
                    "image_base64": b64,
                },
                ensure_ascii=False,
            ),
            str(out_path),
        )
    except Exception as e:
        return f"Error plotting histogram: {e}", ""
    finally:
        if fig is not None:
            plt.close(fig)


@tool(response_format="content_and_artifact")
def plot_correlation_matrix(
    path: str,
    columns: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Tuple[str, str]:
    """Plot a correlation matrix for numeric columns."""
    if pd is None:
        return "Error: pandas is not installed.", ""
    if plt is None:
        return "Error: matplotlib is not installed.", ""

    try:
        csv_path = _resolve_repo_path(path)
    except Exception as e:
        return f"Error: {e}", ""

    if not csv_path.exists():
        return f"Error: file not found: {path}", ""

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}", ""

    try:
        if columns:
            selected = [c.strip() for c in columns.split(",") if c.strip()]
            df = df[selected]
    except Exception as e:
        return f"Error selecting columns: {e}", ""

    numeric = df.select_dtypes(include=["number"])
    if numeric.empty or numeric.shape[1] < 2:
        return "Error: need at least two numeric columns for correlation.", ""

    fig = None
    try:
        corr = numeric.corr()
        size = min(12, max(4, 1 + len(corr) * 0.6))
        fig, ax = plt.subplots(figsize=(size, size))

        if sns is not None:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        else:
            cax = ax.matshow(corr, cmap="coolwarm")
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="left")
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)

        ax.set_title("Correlation matrix")
        out_path = _write_fig_file(fig, filename=output_filename)
        b64 = _save_fig_to_base64(fig)

        return (
            json.dumps(
                {
                    "message": "correlation matrix created",
                    "file": str(out_path.relative_to(REPO_ROOT)),
                    "image_base64": b64,
                },
                ensure_ascii=False,
            ),
            str(out_path),
        )
    except Exception as e:
        return f"Error plotting correlation matrix: {e}", ""
    finally:
        if fig is not None:
            plt.close(fig)


@tool(response_format="content_and_artifact")
def plot_time_series(
    path: str,
    date_column: str,
    value_column: str,
    date_format: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Tuple[str, str]:
    """Plot a time series for one value column by date column."""
    if pd is None:
        return "Error: pandas is not installed.", ""
    if plt is None:
        return "Error: matplotlib is not installed.", ""

    try:
        csv_path = _resolve_repo_path(path)
    except Exception as e:
        return f"Error: {e}", ""

    if not csv_path.exists():
        return f"Error: file not found: {path}", ""

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}", ""

    if date_column not in df.columns or value_column not in df.columns:
        return f"Error: missing columns. Available: {df.columns.tolist()}", ""

    fig = None
    try:
        if date_format:
            df[date_column] = pd.to_datetime(
                df[date_column], format=date_format, errors="coerce"
            )
        else:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        df = df.dropna(subset=[date_column, value_column]).sort_values(by=date_column)
        if df.empty:
            return "Error: no valid rows after datetime/value parsing.", ""

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            df[date_column],
            df[value_column],
            marker="o",
            linestyle="-",
            color="#4C72B0",
        )
        ax.set_xlabel(date_column)
        ax.set_ylabel(value_column)
        ax.set_title(f"Time series: {value_column} over {date_column}")

        out_path = _write_fig_file(fig, filename=output_filename)
        b64 = _save_fig_to_base64(fig)
        return (
            json.dumps(
                {
                    "message": "time series plotted",
                    "file": str(out_path.relative_to(REPO_ROOT)),
                    "image_base64": b64,
                },
                ensure_ascii=False,
            ),
            str(out_path),
        )
    except Exception as e:
        return f"Error plotting time series: {e}", ""
    finally:
        if fig is not None:
            plt.close(fig)


__all__ = [
    "summarize_csv",
    "describe_stats",
    "plot_histogram",
    "plot_correlation_matrix",
    "plot_time_series",
]
