from langchain_core.tools import tool

import os
import json
import io
import base64
from datetime import datetime
from typing import Optional, Tuple

# Optional imports
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

# If user wants wordcloud support
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "output", "analysis")

if not os.path.isdir(OUT_DIR):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
    except Exception:
        pass


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def _save_fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _write_fig_file(fig, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = f"plot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------
# Analysis tools
# ---------------------------


@tool(response_format="content")
def list_analysis_packages() -> str:
    """Return which optional analysis packages are available."""
    available = {
        "pandas": pd is not None,
        "matplotlib": plt is not None,
        "seaborn": sns is not None,
        "wordcloud": WordCloud is not None,
    }
    return json.dumps(available)


@tool(response_format="content")
def summarize_csv(path: str, max_rows: int = 5) -> str:
    """Read a CSV file and return a short summary: head and describe().

    - `path` is relative to the repo root or absolute.
    - Returns JSON with `path`, `head` (CSV), and `describe` (dict) or an error message.
    """
    if pd is None:
        return "Error: pandas is not installed in the environment."

    csv_path = _resolve_path(path)
    if not os.path.exists(csv_path):
        return json.dumps({"error": f"file not found: {path}"})

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return json.dumps({"error": f"failed to read CSV: {e}"})

    head = df.head(max_rows).to_dict(orient="records")
    try:
        desc = df.describe(include="all").fillna("").to_dict()
    except Exception:
        desc = {}

    return json.dumps(
        {"path": os.path.relpath(csv_path, REPO_ROOT), "head": head, "describe": desc},
        default=str,
        ensure_ascii=False,
    )


@tool(response_format="content")
def describe_stats(path: str, column: Optional[str] = None) -> str:
    """Return descriptive statistics for a CSV or a specific column."""
    if pd is None:
        return "Error: pandas is not installed in the environment."

    csv_path = _resolve_path(path)
    if not os.path.exists(csv_path):
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
                    "available_columns": df.columns.tolist(),
                }
            )
        series = df[column]
        stats = series.describe().to_dict()
        # add missing values and dtype
        stats.update({"missing": int(series.isna().sum()), "dtype": str(series.dtype)})
        return json.dumps(
            {"column": column, "stats": stats}, default=str, ensure_ascii=False
        )

    else:
        try:
            desc = df.describe(include="all").fillna("").to_dict()
        except Exception:
            desc = {}
        return json.dumps(
            {"path": os.path.relpath(csv_path, REPO_ROOT), "describe": desc},
            default=str,
            ensure_ascii=False,
        )


@tool(response_format="content_and_artifact")
def plot_histogram(
    path: str, column: str, bins: int = 10, output_filename: Optional[str] = None
) -> Tuple[str, str]:
    """Plot a histogram for `column` from CSV at `path`.

    Returns (message, artifact_path). The artifact is a PNG file saved under `output/analysis/` and a base64 PNG in the response.
    """
    if pd is None:
        return "Error: pandas is not installed.", ""
    if plt is None:
        return "Error: matplotlib is not installed.", ""

    csv_path = _resolve_path(path)
    if not os.path.exists(csv_path):
        return f"Error: file not found: {path}", ""

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}", ""

    if column not in df.columns:
        return f"Error: column not found: {column}", ""

    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = pd.to_numeric(df[column], errors="coerce").dropna()
        ax.hist(data, bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.set_title(f"Histogram: {column}")

        if output_filename is None:
            out_path = _write_fig_file(fig)
        else:
            out_path = _write_fig_file(fig, filename=output_filename)

        b64 = _save_fig_to_base64(fig)
        return (
            json.dumps(
                {
                    "message": "histogram created",
                    "file": os.path.relpath(out_path, REPO_ROOT),
                    "image_base64": b64,
                }
            ),
            out_path,
        )
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        return f"Error plotting histogram: {e}", ""


@tool(response_format="content_and_artifact")
def plot_correlation_matrix(
    path: str, columns: Optional[str] = None, output_filename: Optional[str] = None
) -> Tuple[str, str]:
    """Plot a correlation matrix for numeric columns. `columns` can be a comma-separated list to limit.

    Returns (message, artifact_path).
    """
    if pd is None:
        return "Error: pandas is not installed.", ""
    if plt is None:
        return "Error: matplotlib is not installed.", ""

    csv_path = _resolve_path(path)
    if not os.path.exists(csv_path):
        return f"Error: file not found: {path}", ""

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}", ""

    try:
        if columns:
            cols = [c.strip() for c in columns.split(",")]
            df = df[cols]
        numeric = df.select_dtypes(include=["number"])
        corr = numeric.corr()

        fig, ax = plt.subplots(
            figsize=(min(12, 1 + len(corr) * 0.5), min(12, 1 + len(corr) * 0.5))
        )
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

        if output_filename is None:
            out_path = _write_fig_file(fig)
        else:
            out_path = _write_fig_file(fig, filename=output_filename)

        b64 = _save_fig_to_base64(fig)
        return (
            json.dumps(
                {
                    "message": "correlation matrix created",
                    "file": os.path.relpath(out_path, REPO_ROOT),
                    "image_base64": b64,
                }
            ),
            out_path,
        )
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        return f"Error plotting correlation matrix: {e}", ""


@tool(response_format="content_and_artifact")
def plot_time_series(
    path: str,
    date_column: str,
    value_column: str,
    date_format: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> Tuple[str, str]:
    """Plot a time series for `value_column` indexed by `date_column`.

    Returns (message, artifact_path).
    """
    if pd is None:
        return "Error: pandas is not installed.", ""
    if plt is None:
        return "Error: matplotlib is not installed.", ""

    csv_path = _resolve_path(path)
    if not os.path.exists(csv_path):
        return f"Error: file not found: {path}", ""

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}", ""

    if date_column not in df.columns or value_column not in df.columns:
        return f"Error: missing columns. Available: {df.columns.tolist()}", ""

    try:
        if date_format:
            df[date_column] = pd.to_datetime(
                df[date_column], format=date_format, errors="coerce"
            )
        else:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column])
        df = df.sort_values(by=date_column)

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

        if output_filename is None:
            out_path = _write_fig_file(fig)
        else:
            out_path = _write_fig_file(fig, filename=output_filename)

        b64 = _save_fig_to_base64(fig)
        return (
            json.dumps(
                {
                    "message": "time series plotted",
                    "file": os.path.relpath(out_path, REPO_ROOT),
                    "image_base64": b64,
                }
            ),
            out_path,
        )
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        return f"Error plotting time series: {e}", ""


@tool(response_format="content")
def generate_wordcloud(text: str, output_filename: Optional[str] = None) -> str:
    """Generate a word cloud PNG from input text. Returns JSON with message and path.

    Requires the `wordcloud` package.
    """
    if WordCloud is None:
        return "Error: wordcloud package not installed."
    if plt is None:
        return "Error: matplotlib not installed."

    if not text or not isinstance(text, str):
        return json.dumps({"error": "no text provided"})

    try:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        if output_filename is None:
            out_path = _write_fig_file(fig)
        else:
            out_path = _write_fig_file(fig, filename=output_filename)

        b64 = _save_fig_to_base64(fig)
        return (
            json.dumps(
                {
                    "message": "wordcloud generated",
                    "file": os.path.relpath(out_path, REPO_ROOT),
                    "image_base64": b64,
                }
            ),
        )
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        return json.dumps({"error": str(e)})


__all__ = [
    "list_analysis_packages",
    "summarize_csv",
    "describe_stats",
    "plot_histogram",
    "plot_correlation_matrix",
    "plot_time_series",
    "generate_wordcloud",
]
