#!/usr/bin/env python3
"""
Generate publication-ready figures and the main comparison table from
 evaluation_summary.json.

Usage:
    python generate_paper_figures.py
    python generate_paper_figures.py --input outputs/evaluation_summary.json
    python generate_paper_figures.py --input outputs/evaluation_summary.json --output-dir outputs/paper_figures

Requires:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER = [
    "BM25 (Lexical)",
    "Dense (Ollama)",
    "Hybrid (RRF)",
    "Hybrid + Local Rerank",
]

DISPLAY_NAMES = {
    "BM25 (Lexical)": "BM25 (Lexical)",
    "Dense (Ollama)": "Dense (BGE-M3)",
    "Hybrid (RRF)": "Hybrid (RRF)",
    "Hybrid + Local Rerank": "Hybrid + Local Rerank",
}

CATEGORY_ORDER = ["PEN", "PROC", "CNCP", "CCL", "OFF", "CONST"]


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def method_data(data: dict) -> Dict[str, Dict[str, float]]:
    raw = data["retrieval_benchmarks"]
    missing = [m for m in METHOD_ORDER if m not in raw]
    if missing:
        raise KeyError("Missing expected method(s): " + ", ".join(missing))
    return {m: raw[m] for m in METHOD_ORDER}


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def add_bar_labels(ax, bars, values, fmt="{:.1f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_hit_rates(metrics: Dict[str, Dict[str, float]], out: Path) -> None:
    methods = list(metrics)
    x = np.arange(len(methods))
    width = 0.24
    values = {
        "Hit@1": [metrics[m]["hit_1"] * 100 for m in methods],
        "Hit@3": [metrics[m]["hit_3"] * 100 for m in methods],
        "Hit@5": [metrics[m]["hit_5"] * 100 for m in methods],
    }
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for (label, vals), offset in zip(values.items(), [-width, 0, width]):
        bars = ax.bar(x + offset, vals, width, label=label)
        add_bar_labels(ax, bars, vals)
    ax.set_title("Retrieval Accuracy at Different Cutoffs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods], rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out / "figure_1_hit_rates.png")


def plot_mrr(metrics: Dict[str, Dict[str, float]], out: Path) -> None:
    methods = list(metrics)
    vals = [metrics[m]["mrr"] for m in methods]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(range(len(methods)), vals)
    add_bar_labels(ax, bars, vals, fmt="{:.3f}")
    ax.set_title("Mean Reciprocal Rank (MRR@5)")
    ax.set_ylabel("MRR")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out / "figure_2_mrr.png")


def plot_latency(metrics: Dict[str, Dict[str, float]], out: Path) -> None:
    methods = list(metrics)
    vals = [metrics[m]["latency"] for m in methods]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(range(len(methods)), vals)
    add_bar_labels(ax, bars, vals, fmt="{:.2f}")
    ax.set_title("Average Retrieval Latency")
    ax.set_ylabel("Latency (seconds)")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out / "figure_3_latency.png")


def plot_tradeoff(metrics: Dict[str, Dict[str, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, m in metrics.items():
        x = m["latency"]
        y = m["hit_5"] * 100
        ax.scatter(x, y, s=90)
        ax.annotate(DISPLAY_NAMES.get(method, method), (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_title("Retrieval Accuracy–Latency Trade-off")
    ax.set_xlabel("Average Retrieval Latency (seconds)")
    ax.set_ylabel("Hit@5 (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out / "figure_4_accuracy_latency_tradeoff.png")


def extract_case_category(case_id: str) -> str:
    for category in CATEGORY_ORDER:
        if case_id.startswith(category + "_"):
            return category
    return "OTHER"


def plot_category_hit3(data: dict, out: Path) -> None:
    case_details = data.get("case_details", {})
    methods = [m for m in METHOD_ORDER if m in case_details and case_details[m]]
    categories = [
        cat for cat in CATEGORY_ORDER
        if any(extract_case_category(case["id"]) == cat for method in methods for case in case_details[method])
    ]
    category_scores: Dict[str, List[float]] = {m: [] for m in methods}
    for method in methods:
        by_cat: Dict[str, List[float]] = {cat: [] for cat in categories}
        for case in case_details[method]:
            cat = extract_case_category(case["id"])
            if cat in by_cat:
                by_cat[cat].append(case["metrics"]["hit_3"])
        category_scores[method] = [float(np.mean(by_cat[cat])) if by_cat[cat] else math.nan for cat in categories]

    x = np.arange(len(categories))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11.5, 6))
    for i, method in enumerate(methods):
        vals = [v * 100 if not math.isnan(v) else math.nan for v in category_scores[method]]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=DISPLAY_NAMES.get(method, method))
        valid_bars = [b for b, v in zip(bars, vals) if not math.isnan(v)]
        valid_vals = [v for v in vals if not math.isnan(v)]
        add_bar_labels(ax, valid_bars, valid_vals, fmt="{:.0f}")
    ax.set_title("Hit@3 Retrieval Accuracy by Question Category")
    ax.set_ylabel("Hit@3 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out / "figure_5_category_hit3.png")


def make_main_table(metrics: Dict[str, Dict[str, float]], out: Path) -> None:
    rows = []
    for method in METHOD_ORDER:
        m = metrics[method]
        rows.append([
            DISPLAY_NAMES.get(method, method),
            f"{m['hit_1'] * 100:.1f}%",
            f"{m['hit_3'] * 100:.1f}%",
            f"{m['hit_5'] * 100:.1f}%",
            f"{m['mrr']:.3f}",
            f"{m['latency']:.2f} s",
        ])

    columns = ["Retrieval Method", "Hit@1", "Hit@3", "Hit@5", "MRR@5", "Avg. Latency"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.34, 0.12, 0.12, 0.12, 0.12, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")
    ax.set_title("Table 1. Comparative Retrieval Performance", fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    save_figure(fig, out / "table_1_main_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/evaluation_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures"))
    args = parser.parse_args()

    data = load_results(args.input)
    metrics = method_data(data)

    print(f"Loaded benchmark: {data.get('benchmark_version', 'unknown')}")
    print(f"Embedding model: {data.get('embedding_model', 'unknown')}")
    print(f"LLM model: {data.get('llm_model', 'unknown')}")
    print(f"Methods included in paper comparison: {len(metrics)}")
    print("Excluded method: Hybrid Corrective")

    plot_hit_rates(metrics, args.output_dir)
    plot_mrr(metrics, args.output_dir)
    plot_latency(metrics, args.output_dir)
    plot_tradeoff(metrics, args.output_dir)
    plot_category_hit3(data, args.output_dir)
    make_main_table(metrics, args.output_dir)

    print("\nDone. Publication figures and main comparison table are ready.")


if __name__ == "__main__":
    main()
