"""test2-1 JPEG 품질 실험 결과를 그래프로 시각화한다."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "test_res" / "test2-1" / "test2-1_res.csv"
OUTPUT_DIR = PROJECT_ROOT / "test_res" / "test2-1" / "test2-1_graphs"


def load_results() -> list[dict]:
    rows = []
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            payload = json.loads(row["json"])
            results = payload["vlm_result"]["results"]
            rows.append(
                {
                    "experiment": int(row["experiment"]),
                    "quality": int(row["jpeg_quality"]),
                    "jpeg_size_kb": int(row["jpeg_size_bytes"]) / 1024,
                    "seconds": float(row["vlm_inference_seconds"]),
                    "classes": [item["identity"]["class_name"] for item in results],
                }
            )
    return rows


def apply_style() -> None:
    korean_font = Path("C:/Windows/Fonts/malgun.ttf")
    font_family = (
        font_manager.FontProperties(fname=korean_font).get_name()
        if korean_font.exists()
        else "DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "font.family": font_family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
        }
    )


def plot_latency(rows: list[dict]) -> None:
    qualities = np.array([row["quality"] for row in rows])
    seconds = np.array([row["seconds"] for row in rows])
    order = np.argsort(qualities)
    trend = np.poly1d(np.polyfit(qualities, seconds, 1))
    correlation = np.corrcoef(qualities, seconds)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.scatter(qualities, seconds, s=55, color="#2563EB", alpha=0.85, label="Experiment")
    ax.plot(
        qualities[order],
        trend(qualities[order]),
        color="#DC2626",
        linewidth=2.2,
        label="Linear trend",
    )
    ax.set_title("VLM Inference Time by JPEG Quality")
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("Inference time (seconds)")
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.96,
        f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_quality_vs_latency.png", dpi=180)
    plt.close(fig)


def plot_detection_count(rows: list[dict]) -> None:
    qualities = [row["quality"] for row in rows]
    counts = [len(row["classes"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(qualities, counts, marker="o", markersize=5.5, linewidth=2, color="#0F766E")
    ax.fill_between(qualities, counts, alpha=0.12, color="#14B8A6")
    ax.invert_xaxis()
    ax.set_title("Detected Object Count as JPEG Quality Decreases")
    ax.set_xlabel("JPEG quality (high to low)")
    ax.set_ylabel("Number of classified objects")
    ax.set_yticks(range(0, max(counts) + 2))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_quality_vs_object_count.png", dpi=180)
    plt.close(fig)


def plot_classification_heatmap(rows: list[dict]) -> None:
    frequencies = Counter(label for row in rows for label in row["classes"])
    labels = [label for label, _ in frequencies.most_common()]
    matrix = np.array(
        [[int(label in row["classes"]) for row in rows] for label in labels],
        dtype=int,
    )

    fig_height = max(6.5, 0.34 * len(labels) + 2.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_title("Classification Presence Across JPEG Quality Steps")
    ax.set_xlabel("Experiment (JPEG quality decreases left to right)")
    ax.set_ylabel("VLM class label")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(row["experiment"]) for row in rows], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    top = ax.secondary_xaxis("top")
    top.set_xticks(range(len(rows)))
    top.set_xticklabels([str(row["quality"]) for row in rows], rotation=90, fontsize=8)
    top.set_xlabel("JPEG quality")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_classification_changes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    apply_style()
    plot_latency(rows)
    plot_detection_count(rows)
    plot_classification_heatmap(rows)
    print(f"Saved 3 graphs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
