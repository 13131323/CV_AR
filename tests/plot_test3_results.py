"""test3 캐시 interval 실험 결과를 그래프로 시각화한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test3"
SUMMARY_CSV = RESULT_DIR / "test3_interval_summary.csv"
METADATA_JSON = RESULT_DIR / "test3_metadata.json"
OUTPUT_DIR = RESULT_DIR / "test3_graphs"


def load_results() -> tuple[list[dict], dict]:
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {
                    "interval": int(row["interval"]),
                    "cache_rate": float(row["cache_rate"]),
                    "sam_mean": float(row["mean_sam_difference_ratio"]),
                    "sam_max": float(row["max_sam_difference_ratio"]),
                    "depth_mean": float(row["mean_depth_difference_ratio"]),
                    "depth_max": float(row["max_depth_difference_ratio"]),
                }
            )
    metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    return rows, metadata


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


def add_speed_estimates(rows: list[dict], metadata: dict) -> None:
    yolo = float(metadata["baseline_mean_yolo_seconds"])
    expensive = float(metadata["baseline_mean_sam_seconds"]) + float(
        metadata["baseline_mean_depth_seconds"]
    )
    baseline_seconds = yolo + expensive
    for row in rows:
        row["estimated_seconds"] = yolo + (1.0 - row["cache_rate"]) * expensive
        row["estimated_speedup"] = baseline_seconds / row["estimated_seconds"]


def plot_speed(rows: list[dict]) -> None:
    intervals = [row["interval"] for row in rows]
    seconds = [row["estimated_seconds"] for row in rows]
    speedups = [row["estimated_speedup"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(intervals, seconds, marker="o", linewidth=2.2, color="#2563EB")
    ax.set_title("Estimated Vision Compute Cost and Speedup by Cache Interval")
    ax.set_xlabel("SAM/Depth cache interval")
    ax.set_ylabel("Estimated compute time per frame (seconds)", color="#2563EB")
    ax.tick_params(axis="y", labelcolor="#2563EB")
    ax.set_xticks(intervals)

    speed_axis = ax.twinx()
    speed_axis.grid(False)
    speed_axis.plot(intervals, speedups, marker="s", linewidth=2.2, color="#DC2626")
    speed_axis.set_ylabel("Estimated speedup vs interval 1 (x)", color="#DC2626")
    speed_axis.tick_params(axis="y", labelcolor="#DC2626")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_interval_vs_estimated_speed.png", dpi=180)
    plt.close(fig)


def plot_sam_error(rows: list[dict]) -> None:
    intervals = [row["interval"] for row in rows]
    means = [row["sam_mean"] * 100 for row in rows]
    maxima = [row["sam_max"] * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(intervals, means, marker="o", linewidth=2.2, color="#0F766E", label="Mean")
    ax.plot(intervals, maxima, marker="s", linewidth=1.8, color="#F97316", label="Maximum")
    ax.set_title("SAM Mask Difference from Interval 1 Baseline")
    ax.set_xlabel("SAM/Depth cache interval")
    ax.set_ylabel("Mask difference (1 - IoU, %)")
    ax.set_xticks(intervals)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_interval_vs_sam_difference.png", dpi=180)
    plt.close(fig)


def plot_depth_error(rows: list[dict]) -> None:
    intervals = [row["interval"] for row in rows]
    means = [row["depth_mean"] * 100 for row in rows]
    maxima = [row["depth_max"] * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(intervals, means, marker="o", linewidth=2.2, color="#7C3AED", label="Mean")
    ax.plot(intervals, maxima, marker="s", linewidth=1.8, color="#E11D48", label="Maximum")
    ax.set_title("Depth Map Difference from Interval 1 Baseline")
    ax.set_xlabel("SAM/Depth cache interval")
    ax.set_ylabel("Normalized depth MAE (%)")
    ax.set_xticks(intervals)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_interval_vs_depth_difference.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, metadata = load_results()
    apply_style()
    add_speed_estimates(rows, metadata)
    plot_speed(rows)
    plot_sam_error(rows)
    plot_depth_error(rows)
    print(f"Saved 3 graphs to {OUTPUT_DIR}")
    for row in rows:
        print(
            f"interval={row['interval']:2d} "
            f"speedup={row['estimated_speedup']:.3f}x "
            f"sam_mean={row['sam_mean']:.4f} depth_mean={row['depth_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
