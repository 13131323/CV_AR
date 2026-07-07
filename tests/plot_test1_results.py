"""test1 CSV에서 TXT 결과를 복구하고 추론시간 그래프를 생성한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test1"
CSV_RESULT = RESULT_DIR / "test1_res.csv"
TEXT_RESULT = RESULT_DIR / "test1_res.txt"
GRAPH_RESULT = RESULT_DIR / "test1_word_limit_vs_latency.png"


def load_rows() -> list[dict]:
    rows = []
    with CSV_RESULT.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            payload = json.loads(row["json"])
            rows.append(
                {
                    "experiment": int(row["experiment"]),
                    "word_limit": int(row["word_limit"]),
                    "seconds": float(row["vlm_inference_seconds"]),
                    "timestamp": row["timestamp"],
                    "result": payload["vlm_result"],
                }
            )
    return rows


def restore_text(rows: list[dict]) -> None:
    lines = [
        "test1: image-only VLM reasoning word-limit experiment (30 -> 0)",
        "restored_from: test1_res.csv",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"[experiment {row['experiment']:02d}]",
                f"word_limit: {row['word_limit']}",
                f"vlm_inference_seconds: {row['seconds']:.6f}",
                f"timestamp: {row['timestamp']}",
                "json:",
                json.dumps(row["result"], ensure_ascii=False, indent=2),
                "",
            ]
        )
    temporary = TEXT_RESULT.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(TEXT_RESULT)


def plot_latency(rows: list[dict]) -> None:
    limits = np.array([row["word_limit"] for row in rows], dtype=float)
    seconds = np.array([row["seconds"] for row in rows], dtype=float)
    order = np.argsort(limits)
    trend = np.poly1d(np.polyfit(limits, seconds, 1))
    correlation = float(np.corrcoef(limits, seconds)[0, 1])

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(
        limits,
        seconds,
        marker="o",
        markersize=5.5,
        linewidth=1.7,
        color="#2563EB",
        alpha=0.85,
        label="Experiment",
    )
    ax.plot(
        limits[order],
        trend(limits[order]),
        linewidth=2.2,
        color="#DC2626",
        label="Linear trend",
    )
    ax.invert_xaxis()
    ax.set_title("VLM Inference Time by Micro CoT Word Limit")
    ax.set_xlabel("Reasoning word limit (30 to 0)")
    ax.set_ylabel("Inference time (seconds)")
    ax.set_xticks(range(30, -1, -2))
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
    fig.savefig(GRAPH_RESULT, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if len(rows) != 31:
        raise RuntimeError(f"31개 실험이 필요하지만 CSV에는 {len(rows)}개가 있습니다.")
    restore_text(rows)
    plot_latency(rows)
    print(f"Restored {len(rows)} experiments to {TEXT_RESULT}")
    print(f"Saved graph to {GRAPH_RESULT}")


if __name__ == "__main__":
    main()
