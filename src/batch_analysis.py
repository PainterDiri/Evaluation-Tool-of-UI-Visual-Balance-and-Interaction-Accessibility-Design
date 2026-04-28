from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


@dataclass(slots=True)
class BatchEvaluationPoint:
    index: int
    image_name: str
    image_path: str
    physical_balance: float
    accessibility: float
    final_score: float
    quadrant: str


def _configure_matplotlib_chinese_font() -> None:
    """Configure a Chinese-capable font for matplotlib text rendering."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]

    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        installed = set()

    chosen = ""
    for name in candidates:
        if name in installed:
            chosen = name
            break

    existing = list(plt.rcParams.get("font.sans-serif", []))
    merged: list[str] = []
    if chosen:
        merged.append(chosen)
    for name in candidates + existing:
        if name not in merged:
            merged.append(name)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = merged
    plt.rcParams["axes.unicode_minus"] = False


def run_batch_analysis(
    pipeline,
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    save_individual_artifacts: bool = False,
    x_threshold: float = 0.75,
    y_threshold: float = 0.65,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> dict[str, Any]:
    _configure_matplotlib_chinese_font()

    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("No images provided for batch analysis")

    if not (0.0 <= x_threshold <= 1.0 and 0.0 <= y_threshold <= 1.0):
        raise ValueError("Quadrant thresholds must be in [0, 1]")

    points: list[BatchEvaluationPoint] = []
    total = len(paths)
    for idx, image_path in enumerate(paths, start=1):
        output = pipeline.run(image_path=image_path, save_artifacts=save_individual_artifacts)
        result = output["result"]

        physical_balance = float(result.score_bundle.physical_balance)
        accessibility = float(result.score_bundle.accessibility)
        final_score = float(result.score_bundle.final_score)
        quadrant = _classify_quadrant(physical_balance, accessibility, x_threshold, y_threshold)

        points.append(
            BatchEvaluationPoint(
                index=idx,
                image_name=image_path.name,
                image_path=str(image_path),
                physical_balance=physical_balance,
                accessibility=accessibility,
                final_score=final_score,
                quadrant=quadrant,
            )
        )

        if progress_callback is not None:
            progress_callback(idx, total, image_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    quadrant_plot_path = output_path / "batch_quadrant_plot.png"
    quadrant_thumb_plot_path = output_path / "batch_quadrant_plot_thumbnails.png"
    summary_path = output_path / "batch_evaluation_summary.json"

    _save_quadrant_plot(points, quadrant_plot_path, x_threshold=x_threshold, y_threshold=y_threshold)
    _save_quadrant_thumbnail_plot(points, quadrant_thumb_plot_path, x_threshold=x_threshold, y_threshold=y_threshold)

    quadrant_counts = _build_quadrant_counts(points)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "images_count": len(points),
        "quadrant_thresholds": {
            "physical_balance": round(x_threshold, 4),
            "accessibility": round(y_threshold, 4),
        },
        "quadrant_counts": quadrant_counts,
        "results": [asdict(point) for point in points],
        "artifacts": {
            "quadrant_plot": str(quadrant_plot_path),
            "quadrant_plot_with_thumbnails": str(quadrant_thumb_plot_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "count": len(points),
        "quadrant_counts": quadrant_counts,
        "summary_path": str(summary_path),
        "quadrant_plot": str(quadrant_plot_path),
        "quadrant_plot_with_thumbnails": str(quadrant_thumb_plot_path),
        "records": [asdict(point) for point in points],
    }


def _classify_quadrant(balance: float, accessibility: float, x_threshold: float, y_threshold: float) -> str:
    if balance >= x_threshold and accessibility >= y_threshold:
        return "Q1"
    if balance < x_threshold and accessibility >= y_threshold:
        return "Q2"
    if balance < x_threshold and accessibility < y_threshold:
        return "Q3"
    return "Q4"


def _build_quadrant_counts(points: Sequence[BatchEvaluationPoint]) -> dict[str, int]:
    counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for point in points:
        counts[point.quadrant] = counts.get(point.quadrant, 0) + 1
    return counts


def _save_quadrant_plot(
    points: Sequence[BatchEvaluationPoint],
    file_path: Path,
    *,
    x_threshold: float,
    y_threshold: float,
) -> None:
    color_map = {
        "Q1": "#2ca02c",
        "Q2": "#1f77b4",
        "Q3": "#d62728",
        "Q4": "#ff7f0e",
    }

    fig = plt.figure(figsize=(16, 10), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.4], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])

    ax.axvspan(0.0, x_threshold, y_threshold, 1.0, color="#e8f2ff", alpha=0.65)
    ax.axvspan(x_threshold, 1.0, y_threshold, 1.0, color="#e9f8ec", alpha=0.65)
    ax.axvspan(0.0, x_threshold, 0.0, y_threshold, color="#fdeaea", alpha=0.65)
    ax.axvspan(x_threshold, 1.0, 0.0, y_threshold, color="#fff3e6", alpha=0.65)

    ax.axvline(x_threshold, color="#3b3b3b", linestyle="--", linewidth=1.2)
    ax.axhline(y_threshold, color="#3b3b3b", linestyle="--", linewidth=1.2)

    for point in points:
        color = color_map.get(point.quadrant, "#444444")
        ax.scatter(
            point.physical_balance,
            point.accessibility,
            s=120,
            color=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        ax.text(
            point.physical_balance,
            point.accessibility,
            str(point.index),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Physical Visual Balance", fontsize=11)
    ax.set_ylabel("Interaction Accessibility", fontsize=11)
    ax.set_title("Batch UI Quadrant Distribution (Indexed Points)", fontsize=13)
    ax.grid(alpha=0.25, linestyle=":")

    ax.text(0.99, 0.98, "Q1", transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold")
    ax.text(0.01, 0.98, "Q2", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.text(0.01, 0.02, "Q3", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax.text(0.99, 0.02, "Q4", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, fontweight="bold")

    legend_ax.axis("off")
    legend_ax.set_title("Index -> Image", fontsize=12, pad=8)

    entries = [f"{p.index:02d}. {p.image_name}" for p in points]
    split_at = (len(entries) + 1) // 2
    left_text = "\n".join(entries[:split_at])
    right_text = "\n".join(entries[split_at:])

    legend_ax.text(0.0, 1.0, left_text, ha="left", va="top", fontsize=8)
    legend_ax.text(0.5, 1.0, right_text, ha="left", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)


def _save_quadrant_thumbnail_plot(
    points: Sequence[BatchEvaluationPoint],
    file_path: Path,
    *,
    x_threshold: float,
    y_threshold: float,
) -> None:
    color_map = {
        "Q1": "#2ca02c",
        "Q2": "#1f77b4",
        "Q3": "#d62728",
        "Q4": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(12, 12), dpi=140)

    ax.axvline(x_threshold, color="#3b3b3b", linestyle="--", linewidth=1.2)
    ax.axhline(y_threshold, color="#3b3b3b", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, linestyle=":")
    ax.set_xlabel("Physical Visual Balance", fontsize=11)
    ax.set_ylabel("Interaction Accessibility", fontsize=11)
    ax.set_title("Batch UI Quadrant Distribution (Image Thumbnails)", fontsize=13)

    for point in points:
        color = color_map.get(point.quadrant, "#444444")
        x_pos, y_pos = _jittered_position(point)

        rgb = _read_image_rgb(Path(point.image_path))
        if rgb is None:
            ax.scatter(x_pos, y_pos, s=65, color=color, edgecolors="white", linewidths=0.8, zorder=3)
            ax.text(x_pos, y_pos, str(point.index), ha="center", va="center", fontsize=7, color="white", zorder=4)
            continue

        thumb = _thumbnail(rgb, max_side=96)
        image_box = OffsetImage(thumb, zoom=0.24)
        ab = AnnotationBbox(
            image_box,
            (x_pos, y_pos),
            frameon=True,
            pad=0.12,
            bboxprops={"edgecolor": color, "linewidth": 1.1},
            zorder=4,
        )
        ax.add_artist(ab)
        ax.text(x_pos, max(0.01, y_pos - 0.035), str(point.index), ha="center", va="top", fontsize=6, color="#111111", zorder=5)

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)


def _jittered_position(point: BatchEvaluationPoint) -> tuple[float, float]:
    # Add deterministic micro offsets to reduce full overlap in dense clusters.
    jx = ((point.index * 37) % 11 - 5) * 0.0032
    jy = ((point.index * 23) % 11 - 5) * 0.0032
    x_pos = float(np.clip(point.physical_balance + jx, 0.02, 0.98))
    y_pos = float(np.clip(point.accessibility + jy, 0.02, 0.98))
    return x_pos, y_pos


def _read_image_rgb(image_path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
    except Exception:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _thumbnail(rgb: np.ndarray, max_side: int = 96) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return rgb

    scale = max_side / float(max(h, w))
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)