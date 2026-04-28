from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .metrics import get_origin
from .ui_types import EvaluationResult, UIElement


def draw_annotated_elements(image: np.ndarray, elements: list[UIElement]) -> np.ndarray:
    canvas = image.copy()
    for e in elements:
        color = (40, 180, 80) if e.is_interactive else (160, 160, 160)
        cv2.rectangle(canvas, (e.x, e.y), (e.x + e.w, e.y + e.h), color, 2)
        label = f"#{e.id} a={e.semantic_weight:.2f}"
        cv2.putText(canvas, label, (e.x, max(18, e.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def draw_balance_overlay(
    image: np.ndarray,
    center_of_mass: tuple[float, float],
    geometric_center: tuple[float, float],
) -> np.ndarray:
    canvas = image.copy()
    gx, gy = int(geometric_center[0]), int(geometric_center[1])
    cx, cy = int(center_of_mass[0]), int(center_of_mass[1])

    cv2.line(canvas, (gx, 0), (gx, canvas.shape[0] - 1), (220, 220, 220), 1)
    cv2.line(canvas, (0, gy), (canvas.shape[1] - 1, gy), (220, 220, 220), 1)
    cv2.circle(canvas, (gx, gy), 7, (255, 120, 20), -1)
    cv2.circle(canvas, (cx, cy), 8, (40, 60, 240), -1)
    cv2.line(canvas, (gx, gy), (cx, cy), (40, 60, 240), 2)
    return canvas


def draw_reachability_heatmap(width: int, height: int, origin_mode: str) -> np.ndarray:
    ox, oy = get_origin(width, height, origin_mode)
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)

    sigma = min(width, height) * 0.42
    heat = np.exp(-(distance**2) / (2 * sigma**2))
    heat = (heat * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    cv2.circle(colored, (int(ox), int(oy)), 10, (255, 255, 255), -1)
    return colored


def save_visuals(result: EvaluationResult, source_image: np.ndarray, output_dir: str | Path, origin_mode: str) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    annotated = draw_annotated_elements(source_image, result.elements)
    balance = draw_balance_overlay(source_image, result.center_of_mass, result.geometric_center)
    heatmap = draw_reachability_heatmap(result.width, result.height, origin_mode)

    blend = cv2.addWeighted(source_image, 0.5, heatmap, 0.5, 0)

    annotated_path = output / "elements_annotated.png"
    balance_path = output / "visual_balance_overlay.png"
    heatmap_path = output / "reachability_heatmap.png"
    blend_path = output / "reachability_blend.png"

    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(balance_path), balance)
    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(blend_path), blend)

    score_plot_path = output / "score_radar.png"
    _save_radar_plot(result, score_plot_path)

    return {
        "elements_annotated": str(annotated_path),
        "visual_balance_overlay": str(balance_path),
        "reachability_heatmap": str(heatmap_path),
        "reachability_blend": str(blend_path),
        "score_radar": str(score_plot_path),
    }


def _save_radar_plot(result: EvaluationResult, file_path: Path) -> None:
    labels = ["Physical Balance", "Accessibility", "Final Score"]
    values = [
        result.score_bundle.physical_balance,
        result.score_bundle.accessibility,
        result.score_bundle.final_score,
    ]
    values.append(values[0])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles.append(angles[0])

    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("UI Evaluation Radar", va="bottom")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
