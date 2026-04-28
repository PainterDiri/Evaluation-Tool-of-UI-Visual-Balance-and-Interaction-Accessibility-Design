from __future__ import annotations

import math

from .ui_types import UIElement


def compute_physical_balance(
    elements: list[UIElement], width: int, height: int
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Physical balance score in [0,1] with stronger vertical-imbalance sensitivity."""
    geometric_center = (width / 2.0, height / 2.0)
    if not elements:
        return 0.0, geometric_center, geometric_center

    total_weight = sum(max(0.0, e.visual_weight) for e in elements)
    if total_weight <= 1e-9:
        return 0.0, geometric_center, geometric_center

    x_cm = sum(e.visual_weight * e.center_x for e in elements) / total_weight
    y_cm = sum(e.visual_weight * e.center_y for e in elements) / total_weight

    left_weight = sum(e.visual_weight for e in elements if e.center_x <= geometric_center[0])
    top_weight = sum(e.visual_weight for e in elements if e.center_y <= geometric_center[1])
    right_weight = total_weight - left_weight
    bottom_weight = total_weight - top_weight

    horizontal_mass_imbalance = abs(left_weight - right_weight) / total_weight
    vertical_mass_imbalance = abs(top_weight - bottom_weight) / total_weight

    center_offset_x = abs(x_cm - geometric_center[0]) / max(geometric_center[0], 1e-9)
    center_offset_y = abs(y_cm - geometric_center[1]) / max(geometric_center[1], 1e-9)

    imbalance = (
        0.2 * horizontal_mass_imbalance
        + 0.4 * vertical_mass_imbalance
        + 0.1 * center_offset_x
        + 0.3 * center_offset_y
    )

    bm = 1.0 - min(imbalance, 1.0)
    bm = max(0.0, min(1.0, bm))
    return bm, (x_cm, y_cm), geometric_center


def get_origin(width: int, height: int, mode: str) -> tuple[float, float]:
    if mode == "bottom-right":
        return width * 0.92, height * 0.94
    return width * 0.5, height * 0.94


def fitts_id(origin: tuple[float, float], element: UIElement) -> float:
    dx = element.center_x - origin[0]
    dy = element.center_y - origin[1]
    distance = math.hypot(dx, dy)

    size = max(min(element.w, element.h), 8)
    return math.log2(distance / size + 1.0)


def compute_accessibility(
    interactive_elements: list[UIElement],
    width: int,
    height: int,
    origin_mode: str,
    max_reasonable_id: float = 6.2,
) -> tuple[float, float, tuple[float, float]]:
    """Return accessibility score [0,1], weighted cost, and origin."""
    origin = get_origin(width, height, origin_mode)
    if not interactive_elements:
        fallback_cost = max(0.5, max_reasonable_id * 0.55)
        return 0.45, fallback_cost, origin

    weighted_sum = 0.0
    weight_sum = 0.0
    for element in interactive_elements:
        element.fitts_id = fitts_id(origin, element)
        alpha = max(0.1, min(1.0, element.semantic_weight))
        weighted_sum += alpha * element.fitts_id
        weight_sum += alpha

    weighted_cost = weighted_sum / max(weight_sum, 1e-9)

    # Empirical mapping: mobile UIs often fall within weighted ID in [0.3, 6.2]
    max_id = max(1e-6, max_reasonable_id)
    score = 1.0 - min(weighted_cost / max_id, 1.0)
    score = max(0.0, min(1.0, score))
    return score, weighted_cost, origin
