from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(slots=True)
class UIElement:
    id: int
    x: int
    y: int
    w: int
    h: int
    area: int
    center_x: float
    center_y: float
    contrast: float
    visual_weight: float
    is_interactive: bool
    role_guess: str
    semantic_weight: float = 0.4
    fitts_id: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoreBundle:
    physical_balance: float
    accessibility: float
    weighted_interaction_cost: float
    final_score: float


@dataclass(slots=True)
class EvaluationResult:
    image_path: str
    width: int
    height: int
    elements_total: int
    interactive_total: int
    center_of_mass: tuple[float, float]
    geometric_center: tuple[float, float]
    score_bundle: ScoreBundle
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    elements: list[UIElement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "elements_total": self.elements_total,
            "interactive_total": self.interactive_total,
            "center_of_mass": {
                "x": round(self.center_of_mass[0], 2),
                "y": round(self.center_of_mass[1], 2),
            },
            "geometric_center": {
                "x": round(self.geometric_center[0], 2),
                "y": round(self.geometric_center[1], 2),
            },
            "scores": {
                "physical_balance": round(self.score_bundle.physical_balance, 4),
                "accessibility": round(self.score_bundle.accessibility, 4),
                "weighted_interaction_cost": round(self.score_bundle.weighted_interaction_cost, 4),
                "final_score": round(self.score_bundle.final_score, 4),
            },
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "elements": [e.to_dict() for e in self.elements],
        }
