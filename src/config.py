from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime options shared by CLI and Streamlit app."""

    origin_mode: str = "bottom-center"
    use_llm: bool = False
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    min_area_ratio: float = 0.0015
    max_elements: int = 180
    balance_weight: float = 0.45
    accessibility_weight: float = 0.55
    top_crop_ratio: float = 0.055
    accessibility_max_reasonable_id: float = 6.2
    output_dir: Path = Path("outputs")

    def validate(self) -> None:
        if self.origin_mode not in {"bottom-center", "bottom-right"}:
            raise ValueError("origin_mode must be one of: bottom-center, bottom-right")
        if not 0.0 < self.min_area_ratio < 1.0:
            raise ValueError("min_area_ratio should be in (0, 1)")
        if self.max_elements <= 0:
            raise ValueError("max_elements must be > 0")
        if not 0.0 <= self.top_crop_ratio < 0.25:
            raise ValueError("top_crop_ratio should be in [0, 0.25)")
        if self.accessibility_max_reasonable_id <= 0.0:
            raise ValueError("accessibility_max_reasonable_id must be > 0")
        if self.use_llm and not self.llm_api_key:
            raise ValueError("LLM_API_KEY is required when use_llm=True")


def score_0_to_10(value_0_to_1: float) -> float:
    clamped = max(0.0, min(1.0, value_0_to_1))
    return round(clamped * 10.0, 2)
