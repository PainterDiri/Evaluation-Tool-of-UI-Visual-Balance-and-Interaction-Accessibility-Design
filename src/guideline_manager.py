from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GuidelineManager:
    def __init__(self, guideline_file: str | Path):
        self.guideline_file = Path(guideline_file)
        self.guidelines = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.guideline_file.exists():
            return []
        with self.guideline_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "guidelines" in raw and isinstance(raw["guidelines"], list):
            return raw["guidelines"]
        return []

    def as_prompt_text(self) -> str:
        if not self.guidelines:
            return "No guideline loaded."
        lines: list[str] = []
        for idx, item in enumerate(self.guidelines, start=1):
            title = item.get("title", f"Guideline-{idx}")
            desc = item.get("description", "")
            metric = item.get("metric", "")
            lines.append(f"[{idx}] {title} ({metric}) - {desc}")
        return "\n".join(lines)
