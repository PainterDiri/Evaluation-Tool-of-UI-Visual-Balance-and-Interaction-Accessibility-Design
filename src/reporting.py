from __future__ import annotations

import json
from pathlib import Path

from .ui_types import EvaluationResult


def save_json_report(result: EvaluationResult, output_dir: str | Path, filename: str = "evaluation_report.json") -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / filename

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    return str(report_path)
