from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_GUIDELINE_FILE = PROJECT_ROOT / "data" / "guidelines.json"


def list_input_images() -> list[Path]:
    if not INPUT_DIR.exists():
        return []

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(sorted(INPUT_DIR.glob(pattern)))
    return sorted({path.resolve() for path in images})


def resolve_default_image() -> Path:
    images = list_input_images()
    if not images:
        raise FileNotFoundError(f"No image files found in {INPUT_DIR}")
    return images[0]
