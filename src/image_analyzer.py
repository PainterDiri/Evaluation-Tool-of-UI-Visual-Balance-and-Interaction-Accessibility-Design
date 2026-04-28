from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ui_types import UIElement


@dataclass(slots=True)
class ImageAnalysis:
    image: np.ndarray
    gray: np.ndarray
    width: int
    height: int
    elements: list[UIElement]
    crop_top_px: int
    original_width: int
    original_height: int


class ImageAnalyzer:
    """Extract candidate UI components from screenshot with OpenCV."""

    def __init__(
        self,
        min_area_ratio: float = 0.0015,
        max_elements: int = 180,
        top_crop_ratio: float = 0.055,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_elements = max_elements
        self.top_crop_ratio = top_crop_ratio

    def analyze(self, image_path: str | Path) -> ImageAnalysis:
        image = self._read_image(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        original_h, original_w = image.shape[:2]
        image, crop_top_px = self._crop_top_strip(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        edges = cv2.Canny(gray, 40, 130)
        kernel = np.ones((3, 3), np.uint8)
        merged = cv2.dilate(edges, kernel, iterations=1)
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = int(w * h * self.min_area_ratio)
        components: list[tuple[int, int, int, int, int]] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = int(cw * ch)
            if area < min_area:
                continue
            if cw < 12 or ch < 12:
                continue
            components.append((x, y, cw, ch, area))

        components.sort(key=lambda t: t[4], reverse=True)
        components = self._non_max_suppression(components)[: self.max_elements]

        elements: list[UIElement] = []
        for idx, (x, y, cw, ch, area) in enumerate(components, start=1):
            cx = x + cw / 2.0
            cy = y + ch / 2.0
            contrast = self._local_contrast(gray, x, y, cw, ch)
            visual_weight = area * contrast
            role_guess, is_interactive = self._guess_role(x, y, cw, ch, w, h)
            elements.append(
                UIElement(
                    id=idx,
                    x=x,
                    y=y,
                    w=cw,
                    h=ch,
                    area=area,
                    center_x=cx,
                    center_y=cy,
                    contrast=float(contrast),
                    visual_weight=float(visual_weight),
                    is_interactive=is_interactive,
                    role_guess=role_guess,
                )
            )

        return ImageAnalysis(
            image=image,
            gray=gray,
            width=w,
            height=h,
            elements=elements,
            crop_top_px=crop_top_px,
            original_width=original_w,
            original_height=original_h,
        )

    def _crop_top_strip(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Crop a small top strip to suppress status-bar noise from screenshots."""
        if self.top_crop_ratio <= 0.0:
            return image, 0

        h, _ = image.shape[:2]
        if h < 240:
            return image, 0

        ratio = max(0.0, min(self.top_crop_ratio, 0.18))
        crop_px = int(round(h * ratio))
        crop_px = max(16, min(crop_px, int(h * 0.16)))

        if crop_px <= 0 or crop_px >= h // 3:
            return image, 0
        return image[crop_px:, :].copy(), crop_px

    def _read_image(self, image_path: str | Path) -> np.ndarray | None:
        """Read image robustly on Windows (supports non-ASCII paths)."""
        path = Path(image_path)
        if not path.exists():
            return None

        try:
            buffer = np.fromfile(str(path), dtype=np.uint8)
            if buffer.size == 0:
                return None
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            return image
        except Exception:
            # Fallback for environments where np.fromfile/imdecode may fail.
            return cv2.imread(str(path))

    def _local_contrast(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return 1.0

        inner_std = float(np.std(roi))

        pad = 8
        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + w + pad)
        neighborhood = gray[y1:y2, x1:x2]
        outer_mean = float(np.mean(neighborhood)) if neighborhood.size else 0.0
        inner_mean = float(np.mean(roi))

        local_delta = abs(inner_mean - outer_mean)
        contrast = 1.0 + (inner_std / 64.0) + (local_delta / 96.0)
        return float(max(0.35, min(3.0, contrast)))

    def _guess_role(self, x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[str, bool]:
        aspect = w / max(h, 1)
        rel_h = h / max(height, 1)
        rel_w = w / max(width, 1)
        cy = y + h / 2.0

        if rel_w > 0.7 and rel_h > 0.06 and cy > height * 0.72:
            return "primary_cta", True
        if rel_h > 0.05 and 0.7 <= aspect <= 8.0:
            return "button_or_tab", True
        if rel_w > 0.25 and rel_h > 0.11:
            return "card", True
        if y < height * 0.18 and rel_h < 0.08:
            return "top_nav_or_status", False
        return "content_or_decoration", False

    def _non_max_suppression(self, boxes: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
        if not boxes:
            return []

        picked: list[tuple[int, int, int, int, int]] = []
        for candidate in boxes:
            x, y, w, h, _ = candidate
            keep = True
            for px, py, pw, ph, _ in picked:
                iou = self._iou((x, y, w, h), (px, py, pw, ph))
                if iou > 0.55:
                    keep = False
                    break
            if keep:
                picked.append(candidate)
        return picked

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = float((x2 - x1) * (y2 - y1))
        union = float(aw * ah + bw * bh - inter)
        return inter / union if union > 0 else 0.0
