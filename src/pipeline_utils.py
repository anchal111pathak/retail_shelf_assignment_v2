from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def detect_shelf_rows(image: np.ndarray, min_row_height: int = 25) -> List[Tuple[int, int]]:
    """Estimate horizontal shelf regions using robust horizontal edge density."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    horizontal = np.mean(edges, axis=1)
    smooth = np.convolve(horizontal, np.ones(31, dtype=np.float32) / 31, mode="same")

    threshold = max(float(np.percentile(smooth, 84)), 3.0)
    peaks = np.where(smooth > threshold)[0]
    if len(peaks) == 0:
        h = image.shape[0]
        step = max(h // 4, min_row_height)
        return [(i, min(i + step, h)) for i in range(0, h, step)]

    groups = []
    start = prev = int(peaks[0])
    for p in peaks[1:]:
        p = int(p)
        if p - prev > 12:
            groups.append((start, prev))
            start = p
        prev = p
    groups.append((start, prev))

    boundaries = [0] + [int((a + b) / 2) for a, b in groups] + [image.shape[0]]
    boundaries = sorted(set(boundaries))
    rows = []
    for y1, y2 in zip(boundaries[:-1], boundaries[1:]):
        if y2 - y1 >= min_row_height:
            rows.append((y1, y2))
    return rows


def estimate_share_of_shelf(detections: List[Dict], image_shape) -> Dict[str, float]:
    """Approximate horizontal facing share, not physical shelf area."""
    total = sum(max(0.0, d["bbox"][2] - d["bbox"][0]) for d in detections)
    if total <= 0:
        return {}

    by_brand = {}
    for d in detections:
        width = max(0.0, d["bbox"][2] - d["bbox"][0])
        brand = d.get("brand", "Other")
        by_brand[brand] = by_brand.get(brand, 0.0) + width
    return {k: round(v / total, 4) for k, v in sorted(by_brand.items(), key=lambda x: -x[1])}


def draw_results(image: np.ndarray, detections: List[Dict], ocr_items: List[Dict],
                 shelf_rows: List[Tuple[int, int]], output_path: str):
    canvas = image.copy()

    for idx, (y1, y2) in enumerate(shelf_rows):
        cv2.line(canvas, (0, y1), (canvas.shape[1], y1), (255, 180, 0), 2)
        cv2.putText(canvas, f"Shelf {idx + 1}", (10, max(20, y1 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f'{d.get("brand", "Other")} {d.get("brand_confidence", 0):.2f}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 220, 40), 1)

    for item in ocr_items:
        x1, y1, x2, y2 = map(int, item["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 70, 70), 2)
        cv2.putText(canvas, item["text"][:35], (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 70, 70), 1)

    cv2.imwrite(output_path, canvas)


def save_json(result: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
