from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_utils import detect_shelf_rows, estimate_share_of_shelf, draw_results, load_image, save_json
from model_pipeline import ShelfMLPipeline


def process_one(pipeline, image_path: Path, output_dir: Path):
    image = load_image(str(image_path))
    detections = pipeline.detect_products(str(image_path))
    detections = pipeline.classify_detections(image, detections)
    ocr_items = pipeline.run_ocr(str(image_path))

    rows = detect_shelf_rows(
        image,
        min_row_height=pipeline.cfg["shelf"].get("min_row_height", 25),
    )
    sos = estimate_share_of_shelf(detections, image.shape)

    brand_counts = {}
    for d in detections:
        brand = d.get("brand", "Other")
        brand_counts[brand] = brand_counts.get(brand, 0) + 1

    result = {
        "image_name": image_path.name,
        "total_products": len(detections),
        "brands": dict(sorted(brand_counts.items(), key=lambda x: -x[1])),
        "ocr_labels": [x["text"] for x in ocr_items],
        "share_of_shelf_estimate": sos,
        "shelf_rows": [{"y1": y1, "y2": y2} for y1, y2 in rows],
        "detections": detections,
        "ocr": ocr_items,
        "assumptions": [
            "YOLO26s is the baseline detector; retail-fine-tuned YOLO26 weights are recommended for final product-facing recall.",
            "Overlapping tiled inference is used to improve recall for small products in high-resolution shelf images.",
            "CLIP is a zero-shot brand baseline unless a reference gallery is configured.",
            "Share-of-shelf is an approximate horizontal facing proxy, not calibrated physical shelf area.",
            "OCR quality depends on label resolution, angle, blur, and lighting.",
        ],
    }

    out_img = output_dir / f"{image_path.stem}_annotated.jpg"
    out_json = output_dir / f"{image_path.stem}.json"
    draw_results(image, detections, ocr_items, rows, str(out_img))
    save_json(result, str(out_json))
    return result


def main():
    parser = argparse.ArgumentParser(description="Retail shelf ML pipeline")
    parser.add_argument("--input", default="data/test")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--config", default="configs/config.json")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ShelfMLPipeline(args.config)

    image_paths = sorted(
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    all_results = []
    for image_path in image_paths:
        print(f"[INFO] Processing {image_path.name}")
        result = process_one(pipeline, image_path, output_dir)
        all_results.append(result)
        print(
            f"[INFO] {image_path.name}: {result['total_products']} products | "
            f"OCR={len(result['ocr'])}"
        )

    (output_dir / "predictions.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
