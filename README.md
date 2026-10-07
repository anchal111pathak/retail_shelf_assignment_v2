# Retail Shelf Image Analysis

## 1. Project Overview

This project implements a practical retail shelf analysis pipeline for extracting product presence, brand-wise counts, shelf-label text, shelf rows, and an approximate share-of-shelf estimate from shelf images.

### Business problem

The pipeline is designed for retail intelligence use cases such as:

- On-Shelf Availability (OSA)
- Share of Shelf (SOS)
- Price/shelf-label visibility
- Basic planogram/shelf-row analysis
- Brand-wise product presence

### Case study

> Analyze shelf images and generate brand-wise shelf presence + product availability insights.

## 2. Pipeline

```text
Shelf Image
    |
    v
YOLO26s fine-tuned on SKU-110K
    |
    +--> Product bounding boxes
    |
    v
Product crops
    |
    v
CLIP zero-shot brand classification
    |
    +--> Brand / Other
    |
    +--> Brand-wise counts
    |
    v
EasyOCR
    |
    +--> Shelf labels / prices / visible text
    |
    v
Shelf-row detection + horizontal facing estimate
    |
    +--> Approximate share of shelf
    |
    v
JSON + Annotated Image
```

## 3. Models Used

### Product detection — YOLO26s + SKU-110K fine-tuning

The final detector is the fine-tuned `best.pt` checkpoint.

- Base model: YOLO26s
- Fine-tuning dataset: SKU-110K
- Training: 30 epochs on a GPU
- SKU-110K is a single-class retail-object dataset
- The detector learns product/facing localization, not brand identity

### Brand classification — CLIP

`openai/clip-vit-base-patch32` is used for zero-shot brand classification. Multiple text prompts per brand are averaged.

Current confidence threshold: `0.45`.

- score >= 0.45 -> predicted brand
- score < 0.45 -> `Other`

### OCR — EasyOCR

EasyOCR extracts readable shelf labels, prices and package/shelf text. OCR quality depends on resolution, blur, lighting, perspective and text size.

### Shelf analysis

OpenCV-based shelf-row detection identifies horizontal shelf regions. Share of shelf is an **approximate horizontal facing proxy**, not calibrated physical shelf area.

## 4. Repository Structure

```text
retail_shelf_assignment_v2/
|
├── configs/config.json
├── data/test/
├── docs/architecture.md
├── outputs/
├── src/
│   ├── model_pipeline.py
│   ├── pipeline_utils.py
│   ├── run_pipeline.py
│   └── train_sku110k.py
├── best.pt
├── yolo26s.pt
├── requirements.txt
└── README.md
```

## 5. Setup and Run

```bash
pip install -r requirements.txt
```

The current `configs/config.json` uses `best.pt` as the detector and a CLIP confidence threshold of `0.45`.

On Windows CPU:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
python -u src\run_pipeline.py
```

Custom input/output:

```bash
python src/run_pipeline.py --input data/test --output outputs
```

## 6. Output

For every shelf image the pipeline produces JSON and an annotated image. JSON contains total products, brand counts, OCR labels, detailed detections, confidences, shelf rows, approximate share-of-shelf and assumptions.

Example:

```json
{
  "image_name": "shelf_01.jpg",
  "total_products": 200,
  "brands": {
    "Other": 105,
    "Tropicana": 12,
    "Minute Maid": 11
  },
  "ocr_labels": ["Orange", "Mixed", "FANTA", "Limca"]
}
```

## 7. Example Results

Final run on the three provided shelf images:

| Image | Products detected | OCR items |
|---|---:|---:|
| shelf_01.jpg | 200 | 111 |
| shelf_02.jpg | 103 | 135 |
| shelf_03.jpg | 63 | 140 |

These are pipeline outputs on the supplied test images, not benchmark accuracy metrics.

### Shelf 01

Top brand counts include Other 105, Tropicana 12, Minute Maid 11, Coca-Cola 7, Red Bull 6 and Gatorade 6.

### Shelf 02

Top brand counts include Amul 40, Other 27, Actimel 10, Milky Mist 7 and Yakult 7.

### Shelf 03

Top brand counts include Other 19, Lay's 7, Good Day 4, Pringles 4, Uncle Chipps 4, Kurkure 4, Parle-G 3, Cheetos 3, Doritos 3 and Bingo 3.

## 8. Model Selection Rationale

### Why YOLO26s?

Retail shelves contain many small, adjacent and partially overlapping products. YOLO26s provides a practical balance between localization quality, inference speed and model size. Fine-tuning on SKU-110K specializes the detector for dense retail product localization rather than general COCO objects.

### Why separate brand classification?

SKU-110K is a single-class retail-object dataset. The detector therefore identifies product/facing regions, while CLIP handles the brand taxonomy. This keeps the pipeline modular.

### Why CLIP?

CLIP provides a practical zero-shot baseline without requiring a custom labeled brand dataset. Multiple prompts are averaged to reduce sensitivity to one wording.

### Why EasyOCR?

EasyOCR is practical for CPU inference and provides a lightweight OCR stage for visible shelf labels and package text.

## 9. CPU vs GPU

- **Training:** GPU strongly preferred; the retail fine-tuning was performed on GPU.
- **Inference:** CPU supported for local demonstration; GPU preferred when latency matters.

## 10. Limitations and Trade-offs

1. Brand classification is zero-shot and can confuse visually similar packaging.
2. Low-confidence brand predictions are intentionally mapped to `Other`.
3. OCR quality depends on image quality and text visibility.
4. Share of shelf is an approximate horizontal-facing proxy, not calibrated physical shelf area.
5. SKU-110K is single-class, so brand identity is handled separately.
6. The prototype does not perform video tracking.
7. Full planogram compliance would require expected shelf layouts or planogram reference data.
8. The three supplied images are demonstration inputs, not a statistically representative evaluation set.

## 11. Production Evolution

A production system could add a labeled SKU/brand dataset, a reference-image gallery, a fine-tuned brand classifier, calibrated OCR preprocessing, segmentation for physical shelf-area measurement, planogram reference data, confidence calibration, formal evaluation metrics, and ONNX/TensorRT deployment.

The current implementation intentionally prioritizes a modular, explainable and practical prototype rather than production perfection.
