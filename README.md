
# Retail Shelf Analysis — YOLO26s Prototype

## Objective

This prototype implements the assignment's requested retail shelf analysis flow:

1. Detect product facings / objects.
2. Classify detected products into broad brand categories.
3. OCR shelf labels / price tags.
4. Segment/estimate shelf rows and approximate share of shelf.
5. Produce business metrics and annotated images.

## Architecture

```text
Shelf Image
    |
    v
YOLO26s product detection
    |
    +--> Product crops --> CLIP zero-shot brand classification
    |
    +--> Bounding boxes --> shelf-row / space estimation
    |
    v
PaddleOCR --> price/shelf-label text
    |
    v
Business metrics
    +-- total products
    +-- brand counts
    +-- OCR labels
    +-- approximate share of shelf
```

## Model choices

### YOLO26s
YOLO26 is the current Ultralytics YOLO family. The `s` model is selected as
the accuracy/latency compromise for dense shelf imagery. The nano model is
preferable for strict CPU/edge constraints.

For a production-quality detector, fine-tune YOLO26s on a retail shelf
dataset such as SKU-110K rather than relying only on COCO pretrained classes.

### CLIP
The assignment asks for brand classification. A pretrained CLIP model provides
a practical zero-shot baseline without requiring a brand-labeled training set.
For production SKU recognition, replace this with an embedding/reference-gallery
system or a fine-tuned classifier.

### PaddleOCR
Used to extract visible price/shelf-label text. OCR is intentionally treated
as a separate stage because product detection and price-tag reading have
different visual characteristics.

### Shelf segmentation
A lightweight OpenCV horizontal-edge method estimates shelf rows. It avoids
introducing a large segmentation model for a task where shelf boundaries are
usually strong horizontal structures. A YOLO26 segmentation model can replace
this component when pixel-level shelf masks are required.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads pretrained model weights from their respective
open-source model hubs.

## Test images

The repository includes three supplied shelf images: `shelf_01.jpg`, `shelf_02.jpg`, and `shelf_03.jpg`.

## Run

From the repository root:

```bash
python src/run_pipeline.py --input data/test --output outputs
```

The pipeline creates:

```text
outputs/
  predictions.json
  shelf_01.json
  shelf_01_annotated.jpg
  shelf_02.json
  shelf_02_annotated.jpg
```

## GPU

If CUDA is available, YOLO automatically uses GPU through the `device=auto`
setting. Otherwise it runs on CPU.

For an NVIDIA GPU, verify PyTorch CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Optional retail fine-tuning

The file `src/train_sku110k.py` is a training template.

After preparing a YOLO-format SKU-110K dataset:

```bash
python src/train_sku110k.py
```

Then point `configs/config.json` to the resulting:

```text
runs/yolo26s_sku110k/weights/best.pt
```

This improves generic retail-product detection. Brand classification remains a
separate problem because SKU-110K is primarily a dense product detection
benchmark.

## Output example

```json
{
  "image_name": "shelf_01.jpg",
  "total_products": 42,
  "brands": {
    "Coca-Cola": 18,
    "Pepsi": 10,
    "Other": 14
  },
  "ocr_labels": ["₹50", "₹55"]
}
```

The exact numbers depend on model weights, confidence threshold, image quality,
and whether the detector has been fine-tuned.

## Assumptions

- A visible product facing is treated as one product instance.
- Occluded/partially visible products may be missed.
- Brand classification is broad-category classification, not guaranteed SKU
  recognition.
- OCR works best on high-resolution, front-facing shelf labels.
- Share-of-shelf is an approximation based on detected product-box width.
- The supplied images do not constitute a labeled training dataset.

## Limitations

1. COCO-pretrained YOLO does not know retail brands by itself.
2. Zero-shot CLIP can confuse visually similar packaging.
3. OCR can fail on small, blurred or angled price tags.
4. Approximate shelf-space calculation is not a replacement for calibrated
   physical-area measurement.
5. Production deployment should use a labeled retail dataset, confidence
   calibration, tracking/duplicate suppression, and a product reference
   catalog.

## Interview explanation

> I separated detection, recognition, OCR and shelf geometry because each task
> has different visual requirements. YOLO26s handles dense product detection,
> CLIP provides a zero-shot brand baseline, PaddleOCR extracts shelf-label
> text, and a lightweight shelf-row module estimates shelf occupancy. For
> production, I would fine-tune the detector on a retail shelf dataset such as
> SKU-110K and use a reference-gallery or fine-tuned embedding model for
> brand/SKU recognition.

## Improved inference configuration

The current prototype uses YOLO26s at `imgsz=1280`, confidence `0.15`, and overlapping tiled inference. Tiling is useful for high-resolution shelf photos because small products occupy fewer pixels in the full frame. Ultralytics supports `imgsz`, `conf`, `iou`, and `device` directly in prediction. For final retail accuracy, replace `yolo26s.pt` with a checkpoint fine-tuned on SKU-110K.

The OCR stage uses PaddleOCR 3.x with `enable_mkldnn=False` on CPU to avoid common Windows oneDNN runtime issues. OCR input is also upscaled before recognition.

### Important limitation

YOLO26 official detection weights are COCO-pretrained, so they are not a true retail-product detector. SKU-110K is a single-class dense retail-object dataset and is the correct next training stage for product-facing detection. The assignment should not claim that COCO-pretrained YOLO26 recognizes every snack packet or brand by itself.
