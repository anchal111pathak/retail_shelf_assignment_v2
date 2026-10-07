from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

# EasyOCR is used for OCR because it is simpler to run reliably on
# Windows CPU environments than PaddleOCR in this prototype.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import easyocr


class ShelfMLPipeline:
    """
    Retail shelf prototype:
      1) YOLO26s detection, with optional overlapping tiled inference
      2) CLIP zero-shot brand recognition with prompt ensembling
      3) EasyOCR pretrained OCR running on CPU
      4) Optional reference-gallery brand recognition

    IMPORTANT:
    The official YOLO26 detection weights are COCO-pretrained. For the best
    retail-product recall, set detector.model to a YOLO26 checkpoint fine-tuned
    on SKU-110K (single class: product/object). The code remains runnable with
    yolo26s.pt as a baseline.
    """

    def __init__(self, config_path="configs/config.json"):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.cfg = cfg

        detector_cfg = cfg["detector"]
        self.detector = YOLO(detector_cfg["model"])

        clip_name = cfg["classifier"]["model"]
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.clip_model.eval()

        self.brands = cfg["classifier"]["brands"]
        self.brand_threshold = float(cfg["classifier"].get("min_confidence", 0.0))
        self.reference_gallery = self._load_reference_gallery(
            cfg["classifier"].get("reference_gallery", "")
        )

        self.ocr = self._build_ocr(cfg.get("ocr", {}))

    @staticmethod
    def _device(value: str):
        if value != "auto":
            return value
        return 0 if torch.cuda.is_available() else "cpu"

    def _build_ocr(self, cfg: Dict):
        """Initialize EasyOCR for English text on CPU."""
        try:
            return easyocr.Reader(
                [cfg.get("lang", "en")],
                gpu=False,
                verbose=False,
            )
        except Exception as exc:
            print(f"[WARN] Could not initialize EasyOCR: {exc}")
            return None

    def _load_reference_gallery(self, gallery_dir: str):
        """Load optional brand reference images as normalized CLIP embeddings."""
        if not gallery_dir:
            return {}
        root = Path(gallery_dir)
        if not root.exists():
            return {}

        gallery = {}
        for brand_dir in root.iterdir():
            if not brand_dir.is_dir():
                continue
            vectors = []
            for p in sorted(brand_dir.glob("*")):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                try:
                    image = Image.open(p).convert("RGB")
                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    with torch.inference_mode():
                        emb = self.clip_model.get_image_features(**inputs)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    vectors.append(emb[0].cpu())
                except Exception as exc:
                    print(f"[WARN] Gallery image failed: {p}: {exc}")
            if vectors:
                gallery[brand_dir.name] = torch.stack(vectors).mean(dim=0)
                gallery[brand_dir.name] /= gallery[brand_dir.name].norm()
        return gallery

    def _predict_one(self, image, conf, iou, imgsz, device):
        result = self.detector.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
            nms=True,
        )[0]
        detections = []
        if result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            conf_value = float(box.conf[0].detach().cpu())
            cls_id = int(box.cls[0].detach().cpu())
            detections.append({
                "bbox": [float(v) for v in xyxy],
                "detector_class": names.get(cls_id, str(cls_id)),
                "confidence": conf_value,
            })
        return detections

    @staticmethod
    def _tile_boxes(width: int, height: int, tile_size: int, overlap: float):
        step = max(1, int(tile_size * (1.0 - overlap)))
        xs = list(range(0, max(width - tile_size, 0) + 1, step))
        ys = list(range(0, max(height - tile_size, 0) + 1, step))
        if not xs or xs[-1] != max(width - tile_size, 0):
            xs.append(max(width - tile_size, 0))
        if not ys or ys[-1] != max(height - tile_size, 0):
            ys.append(max(height - tile_size, 0))
        return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
                for y in ys for x in xs]

    @staticmethod
    def _global_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
        if not detections:
            return []

        boxes = []
        scores = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append([int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))])
            scores.append(float(d["confidence"]))

        keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0,
                                nms_threshold=iou_threshold)
        if keep is None or len(keep) == 0:
            return []
        keep = np.asarray(keep).reshape(-1).tolist()
        output = [detections[int(i)] for i in keep]
        output.sort(key=lambda x: x["confidence"], reverse=True)
        return output

    def detect_products(self, image_path: str) -> List[Dict]:
        cfg = self.cfg["detector"]
        device = self._device(cfg.get("device", "auto"))
        conf = float(cfg.get("confidence", 0.15))
        iou = float(cfg.get("iou", 0.45))
        imgsz = int(cfg.get("imgsz", 1280))

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]

        # Full-image prediction keeps large objects/context.
        all_detections = self._predict_one(image, conf, iou, imgsz, device)

        # Overlapping tiles improve recall for small shelf products. This is
        # especially useful when the source image is high-resolution.
        if cfg.get("tiled_inference", True):
            tile_size = int(cfg.get("tile_size", 900))
            overlap = float(cfg.get("tile_overlap", 0.20))
            for x1, y1, x2, y2 in self._tile_boxes(width, height, tile_size, overlap):
                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                tile_dets = self._predict_one(tile, conf, iou, imgsz, device)
                for d in tile_dets:
                    bx1, by1, bx2, by2 = d["bbox"]
                    d["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                    all_detections.append(d)

        # Suppress duplicates produced by overlapping tiles.
        all_detections = self._global_nms(all_detections, iou_threshold=0.45)

        # Round only at the output boundary.
        for d in all_detections:
            d["bbox"] = [round(float(v), 2) for v in d["bbox"]]
            d["confidence"] = round(float(d["confidence"]), 4)
        return all_detections

    def _zero_shot_brand_batch(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not crops:
            return []

        prompts = []
        for brand in self.brands:
            prompts.extend([
                f"a photo of a {brand} retail product",
                f"a {brand} product package on a store shelf",
                f"the packaging of {brand}",
            ])

        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self.clip_processor(text=prompts, images=images,
                                     return_tensors="pt", padding=True)
        with torch.inference_mode():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image

        # Average the three prompt templates per brand.
        probs = logits.reshape(len(crops), len(self.brands), 3).mean(dim=2).softmax(dim=1)
        values, indices = probs.max(dim=1)

        results = []
        for value, idx in zip(values.tolist(), indices.tolist()):
            brand = self.brands[int(idx)]
            score = float(value)
            if brand == "Other" or score < self.brand_threshold:
                brand = "Other"
            results.append((brand, score))
        return results

    def _gallery_brand_batch(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not self.reference_gallery:
            return []
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self.clip_processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            emb = self.clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        names = list(self.reference_gallery.keys())
        refs = torch.stack([self.reference_gallery[n] for n in names])
        scores = emb.cpu() @ refs.T
        values, indices = scores.max(dim=1)
        return [(names[int(i)], float(v)) for v, i in zip(values, indices)]

    def classify_detections(self, image_bgr: np.ndarray,
                            detections: List[Dict]) -> List[Dict]:
        h, w = image_bgr.shape[:2]
        valid = []
        crops = []
        for idx, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            valid.append((idx, d))
            crops.append(crop)

        if not crops:
            return []

        gallery_results = self._gallery_brand_batch(crops)
        if gallery_results:
            brand_results = gallery_results
        else:
            brand_results = self._zero_shot_brand_batch(crops)

        enriched = []
        for (idx, d), (brand, brand_score) in zip(valid, brand_results):
            item = dict(d)
            item["brand"] = brand
            item["brand_confidence"] = round(float(brand_score), 4)
            enriched.append(item)
        return enriched

    def _ocr_once(self, image):
        """Run EasyOCR and return [bbox, text, confidence] results."""
        if self.ocr is None:
            return []

        return self.ocr.readtext(image)

    def run_ocr(self, image_path: str) -> List[Dict]:
        """Extract readable shelf-label/price-tag text using EasyOCR."""
        if self.ocr is None:
            return []

        image = cv2.imread(image_path)
        if image is None:
            return []

        # Run on a mildly upscaled image to improve small shelf-label text.
        scale = float(self.cfg.get("ocr", {}).get("upscale", 1.5))
        if scale > 1.0:
            image_for_ocr = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            image_for_ocr = image

        try:
            raw_results = self._ocr_once(image_for_ocr)
        except Exception as exc:
            print(f"[WARN] OCR inference failed: {exc}")
            return []

        items = []
        score_threshold = float(
            self.cfg.get("ocr", {}).get("min_confidence", 0.40)
        )

        # EasyOCR result format:
        # [
        #   [
        #       [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        #       "recognized text",
        #       confidence
        #   ],
        #   ...
        # ]
        for res in raw_results:
            if not isinstance(res, (list, tuple)) or len(res) < 3:
                continue

            box, text, score = res[0], res[1], float(res[2])
            text = str(text).strip()

            if not text or score < score_threshold:
                continue

            try:
                pts = np.asarray(box, dtype=float).reshape(-1, 2)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
            except Exception:
                continue

            items.append({
                "text": text,
                "confidence": round(score, 4),
                "bbox": [
                    round(float(x1 / scale), 2),
                    round(float(y1 / scale), 2),
                    round(float(x2 / scale), 2),
                    round(float(y2 / scale), 2),
                ],
            })

        # Remove exact duplicate OCR boxes/texts.
        unique = {}
        for item in items:
            key = (
                item["text"],
                tuple(round(v) for v in item["bbox"]),
            )
            unique[key] = item

        return list(unique.values())

