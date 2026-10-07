from ultralytics import YOLO


def main():
    print("=" * 60)
    print("YOLO26s + SKU-110K Retail Product Detector Training")
    print("=" * 60)

    model = YOLO("yolo26s.pt")

    model.train(
        data="SKU-110K.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        patience=8,
        project="runs",
        name="yolo26s_sku110k",
        exist_ok=True,
        save=True,
        plots=True,
    )

    print("\nTraining completed.")
    print(
        "Best model:"
        " runs/detect/yolo26s_sku110k/weights/best.pt"
    )


if __name__ == "__main__":
    main()