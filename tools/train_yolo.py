"""訓練「箱子」YOLO 模型（建議在有 GPU 的電腦或 Google Colab 上執行，而不是在樹莓派上）。

資料集格式為 YOLO 格式（可用 Roboflow、CVAT、Label Studio 標註後匯出），例如：
    datasets/boxes/
        data.yaml          # names: {0: box}，train/val 路徑
        images/train, images/val
        labels/train, labels/val

用法：
    python tools/train_yolo.py --data datasets/boxes/data.yaml --epochs 100
    # 訓練完成後將 runs/detect/train/weights/best.pt 複製為 models/boxes.pt

部署到 Jetson 時建議轉成 TensorRT 加速（在 Jetson 上執行）：
    python tools/train_yolo.py --export models/boxes.pt --format engine
樹莓派可轉成 NCNN 格式加速：
    python tools/train_yolo.py --export models/boxes.pt --format ncnn
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="YOLO 資料集 data.yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="預訓練模型（n=最小最快，適合邊緣裝置）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--export", help="要匯出的 .pt 模型路徑")
    ap.add_argument("--format", default="engine", help="匯出格式：engine(TensorRT)、ncnn、onnx")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.export:
        path = YOLO(args.export).export(format=args.format, imgsz=args.imgsz, half=args.format == "engine")
        print(f"已匯出：{path}，請將 config.yaml 的 detector.model_path 指向此檔案")
        return
    if not args.data:
        ap.error("請指定 --data 或 --export")
    YOLO(args.base).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
