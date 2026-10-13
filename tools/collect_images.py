"""定時從攝影機拍照存檔，用來收集訓練 YOLO 箱子模型的資料集。

建議在不同時段（早/午/晚、開燈/關燈）、不同堆放狀況下各收集一些，
總共 200～500 張即可訓練出堪用的模型。

用法：
    python tools/collect_images.py -c config/config.yaml --count 300 --interval 20
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_vision.camera import open_camera  # noqa: E402
from inventory_vision.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/config.yaml")
    ap.add_argument("--out", default="datasets/raw")
    ap.add_argument("--count", type=int, default=100, help="拍攝張數")
    ap.add_argument("--interval", type=float, default=30, help="每張間隔秒數")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open_camera(cfg["camera"]) as cam:
        for i in range(args.count):
            frame = cam.read()
            name = out / f"{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(name), frame)
            print(f"[{i + 1}/{args.count}] {name}")
            if i + 1 < args.count:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
