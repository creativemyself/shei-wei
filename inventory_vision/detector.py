"""箱子偵測器。

- YoloDetector   ：使用 Ultralytics YOLO（自行訓練的「箱子」模型），準確度高，建議正式使用。
- ContourDetector：傳統影像處理找矩形輪廓，不需訓練，適合架設初期驗證鏡頭位置。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0

    def anchor(self, mode: str = "center") -> tuple[float, float]:
        cx = (self.x1 + self.x2) / 2
        if mode == "bottom":
            return cx, self.y2
        return cx, (self.y1 + self.y2) / 2


class Detector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class YoloDetector(Detector):
    def __init__(self, model_path: str, confidence: float, iou: float, imgsz: int, classes=None, device=None):
        from ultralytics import YOLO

        log.info("載入 YOLO 模型：%s", model_path)
        self.model = YOLO(model_path)
        self.kwargs = dict(conf=confidence, iou=iou, imgsz=imgsz, classes=classes, verbose=False)
        if device is not None:
            self.kwargs["device"] = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, **self.kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [Detection(*map(float, b), float(c), int(k)) for b, c, k in zip(xyxy, conf, cls)]


class ContourDetector(Detector):
    """以邊緣偵測 + 輪廓近似，找出面積與「矩形程度」符合條件的區塊。

    rectangularity = 輪廓面積 / 最小外接旋轉矩形面積，越接近 1 越像矩形。
    """

    def __init__(self, min_area: float, max_area: float, min_rectangularity: float):
        self.min_area = min_area
        self.max_area = max_area
        self.min_rect = min_rectangularity

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            (_, (rw, rh), _) = cv2.minAreaRect(c)
            if rw * rh <= 0:
                continue
            rectangularity = area / (rw * rh)
            if rectangularity < self.min_rect:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection(x, y, x + w, y + h, confidence=float(min(rectangularity, 1.0))))
        return out


def build_detector(cfg: dict) -> Detector:
    kind = cfg.get("type", "yolo")
    if kind == "yolo":
        return YoloDetector(
            cfg["model_path"], cfg["confidence"], cfg["iou"], cfg["imgsz"], cfg.get("classes"), cfg.get("device")
        )
    if kind == "contour":
        return ContourDetector(cfg["min_area"], cfg["max_area"], cfg["min_rectangularity"])
    raise ValueError(f"未知的偵測器類型：{kind}")
