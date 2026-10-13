"""主程式：每分鐘 拍照 → 偵測箱子 → 依區域計數 → 平滑化 → 更新 Google 試算表。"""
from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .camera import Camera, open_camera
from .config import load_config
from .detector import Detection, Detector, build_detector
from .zones import CountSmoother, Zone, assign_to_zones, build_zones

log = logging.getLogger("inventory_vision")


@dataclass
class ScanResult:
    timestamp: str
    counts: dict[str, int]        # 平滑化後、乘上層數的箱數（寫入試算表）
    raw_counts: dict[str, int]    # 本次實際偵測到的框數
    detections: dict[str, list[Detection]]


def annotate(frame: np.ndarray, zones: list[Zone], per_zone: dict[str, list[Detection]], counts: dict[str, int]) -> np.ndarray:
    img = frame.copy()
    for z in zones:
        if z.polygon is not None:
            cv2.polylines(img, [z.polygon.astype(np.int32)], True, (255, 160, 0), 3)
            x, y = z.polygon[0, 0]
            label_pos = (int(x) + 8, int(y) + 36)
        else:
            label_pos = (10, 36)
        # OpenCV 內建字型不支援中文，標註文字只放數字
        cv2.putText(img, f"{counts.get(z.name, 0)}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 160, 0), 3)
        for d in per_zone.get(z.name, []):
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 220, 0), 2)
    return img


class InventoryScanner:
    def __init__(self, cfg: dict, camera: Camera, detector: Detector, writer=None):
        self.cfg = cfg
        self.camera = camera
        self.detector = detector
        self.writer = writer
        self.zones = build_zones(cfg["zones"])
        self.smoother = CountSmoother(cfg["schedule"]["smoothing_window"])
        self.tz = ZoneInfo(cfg["google_sheets"].get("timezone") or "Asia/Taipei")

    def scan_once(self) -> ScanResult:
        frame = self.camera.read()
        detections = self.detector.detect(frame)
        per_zone = assign_to_zones(detections, self.zones, self.cfg["zone_anchor"])
        raw = {z.name: len(per_zone[z.name]) for z in self.zones}
        smoothed = self.smoother.update(raw)
        counts = {z.name: smoothed[z.name] * z.layers for z in self.zones}
        ts = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S")

        log.info("掃描完成 %s | 偵測 %s | 回報 %s | 合計 %d", ts, raw, counts, sum(counts.values()))

        dbg = self.cfg["debug"]
        if dbg.get("save_annotated"):
            path = Path(dbg["annotated_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), annotate(frame, self.zones, per_zone, counts))
            # 原始畫面供 tools/calibrate_roi.py --image 與收集訓練資料使用
            cv2.imwrite(str(path.with_name("latest_raw.jpg")), frame)

        if self.writer is not None:
            self.writer.write(ts, counts, raw)
        return ScanResult(ts, counts, raw, per_zone)

    def run_forever(self, stop: threading.Event) -> None:
        """固定間隔執行；以「下一次預定時間」排程，不會因處理時間而逐漸延遲。"""
        interval = self.cfg["schedule"]["interval_seconds"]
        next_run = time.monotonic()
        while not stop.is_set():
            try:
                self.scan_once()
            except Exception:
                log.exception("本次掃描失敗，將於下一輪重試")
            next_run += interval
            now = time.monotonic()
            if next_run < now:  # 處理時間超過間隔，跳過落後的輪次
                next_run = now
            stop.wait(next_run - now)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="地板箱數影像辨識，定時更新 Google 試算表")
    ap.add_argument("-c", "--config", default="config/config.yaml", help="設定檔路徑")
    ap.add_argument("--once", action="store_true", help="只掃描一次後結束（測試用）")
    ap.add_argument("--dry-run", action="store_true", help="不寫入 Google 試算表")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg["debug"].get("log_level", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    writer = None
    if cfg["google_sheets"]["enabled"] and not args.dry_run:
        from .sheets import SheetsWriter

        writer = SheetsWriter(cfg["google_sheets"])

    detector = build_detector(cfg["detector"])
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    with open_camera(cfg["camera"]) as camera:
        scanner = InventoryScanner(cfg, camera, detector, writer)
        if args.once:
            scanner.scan_once()
        else:
            log.info("開始監測，每 %s 秒掃描一次（Ctrl+C 結束）", cfg["schedule"]["interval_seconds"])
            scanner.run_forever(stop)
    log.info("已停止")
    return 0
