"""中央伺服器（NVIDIA Jetson）：接收各鏡頭節點的照片，每分鐘集中辨識，整批寫入 Google 試算表。

  節點 (:00 拍照上傳) ──HTTP──► 接收伺服器（保存各鏡頭最新照片）
                                     │
                         每分鐘 :20 ─┴─► YOLO 辨識 → 各區計數 → 平滑化 → 試算表（一次批次寫入）

HTTP 介面（同一區網內使用，皆需 token）：
  POST /upload/<camera_id>          節點上傳 JPEG（標頭 X-Token）
  GET  /status?token=...            各鏡頭狀態與箱數（JSON）
  GET  /image/<camera_id>?token=... 最新標註圖（用手機/瀏覽器檢查鏡頭畫面與辨識框）

用法：
    python -m inventory_vision.central -c config/central.yaml
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .app import annotate
from .config import load_central_config
from .detector import Detector, build_detector
from .timing import wait_next_tick
from .zones import CountSmoother, Zone, assign_to_zones, build_zones

log = logging.getLogger("inventory_vision.central")

CURRENT_HEADER = ["鏡頭", "位置", "區域", "箱數", "原始偵測數", "狀態", "照片時間", "更新時間"]
LOG_HEADER = ["時間", "鏡頭", "位置", "區域", "箱數", "原始偵測數"]

STATUS_OK = "正常"
STATUS_STALE = "離線"
STATUS_NEVER = "未連線"
STATUS_ERROR = "錯誤"


# ── 照片暫存 ────────────────────────────────────────────
@dataclass
class Frame:
    jpeg: bytes
    captured_at: float   # 節點拍攝時間（顯示用）
    received_at: float   # 中央收到時間（判斷離線用，不受節點時鐘誤差影響）


class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[str, Frame] = {}

    def put(self, camera_id: str, frame: Frame) -> None:
        with self._lock:
            self._frames[camera_id] = frame

    def get(self, camera_id: str) -> Frame | None:
        with self._lock:
            return self._frames.get(camera_id)


# ── 每個鏡頭的狀態 ──────────────────────────────────────
@dataclass
class CameraState:
    id: str
    name: str
    zones: list[Zone]
    anchor: str
    smoother: CountSmoother
    counts: dict[str, int] = field(default_factory=dict)
    raw: dict[str, int] = field(default_factory=dict)
    status: str = STATUS_NEVER
    captured_at: float | None = None
    annotated_jpeg: bytes | None = None


class CentralProcessor:
    def __init__(self, cfg: dict, detector: Detector, store: FrameStore, writer=None):
        self.cfg = cfg
        self.detector = detector
        self.store = store
        self.writer = writer
        self.tz = ZoneInfo(cfg["google_sheets"].get("timezone") or "Asia/Taipei")
        window = cfg["schedule"]["smoothing_window"]
        self.cameras = [
            CameraState(
                id=str(c["id"]),
                name=str(c.get("name", c["id"])),
                zones=build_zones(c.get("zones", [])),
                anchor=c.get("zone_anchor", "center"),
                smoother=CountSmoother(window),
            )
            for c in cfg["cameras"]
        ]
        self.by_id = {c.id: c for c in self.cameras}
        self._last_logged: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def _fmt(self, ts: float | None) -> str:
        return datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d %H:%M:%S") if ts else ""

    def _process_camera(self, cam: CameraState, now: float) -> None:
        frame = self.store.get(cam.id)
        if frame is None:
            cam.status = STATUS_NEVER
            return
        if now - frame.received_at > self.cfg["schedule"]["stale_seconds"]:
            cam.status = STATUS_STALE  # 保留最後一次的箱數
            return
        img = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            cam.status = STATUS_ERROR
            log.warning("鏡頭 %s 的照片無法解碼", cam.id)
            return

        per_zone = assign_to_zones(self.detector.detect(img), cam.zones, cam.anchor)
        raw = {z.name: len(per_zone[z.name]) for z in cam.zones}
        smoothed = cam.smoother.update(raw)
        cam.raw = raw
        cam.counts = {z.name: smoothed[z.name] * z.layers for z in cam.zones}
        cam.status = STATUS_OK
        cam.captured_at = frame.captured_at

        annotated = annotate(img, cam.zones, per_zone, cam.counts)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cam.annotated_jpeg = buf.tobytes() if ok else None
        dbg = self.cfg["debug"]
        if dbg.get("save_images"):
            d = Path(dbg["image_dir"]) / cam.id
            d.mkdir(parents=True, exist_ok=True)
            (d / "latest_raw.jpg").write_bytes(frame.jpeg)
            if cam.annotated_jpeg:
                (d / "latest.jpg").write_bytes(cam.annotated_jpeg)

    def process_once(self, now: float | None = None) -> tuple[list[list], list[list]]:
        now = time.time() if now is None else now
        with self._lock:
            for cam in self.cameras:
                try:
                    self._process_camera(cam, now)
                except Exception:
                    cam.status = STATUS_ERROR
                    log.exception("鏡頭 %s 處理失敗", cam.id)

            ts = self._fmt(now)
            current_rows, log_rows = [], []
            for cam in self.cameras:
                for z in cam.zones:
                    count = cam.counts.get(z.name, "")
                    current_rows.append(
                        [cam.id, cam.name, z.name, count, cam.raw.get(z.name, ""), cam.status, self._fmt(cam.captured_at), ts]
                    )
                    if cam.status != STATUS_OK:
                        continue  # 離線鏡頭不寫歷史紀錄，避免把舊數字當成新紀錄
                    key = (cam.id, z.name)
                    if self.cfg["google_sheets"].get("log_only_on_change") and self._last_logged.get(key) == count:
                        continue
                    self._last_logged[key] = count
                    log_rows.append([ts, cam.id, cam.name, z.name, count, cam.raw.get(z.name, 0)])

            total = sum(sum(c.counts.values()) for c in self.cameras)
            online = sum(c.status == STATUS_OK for c in self.cameras)
            current_rows.append(["合計", "", "", total, "", f"{online}/{len(self.cameras)} 台正常", "", ts])

        log.info("辨識完成：合計 %d 箱，%d/%d 台鏡頭正常 %s", total, online, len(self.cameras),
                 {c.id: (c.status, sum(c.counts.values())) for c in self.cameras})
        if self.writer is not None:
            self.writer.write_rows(current_rows, log_rows)
        return current_rows, log_rows

    def status(self) -> dict:
        with self._lock:
            return {
                c.id: {
                    "name": c.name,
                    "status": c.status,
                    "counts": c.counts,
                    "raw": c.raw,
                    "captured_at": self._fmt(c.captured_at),
                }
                for c in self.cameras
            }

    def run_forever(self, stop: threading.Event) -> None:
        s = self.cfg["schedule"]
        while wait_next_tick(s["interval_seconds"], s["process_offset_seconds"], stop):
            try:
                self.process_once()
            except Exception:
                log.exception("本輪處理失敗，下一輪重試")


# ── HTTP 伺服器 ─────────────────────────────────────────
def make_handler(processor: CentralProcessor, store: FrameStore, token: str, max_bytes: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "InventoryCentral/1.0"

        def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, supplied: str | None) -> bool:
            return bool(supplied) and hmac.compare_digest(supplied.encode(), token.encode())

        def do_POST(self):
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) != 2 or parts[0] != "upload":
                return self._send(404, b"not found")
            if not self._authorized(self.headers.get("X-Token")):
                return self._send(401, b"unauthorized")
            camera_id = parts[1]
            if camera_id not in processor.by_id:
                return self._send(404, f"unknown camera {camera_id}".encode())
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= max_bytes:
                return self._send(413, b"bad size")
            body = self.rfile.read(length)
            if not body.startswith(b"\xff\xd8"):
                return self._send(400, b"not a jpeg")
            now = time.time()
            try:
                captured_at = float(self.headers.get("X-Captured-At") or now)
            except ValueError:
                captured_at = now
            store.put(camera_id, Frame(body, captured_at, now))
            log.debug("收到 %s 照片 %d KB", camera_id, len(body) // 1024)
            self._send(200, b"ok")

        def do_GET(self):
            url = urlparse(self.path)
            if not self._authorized(parse_qs(url.query).get("token", [None])[0]):
                return self._send(401, b"unauthorized")
            parts = url.path.strip("/").split("/")
            if parts == ["status"]:
                body = json.dumps(processor.status(), ensure_ascii=False, indent=2).encode()
                return self._send(200, body, "application/json; charset=utf-8")
            if len(parts) == 2 and parts[0] == "image" and parts[1] in processor.by_id:
                jpeg = processor.by_id[parts[1]].annotated_jpeg
                return self._send(200, jpeg, "image/jpeg") if jpeg else self._send(404, b"no image yet")
            self._send(404, b"not found")

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="中央伺服器：接收鏡頭照片、集中辨識、更新 Google 試算表")
    ap.add_argument("-c", "--config", default="config/central.yaml")
    ap.add_argument("--dry-run", action="store_true", help="不寫入 Google 試算表")
    args = ap.parse_args(argv)

    cfg = load_central_config(args.config)
    logging.basicConfig(level=cfg["debug"]["log_level"], format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    writer = None
    if cfg["google_sheets"]["enabled"] and not args.dry_run:
        from .sheets import SheetsWriter

        writer = SheetsWriter(cfg["google_sheets"], current_header=CURRENT_HEADER, log_header=LOG_HEADER)

    store = FrameStore()
    processor = CentralProcessor(cfg, build_detector(cfg["detector"]), store, writer)
    srv_cfg = cfg["server"]
    handler = make_handler(processor, store, srv_cfg["token"], int(srv_cfg["max_upload_mb"] * 1024 * 1024))
    server = ThreadingHTTPServer((srv_cfg["host"], srv_cfg["port"]), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("中央伺服器啟動：http://%s:%s，管理 %d 台鏡頭", srv_cfg["host"], srv_cfg["port"], len(processor.cameras))
    try:
        processor.run_forever(stop)
    finally:
        server.shutdown()
    log.info("已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
