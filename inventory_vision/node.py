"""鏡頭節點（Raspberry Pi Zero 2 W + IMX219）：每分鐘拍一張照片，上傳到中央 Jetson。

節點本身不做辨識，只需要 picamera2 與 OpenCV（JPEG 壓縮），不需安裝 YOLO。

用法：
    python -m inventory_vision.node -c config/node.yaml
"""
from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

import cv2

from .camera import open_camera
from .config import load_node_config
from .timing import wait_next_tick

log = logging.getLogger("inventory_vision.node")


def encode_jpeg(frame, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG 壓縮失敗")
    return buf.tobytes()


def upload(central_url: str, camera_id: str, token: str, jpeg: bytes, captured_at: float, timeout: float) -> None:
    url = f"{central_url.rstrip('/')}/upload/{camera_id}"
    req = urllib.request.Request(
        url,
        data=jpeg,
        method="POST",
        headers={
            "Content-Type": "image/jpeg",
            "X-Token": token,
            "X-Captured-At": f"{captured_at:.3f}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"中央回應 HTTP {resp.status}")


class CameraNode:
    def __init__(self, cfg: dict, camera, uploader=upload):
        self.cfg = cfg["node"]
        self.camera = camera
        self.uploader = uploader

    def capture_and_upload(self) -> bool:
        n = self.cfg
        frame = self.camera.read()
        captured_at = time.time()
        jpeg = encode_jpeg(frame, n["jpeg_quality"])
        for attempt in range(1, n["retries"] + 1):
            try:
                self.uploader(n["central_url"], n["camera_id"], n["token"], jpeg, captured_at, n["timeout_seconds"])
                log.info("已上傳 %s（%d KB）", datetime.fromtimestamp(captured_at).strftime("%H:%M:%S"), len(jpeg) // 1024)
                return True
            except (urllib.error.URLError, OSError, RuntimeError) as e:
                log.warning("上傳失敗（第 %d/%d 次）：%s", attempt, n["retries"], e)
                if attempt < n["retries"]:
                    time.sleep(min(2 ** attempt, 10))
        # 庫存只需要最新狀態，失敗的照片不補傳，下一分鐘會再拍
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="鏡頭節點：定時拍照上傳到中央")
    ap.add_argument("-c", "--config", default="config/node.yaml")
    ap.add_argument("--once", action="store_true", help="只拍一張上傳後結束（測試用）")
    args = ap.parse_args(argv)

    cfg = load_node_config(args.config)
    logging.basicConfig(level=cfg["node"]["log_level"], format="%(asctime)s [%(levelname)s] %(message)s")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    with open_camera(cfg["camera"]) as camera:
        node = CameraNode(cfg, camera)
        if args.once:
            return 0 if node.capture_and_upload() else 1
        log.info("節點 %s 啟動，每 %s 秒上傳到 %s", cfg["node"]["camera_id"], cfg["node"]["interval_seconds"], cfg["node"]["central_url"])
        while wait_next_tick(cfg["node"]["interval_seconds"], 0, stop):
            try:
                node.capture_and_upload()
            except Exception:
                log.exception("拍照失敗，下一輪重試")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
