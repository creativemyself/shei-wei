"""攝影機抽象層：樹莓派 (picamera2)、Jetson (GStreamer)、USB (OpenCV)、靜態圖片。

所有後端的 read() 都回傳 OpenCV 慣用的 BGR numpy 陣列。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Camera:
    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def jetson_gstreamer_pipeline(sensor_id: int, width: int, height: int, flip_method: int = 0) -> str:
    """IMX219 在 Jetson 上的 GStreamer pipeline。

    以 3280x2464 全感光元件讀取（全視角），再由硬體 nvvidconv 縮放到指定解析度。
    appsink drop/max-buffers=1 確保每次讀到的是最新畫面，而不是緩衝區裡的舊畫面。
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), width=3280, height=2464, framerate=21/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1 max-buffers=1"
    )


class OpenCVCamera(Camera):
    """OpenCV VideoCapture：USB 攝影機，或 Jetson 上的 GStreamer pipeline。"""

    def __init__(self, source, width: int, height: int, warmup_frames: int, gstreamer: bool = False):
        import cv2

        self._cv2 = cv2
        if gstreamer:
            self.cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            hint = "（請確認 OpenCV 有啟用 GStreamer，Jetson 請用 JetPack 內建的 OpenCV，勿用 pip 版）" if gstreamer else ""
            raise RuntimeError(f"無法開啟攝影機：{source}{hint}")
        self.warmup_frames = warmup_frames

    def read(self) -> np.ndarray:
        for _ in range(self.warmup_frames):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("攝影機讀取失敗")
        return frame

    def close(self) -> None:
        self.cap.release()


class PiCamera2Camera(Camera):
    """樹莓派 CSI 鏡頭（IMX219 / Camera Module v2），使用 libcamera 的 picamera2。"""

    def __init__(self, width: int, height: int, sensor_id: int, warmup_frames: int):
        from picamera2 import Picamera2  # 樹莓派 OS: sudo apt install python3-picamera2

        self.cam = Picamera2(sensor_id)
        # RGB888 在記憶體中的排列為 B,G,R，可直接給 OpenCV 使用
        config = self.cam.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()
        self.warmup_frames = warmup_frames

    def read(self) -> np.ndarray:
        for _ in range(self.warmup_frames):
            self.cam.capture_array("main")
        return self.cam.capture_array("main")

    def close(self) -> None:
        self.cam.stop()
        self.cam.close()


class FileCamera(Camera):
    """每次讀取同一張（或資料夾中輪流的）圖片，供沒有攝影機時開發測試。"""

    def __init__(self, path: str):
        import cv2

        self._cv2 = cv2
        p = Path(path)
        if p.is_dir():
            self.files = sorted(x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        else:
            self.files = [p]
        if not self.files or not self.files[0].exists():
            raise FileNotFoundError(f"找不到測試圖片：{path}")
        self._i = 0

    def read(self) -> np.ndarray:
        f = self.files[self._i % len(self.files)]
        self._i += 1
        frame = self._cv2.imread(str(f))
        if frame is None:
            raise RuntimeError(f"無法讀取圖片：{f}")
        return frame


def _is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/sys/module/tegra_fuse")


def _is_raspberry_pi() -> bool:
    try:
        return "raspberry pi" in Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except OSError:
        return False


def open_camera(cfg: dict) -> Camera:
    backend = cfg.get("backend", "auto")
    if backend == "auto":
        backend = "jetson" if _is_jetson() else "picamera2" if _is_raspberry_pi() else "opencv"
        log.info("自動選擇攝影機後端：%s", backend)

    w, h, warm = cfg["width"], cfg["height"], cfg["warmup_frames"]
    if backend == "picamera2":
        return PiCamera2Camera(w, h, cfg["sensor_id"], warm)
    if backend == "jetson":
        pipeline = jetson_gstreamer_pipeline(cfg["sensor_id"], w, h, cfg["flip_method"])
        return OpenCVCamera(pipeline, w, h, warm, gstreamer=True)
    if backend == "opencv":
        return OpenCVCamera(cfg["device_index"], w, h, warm)
    if backend == "file":
        return FileCamera(cfg["file_path"])
    raise ValueError(f"未知的攝影機後端：{backend}")
