"""讀取 YAML 設定檔，並補上預設值。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "camera": {
        "backend": "auto",
        "width": 1640,
        "height": 1232,
        "sensor_id": 0,
        "flip_method": 0,
        "device_index": 0,
        "file_path": "samples/floor.jpg",
        "warmup_frames": 5,
    },
    "detector": {
        "type": "yolo",
        "model_path": "models/boxes.pt",
        "confidence": 0.45,
        "iou": 0.5,
        "imgsz": 640,
        "classes": None,
        "device": None,
        "min_area": 2500,
        "max_area": 200000,
        "min_rectangularity": 0.75,
    },
    "zones": [],
    "zone_anchor": "center",
    "schedule": {"interval_seconds": 60, "smoothing_window": 3},
    "google_sheets": {
        "enabled": False,
        "credentials_file": "credentials/service_account.json",
        "spreadsheet_id": "",
        "current_worksheet": "即時庫存",
        "log_worksheet": "掃描紀錄",
        "log_only_on_change": False,
        "pending_file": "data/pending_rows.jsonl",
        "timezone": "Asia/Taipei",
    },
    "debug": {
        "save_annotated": True,
        "annotated_path": "data/latest.jpg",
        "log_level": "INFO",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user_cfg)
    if cfg["schedule"]["interval_seconds"] <= 0:
        raise ValueError("schedule.interval_seconds 必須大於 0")
    if cfg["schedule"]["smoothing_window"] < 1:
        raise ValueError("schedule.smoothing_window 必須 >= 1")
    return cfg


# ── 多鏡頭：節點（Pi Zero 2 W）與中央（Jetson） ────────────────────────
NODE_DEFAULTS: dict[str, Any] = {
    "node": {
        "camera_id": "",
        "central_url": "http://jetson.local:8000",
        "token": "",
        "interval_seconds": 60,
        "jpeg_quality": 85,
        "timeout_seconds": 15,
        "retries": 3,
        "log_level": "INFO",
    },
    "camera": DEFAULTS["camera"],
}

CENTRAL_DEFAULTS: dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8000, "token": "", "max_upload_mb": 10},
    "detector": DEFAULTS["detector"],
    "schedule": {
        "interval_seconds": 60,
        "process_offset_seconds": 20,  # 節點在 :00 拍照，中央在 :20 統一辨識
        "stale_seconds": 150,          # 超過此秒數沒收到照片 → 標示離線
        "smoothing_window": 3,
    },
    "cameras": [],
    "google_sheets": DEFAULTS["google_sheets"],
    "debug": {"save_images": True, "image_dir": "data/cameras", "log_level": "INFO"},
}


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_node_config(path: str | Path) -> dict[str, Any]:
    cfg = _merge(NODE_DEFAULTS, _load_yaml(path))
    if not cfg["node"]["camera_id"]:
        raise ValueError("node.camera_id 未設定（例如 cam01）")
    if not cfg["node"]["token"]:
        raise ValueError("node.token 未設定（需與中央 server.token 相同）")
    return cfg


def load_central_config(path: str | Path) -> dict[str, Any]:
    cfg = _merge(CENTRAL_DEFAULTS, _load_yaml(path))
    if not cfg["server"]["token"]:
        raise ValueError("server.token 未設定")
    ids = [str(c.get("id", "")) for c in cfg["cameras"]]
    if not ids or not all(ids):
        raise ValueError("cameras 至少需要一個鏡頭，且每個都要有 id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"cameras 中有重複的 id：{ids}")
    s = cfg["schedule"]
    if not 0 <= s["process_offset_seconds"] < s["interval_seconds"]:
        raise ValueError("schedule.process_offset_seconds 必須介於 0 與 interval_seconds 之間")
    return cfg
