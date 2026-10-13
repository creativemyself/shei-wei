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
