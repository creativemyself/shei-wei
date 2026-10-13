import json

import cv2
import numpy as np
import pytest

from inventory_vision.app import InventoryScanner
from inventory_vision.config import DEFAULTS, _merge
from inventory_vision.detector import ContourDetector, Detection
from inventory_vision.sheets import CURRENT_HEADER, SheetsWriter
from inventory_vision.zones import WHOLE_FRAME, CountSmoother, assign_to_zones, build_zones


def synthetic_floor(n_boxes: int, size=(600, 800)) -> np.ndarray:
    """灰色地板上畫 n 個深色矩形當作箱子。"""
    img = np.full((*size, 3), 170, np.uint8)
    for i in range(n_boxes):
        x = 40 + (i % 5) * 150
        y = 40 + (i // 5) * 150
        cv2.rectangle(img, (x, y), (x + 100, y + 80), (60, 90, 120), -1)
    return img


def test_contour_detector_counts_rectangles():
    det = ContourDetector(min_area=3000, max_area=50000, min_rectangularity=0.75)
    assert len(det.detect(synthetic_floor(7))) == 7
    assert det.detect(synthetic_floor(0)) == []


def test_assign_to_zones_and_ignore_outside():
    zones = build_zones([
        {"name": "A", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        {"name": "B", "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]], "layers": 3},
    ])
    dets = [Detection(10, 10, 30, 30), Detection(210, 10, 250, 50), Detection(500, 500, 520, 520)]
    out = assign_to_zones(dets, zones)
    assert len(out["A"]) == 1 and len(out["B"]) == 1
    assert zones[1].layers == 3


def test_empty_zones_means_whole_frame():
    zones = build_zones([])
    assert zones[0].name == WHOLE_FRAME
    assert len(assign_to_zones([Detection(1, 1, 2, 2)], zones)[WHOLE_FRAME]) == 1


def test_smoother_filters_transient_occlusion():
    s = CountSmoother(window=3)
    s.update({"A": 10})
    s.update({"A": 10})
    assert s.update({"A": 4})["A"] == 10  # 有人短暫遮住
    assert s.update({"A": 4})["A"] == 4   # 持續變化才更新


class FakeCamera:
    def __init__(self, frames):
        self.frames = frames

    def read(self):
        return self.frames.pop(0)


class FakeWorksheet:
    def __init__(self):
        self.values, self.appended = [], []

    def batch_clear(self, ranges):
        self.values = []

    def update(self, values, range_name, value_input_option=None):
        self.values = values

    def append_row(self, row):
        self.appended.append(row)

    def append_rows(self, rows, value_input_option=None):
        self.appended.extend(rows)


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.sheets = {}

    def open_by_key(self, key):
        if self.fail:
            raise ConnectionError("offline")
        return self

    def worksheet(self, title):
        if title not in self.sheets:
            raise type("WorksheetNotFound", (Exception,), {})()
        return self.sheets[title]

    def add_worksheet(self, title, rows, cols):
        self.sheets[title] = FakeWorksheet()
        return self.sheets[title]


@pytest.fixture
def cfg(tmp_path):
    c = _merge(DEFAULTS, {
        "zones": [
            {"name": "A", "polygon": [[0, 0], [400, 0], [400, 600], [0, 600]]},
            {"name": "B", "polygon": [[400, 0], [800, 0], [800, 600], [400, 600]], "layers": 2},
        ],
        "schedule": {"smoothing_window": 1},
        "debug": {"annotated_path": str(tmp_path / "latest.jpg")},
    })
    c["google_sheets"]["pending_file"] = str(tmp_path / "pending.jsonl")
    return c


def test_scan_once_end_to_end(cfg, tmp_path):
    client = FakeClient()
    writer = SheetsWriter(cfg["google_sheets"], client=client)
    det = ContourDetector(3000, 50000, 0.75)
    scanner = InventoryScanner(cfg, FakeCamera([synthetic_floor(7)]), det, writer)

    r = scanner.scan_once()
    # 7 個箱子：x=40,190,340 在 A 區（中心 < 400），x=490,640 在 B 區；第二排 2 個在 A 區
    assert r.raw_counts == {"A": 5, "B": 2}
    assert r.counts == {"A": 5, "B": 4}  # B 區 2 層
    current = client.sheets["即時庫存"].values
    assert current[0] == CURRENT_HEADER
    assert current[-1][:2] == ["合計", 9]
    assert len(client.sheets["掃描紀錄"].appended) == 1 + 2  # 標題 + 2 區
    assert (tmp_path / "latest.jpg").exists()


def test_sheets_offline_buffer_then_flush(cfg):
    gs = cfg["google_sheets"]
    offline = SheetsWriter(gs, client=FakeClient(fail=True))
    assert offline.write("t1", {"A": 1}, {"A": 1}) is False
    assert offline.write("t2", {"A": 2}, {"A": 2}) is False
    with open(gs["pending_file"], encoding="utf-8") as f:
        assert [json.loads(line)[0] for line in f] == ["t1", "t2"]

    client = FakeClient()
    online = SheetsWriter(gs, client=client)
    assert online.write("t3", {"A": 3}, {"A": 3}) is True
    log_rows = client.sheets["掃描紀錄"].appended[1:]
    assert [r[0] for r in log_rows] == ["t1", "t2", "t3"]


def test_log_only_on_change(cfg):
    gs = dict(cfg["google_sheets"], log_only_on_change=True)
    client = FakeClient()
    w = SheetsWriter(gs, client=client)
    w.write("t1", {"A": 1}, {"A": 1})
    w.write("t2", {"A": 1}, {"A": 1})
    w.write("t3", {"A": 2}, {"A": 2})
    assert [r[0] for r in client.sheets["掃描紀錄"].appended[1:]] == ["t1", "t3"]
