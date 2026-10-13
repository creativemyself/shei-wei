import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from inventory_vision.central import (
    CURRENT_HEADER, LOG_HEADER, STATUS_NEVER, STATUS_OK, STATUS_STALE,
    CentralProcessor, FrameStore, make_handler,
)
from inventory_vision.config import CENTRAL_DEFAULTS, NODE_DEFAULTS, _merge
from inventory_vision.detector import ContourDetector
from inventory_vision.node import CameraNode, upload
from inventory_vision.sheets import SheetsWriter
from inventory_vision.timing import seconds_until_next_tick

from test_pipeline import FakeClient, synthetic_floor

TOKEN = "test-token"


def test_seconds_until_next_tick():
    assert seconds_until_next_tick(60, 0, now=120.0) == 60
    assert seconds_until_next_tick(60, 0, now=125.0) == 55
    assert seconds_until_next_tick(60, 20, now=125.0) == 15
    assert seconds_until_next_tick(60, 20, now=141.0) == 59  # 下一次是 200（每分鐘的 :20）


@pytest.fixture
def central(tmp_path):
    cfg = _merge(CENTRAL_DEFAULTS, {
        "server": {"token": TOKEN},
        "schedule": {"smoothing_window": 1, "stale_seconds": 150},
        "cameras": [
            {"id": "cam01", "name": "A排", "zones": [
                {"name": "左", "polygon": [[0, 0], [400, 0], [400, 600], [0, 600]]},
                {"name": "右", "polygon": [[400, 0], [800, 0], [800, 600], [400, 600]], "layers": 2},
            ]},
            {"id": "cam02", "name": "B排"},
        ],
        "debug": {"image_dir": str(tmp_path / "cameras")},
    })
    cfg["google_sheets"]["pending_file"] = str(tmp_path / "pending.jsonl")
    client = FakeClient()
    writer = SheetsWriter(cfg["google_sheets"], client=client, current_header=CURRENT_HEADER, log_header=LOG_HEADER)
    store = FrameStore()
    proc = CentralProcessor(cfg, ContourDetector(3000, 50000, 0.75), store, writer)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proc, store, TOKEN, 10 * 1024 * 1024))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield proc, url, client, tmp_path
    server.shutdown()


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame


def node_for(camera_id, url, frame, token=TOKEN):
    cfg = _merge(NODE_DEFAULTS, {"node": {"camera_id": camera_id, "central_url": url, "token": token, "retries": 1}})
    return CameraNode(cfg, FakeCamera(frame))


def test_node_upload_and_central_batch_write(central):
    proc, url, client, tmp_path = central
    assert node_for("cam01", url, synthetic_floor(7)).capture_and_upload()

    current, logs = proc.process_once()
    by_zone = {(r[0], r[2]): r for r in current}
    # 左半邊 5 個箱子；右半邊 2 個 × 2 層 = 4
    assert by_zone[("cam01", "左")][3:6] == [5, 5, STATUS_OK]
    assert by_zone[("cam01", "右")][3:6] == [4, 2, STATUS_OK]
    assert by_zone[("cam02", "全區")][5] == STATUS_NEVER
    assert current[-1][0] == "合計" and current[-1][3] == 9 and current[-1][5] == "1/2 台正常"
    assert [(r[1], r[3], r[4]) for r in logs] == [("cam01", "左", 5), ("cam01", "右", 4)]

    sheet = client.sheets["即時庫存"].values
    assert sheet[0] == CURRENT_HEADER and len(sheet) == 1 + 3 + 1
    assert (tmp_path / "cameras" / "cam01" / "latest.jpg").exists()

    with urllib.request.urlopen(f"{url}/status?token={TOKEN}") as r:
        status = json.loads(r.read())
    assert status["cam01"]["counts"] == {"左": 5, "右": 4}
    with urllib.request.urlopen(f"{url}/image/cam01?token={TOKEN}") as r:
        assert r.read(2) == b"\xff\xd8"


def test_stale_camera_keeps_last_count_and_skips_log(central):
    proc, url, _, _ = central
    node_for("cam01", url, synthetic_floor(3)).capture_and_upload()
    t = proc.store.get("cam01").received_at
    proc.process_once(now=t + 1)
    current, logs = proc.process_once(now=t + 1000)
    row = next(r for r in current if r[0] == "cam01" and r[2] == "左")
    assert row[3] == 3 and row[5] == STATUS_STALE
    assert logs == []


def test_upload_rejects_bad_token_unknown_camera_and_non_jpeg(central):
    _, url, _, _ = central
    jpeg = b"\xff\xd8fake"
    with pytest.raises(urllib.error.HTTPError) as e:
        upload(url, "cam01", "wrong", jpeg, 0, 5)
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as e:
        upload(url, "cam99", TOKEN, jpeg, 0, 5)
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        upload(url, "cam01", TOKEN, b"GIF89a", 0, 5)
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{url}/status")
    assert e.value.code == 401


def test_node_returns_false_when_central_unreachable():
    assert node_for("cam01", "http://127.0.0.1:9", synthetic_floor(1)).capture_and_upload() is False
