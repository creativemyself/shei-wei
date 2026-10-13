"""對齊時鐘的排程：所有節點都在每分鐘整點（:00）同時拍照，中央在 :20 統一辨識。

各台裝置請開啟 NTP 校時（Raspberry Pi OS / JetPack 預設已開啟）。
"""
from __future__ import annotations

import threading
import time


def seconds_until_next_tick(interval: float, offset: float = 0.0, now: float | None = None) -> float:
    now = time.time() if now is None else now
    next_tick = ((now - offset) // interval + 1) * interval + offset
    return next_tick - now


def wait_next_tick(interval: float, offset: float, stop: threading.Event) -> bool:
    """等到下一個時間點；收到停止訊號時回傳 False。"""
    return not stop.wait(seconds_until_next_tick(interval, offset))
