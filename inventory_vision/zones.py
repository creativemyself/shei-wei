"""地板區域（ROI）判斷、各區計數與平滑化。"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection

WHOLE_FRAME = "全區"


@dataclass
class Zone:
    name: str
    polygon: np.ndarray | None  # None = 整個畫面
    layers: int = 1

    def contains(self, x: float, y: float) -> bool:
        if self.polygon is None:
            return True
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0


def build_zones(zone_cfgs: list[dict]) -> list[Zone]:
    if not zone_cfgs:
        return [Zone(WHOLE_FRAME, None, 1)]
    zones = []
    for z in zone_cfgs:
        poly = np.asarray(z["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        if len(poly) < 3:
            raise ValueError(f"區域 {z['name']} 的多邊形至少需要 3 個點")
        zones.append(Zone(str(z["name"]), poly, int(z.get("layers", 1))))
    return zones


def assign_to_zones(detections: list[Detection], zones: list[Zone], anchor: str = "center") -> dict[str, list[Detection]]:
    """每個偵測框只歸入第一個包含其錨點的區域；不在任何區域內的忽略（例如走道、牆面）。"""
    result: dict[str, list[Detection]] = {z.name: [] for z in zones}
    for d in detections:
        x, y = d.anchor(anchor)
        for z in zones:
            if z.contains(x, y):
                result[z.name].append(d)
                break
    return result


@dataclass
class CountSmoother:
    """保留最近 N 次的計數，回傳中位數，過濾短暫遮擋造成的誤差。"""

    window: int = 3
    _history: dict[str, deque] = field(default_factory=dict)

    def update(self, counts: dict[str, int]) -> dict[str, int]:
        out = {}
        for name, value in counts.items():
            h = self._history.setdefault(name, deque(maxlen=self.window))
            h.append(value)
            out[name] = int(round(statistics.median(h)))
        return out
