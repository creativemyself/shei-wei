"""在畫面上點選地板區域（多邊形），輸出可貼進 config.yaml 的 zones 設定。

用法：
    # 直接從攝影機拍一張（需接螢幕或 VNC）
    python tools/calibrate_roi.py -c config/config.yaml
    # 或用事先拍好的照片（可在開發用電腦的 VS Code 中執行）
    python tools/calibrate_roi.py --image data/latest_raw.jpg

操作：
    左鍵   新增頂點
    右鍵   刪除上一個頂點
    n      完成目前區域，開始下一個區域
    s      儲存並輸出 YAML（同時寫入 data/zones.yaml）
    q/Esc  離開
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_vision.camera import open_camera  # noqa: E402
from inventory_vision.config import load_config  # noqa: E402

WINDOW = "calibrate ROI"
COLORS = [(255, 160, 0), (0, 200, 255), (200, 0, 255), (0, 255, 120)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/config.yaml")
    ap.add_argument("--image", help="使用圖片檔而非攝影機")
    ap.add_argument("--out", default="data/zones.yaml")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"無法讀取圖片：{args.image}")
    else:
        cfg = load_config(args.config)
        with open_camera(cfg["camera"]) as cam:
            frame = cam.read()
        Path("data").mkdir(exist_ok=True)
        cv2.imwrite("data/latest_raw.jpg", frame)

    zones: list[list[list[int]]] = []
    current: list[list[int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        img = frame.copy()
        for i, poly in enumerate(zones + [current]):
            color = COLORS[i % len(COLORS)]
            pts = np.array(poly, np.int32)
            if len(pts):
                cv2.polylines(img, [pts], poly is not current, color, 3)
                for p in poly:
                    cv2.circle(img, tuple(p), 6, color, -1)
                cv2.putText(img, f"zone {i + 1}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n") and len(current) >= 3:
            zones.append(current.copy())
            current.clear()
        elif key == ord("s"):
            if len(current) >= 3:
                zones.append(current.copy())
                current.clear()
            data = {"zones": [{"name": f"區域{i + 1}", "layers": 1, "polygon": z} for i, z in enumerate(zones)]}
            text = yaml.safe_dump(data, allow_unicode=True, default_flow_style=None, sort_keys=False)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            print(text)
            print(f"已寫入 {args.out}，請將內容貼到 config/config.yaml 的 zones 區塊並修改名稱。")
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
