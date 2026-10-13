#!/usr/bin/env bash
# 一鍵設定鏡頭節點（Raspberry Pi Zero 2 W + IMX219，Raspberry Pi OS Lite 64-bit）
#
# 用法（在節點上、專案資料夾內執行）：
#   bash deploy/setup_node.sh cam01 http://192.168.1.10:8000 <token>
#
# 會做的事：安裝套件 → 產生 config/node.yaml → 安裝開機自動執行的 systemd 服務
set -euo pipefail

CAMERA_ID="${1:?請提供鏡頭編號，例如 cam01}"
CENTRAL_URL="${2:?請提供中央網址，例如 http://192.168.1.10:8000}"
TOKEN="${3:?請提供 token（與中央 central.yaml 的 server.token 相同）}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="$(id -un)"
cd "${PROJECT_DIR}"

echo "==> 安裝系統套件（picamera2、OpenCV、PyYAML）"
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-yaml

echo "==> 產生 config/node.yaml（鏡頭 ${CAMERA_ID}）"
sed -e "s|camera_id: cam01|camera_id: ${CAMERA_ID}|" \
    -e "s|central_url: .*|central_url: ${CENTRAL_URL}|" \
    -e "s|token: .*|token: \"${TOKEN}\"|" \
    "${PROJECT_DIR}/config/node.example.yaml" > "${PROJECT_DIR}/config/node.yaml"

echo "==> 測試鏡頭與上傳"
if ! python3 -m inventory_vision.node -c "${PROJECT_DIR}/config/node.yaml" --once; then
  echo "!! 測試上傳失敗：請確認鏡頭排線、中央已啟動、網址與 token 正確。服務仍會安裝，修正後會自動恢復。"
fi

echo "==> 安裝 systemd 服務"
sed -e "s|__USER__|${RUN_USER}|g" -e "s|__DIR__|${PROJECT_DIR}|g" \
    "${PROJECT_DIR}/deploy/inventory-node.service" | sudo tee /etc/systemd/system/inventory-node.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now inventory-node
echo "完成！查看紀錄：journalctl -u inventory-node -f"
