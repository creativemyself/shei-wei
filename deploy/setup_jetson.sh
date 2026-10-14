#!/usr/bin/env bash
# 一鍵設定中央伺服器（Jetson Orin Nano，JetPack 6）
#
# 用法（在 Jetson 上、專案資料夾內執行）：
#   bash deploy/setup_jetson.sh            # 安裝並啟動
#   bash deploy/setup_jetson.sh export     # 把 models/boxes.pt 轉成 TensorRT（models/boxes.engine）
#
# 中央程式在 Docker 容器中執行，開機自動啟動（--restart unless-stopped）。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"
IMAGE=ai-inventory-central
NAME=ai-inventory-central

if [[ "${1:-}" == "export" ]]; then
  [[ -f models/boxes.pt ]] || { echo "找不到 models/boxes.pt，請先放入訓練好的模型"; exit 1; }
  echo "==> 匯出 TensorRT 模型（約 5～15 分鐘）"
  sudo docker run --rm --runtime nvidia --ipc=host -v "${PROJECT_DIR}/models:/app/models" "${IMAGE}" \
    yolo export model=models/boxes.pt format=engine half=True imgsz=640
  echo "完成：models/boxes.engine。請確認 config/central.yaml 的 detector.model_path 指向它，然後執行："
  echo "  sudo docker restart ${NAME}"
  exit 0
fi

echo "==> 檢查 JetPack 版本"
if grep -q "R36" /etc/nv_tegra_release 2>/dev/null; then
  head -1 /etc/nv_tegra_release
else
  echo "!! 本腳本適用 JetPack 6（L4T R36）。目前版本：$(head -1 /etc/nv_tegra_release 2>/dev/null || echo 未知)"
  exit 1
fi

echo "==> 檢查 Docker"
if ! command -v docker >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y nvidia-container   # JetPack 的 Docker + NVIDIA 容器執行環境
fi
sudo systemctl enable --now docker

echo "==> 準備設定檔與資料夾"
mkdir -p data models credentials
if [[ ! -f config/central.yaml ]]; then
  TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
  sed "s|token: \"請改成一組長的隨機字串\"|token: \"${TOKEN}\"|" config/central.example.yaml > config/central.yaml
  echo "   已建立 config/central.yaml，節點用的 token：${TOKEN}"
fi

echo "==> 建立 Docker 映像（第一次需下載約數 GB，請耐心等候）"
sudo docker build -f deploy/jetson/Dockerfile -t "${IMAGE}" .

echo "==> 確認 GPU 可用"
sudo docker run --rm --runtime nvidia "${IMAGE}" python3 -c \
  "import torch; ok = torch.cuda.is_available(); print('GPU 可用：', ok); raise SystemExit(0 if ok else 1)"

if [[ ! -f models/boxes.pt && ! -f models/boxes.engine ]]; then
  echo "!! models/ 內還沒有模型。訓練完成前，可先把 config/central.yaml 的 detector.type 改成 contour 測試。"
fi
if grep -q "請貼上試算表 ID" config/central.yaml; then
  echo "!! 尚未設定 Google 試算表 ID；在填好之前，可先把 google_sheets.enabled 改成 false 測試。"
fi

echo "==> 啟動中央伺服器（開機自動啟動）"
sudo docker rm -f "${NAME}" >/dev/null 2>&1 || true
sudo docker run -d --name "${NAME}" --restart unless-stopped \
  --runtime nvidia --ipc=host --network host \
  -v "${PROJECT_DIR}/config:/app/config:ro" \
  -v "${PROJECT_DIR}/credentials:/app/credentials:ro" \
  -v "${PROJECT_DIR}/models:/app/models" \
  -v "${PROJECT_DIR}/data:/app/data" \
  "${IMAGE}"

echo "完成！查看紀錄：sudo docker logs -f ${NAME}"
echo "修改 config/central.yaml 後重新啟動：sudo docker restart ${NAME}"
echo "更新程式：git pull && bash deploy/setup_jetson.sh"
