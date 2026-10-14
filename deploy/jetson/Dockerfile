# 中央伺服器（Jetson Orin Nano，JetPack 6）
# 以 Ultralytics 官方 Jetson 映像為基底：已內建支援 GPU 的 PyTorch、TensorRT 與 OpenCV，
# 避免在 Jetson 上手動安裝 PyTorch 時版本不相容。
FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app
# tzdata：容器內若無系統時區資料，Asia/Taipei 也能正常換算
RUN pip install --no-cache-dir "gspread>=6.0" "google-auth>=2.20" tzdata

COPY inventory_vision /app/inventory_vision
COPY tools /app/tools

ENV PYTHONUNBUFFERED=1 TZ=Asia/Taipei
CMD ["python3", "-m", "inventory_vision.central", "-c", "config/central.yaml"]
