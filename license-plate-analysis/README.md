# License Plate Analysis (Ultralytics YOLO & EasyOCR)

RTSP truck license plate detector and analysis with two run modes:
- **Headless** for servers/containers (no windows, logs only)
- **GUI** for local debugging (OpenCV windows)

Ultralytics YOLOv11 + CPU PyTorch + EasyOCR. Designed for Docker/Kubernetes.

## TL;DR

#### Docker container

```bash
# Run (headless)
docker run --name truck --rm ghcr.io/vlanx/license-plate-analysis:v0.1.0 --source rtsp://10.255.35.86/stream1 --conf 0.60

```

> If using Kubernetes, set flags via `args:` (example below).

---

## Two Modes:

- **Headless**: production-style, no GUI calls, safe in containers, logs to stdout.
- **GUI**: for local tuning/visual checks (`cv2.imshow`), not meant for servers.

Repo entries :

```
license-plate-analysis.py        # headless entrypoint (ENTRYPOINT in Dockerfile)
license-plate-analysis-gui.py        # local debugging (uses cv2 windows)
```

---

## Requirements

- Python ≥ 3.9 (or Docker)
- Models: YOLO11 custom license plate models from: [Link](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/blob/main/license_plate_detector.pt)
- You can also use other models from Hugging Face: [Link](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/tree/main)

#### I've disabled EasyOCR from downloading its models at runtime. I download them manually and import them in the code

The models are here: [Link](https://www.jaided.ai/easyocr/modelhub/)

You need the `english_g2` and the `CRAFT` models. I put them in directory `easy-ocr-models/`

In the code I load them via:

```python
# Easy OCR config
reader = easyocr.Reader(
    ["en"], gpu=False, download_enabled=False, model_storage_directory="easy-ocr-models"
)
```

### Runtime GUI(lean)

Install with:

```bash
pip install -r requirements_gui.txt
```

---

### Run (GUI, local only)**

```bash
python3 license-plate-analysis-gui.py   --source=rtsp://10.255.35.86/stream1 --conf=0.6
```

---

## Kubernetes

Use `args:` for flags; you can also map envs to args if you prefer.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: license-plate-analysis
spec:
  replicas: 1
  selector:
    matchLabels: { app: license-plate-analysis}
  template:
    metadata:
      labels: { app: license-plate-analysis }
    spec:
      containers:
      - name: detector
        image: ghcr.io/vlanx/license-plate-analysis:latest
        # image ENTRYPOINT runs truck-detection.py
        args:
          - "--source=rtsp://10.255.35.86/stream2"
          - "--conf=0.8"
        resources:
          limits:
            cpu: "2"
            memory: "1Gi"
        # If your RTSP source is on the node/LAN, host networking can help:
        # hostNetwork: true
        # securityContext:
        #   runAsUser: 65532
        #   runAsGroup: 65532
```

---

## Configuration (flags)

| Flag               | Type   | Default      | Notes                                                     |
|--------------------|--------|--------------|-----------------------------------------------------------|
| `--source`         | str    | **required** | RTSP URL / file path / camera index                       |
| `--model`          | str    | `yolo11n.pt` | e.g., `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, custom    |
| `--conf`           | float  | `0.5`        | Confidence threshold (0..1)                               |
| `--device`         | str    | `cpu/None`   | `'cpu'` or CUDA device `'0'` (GPU image required)         |
| `--verbose`        | bool   | `False`      | Extra logs from predictor                                 |
| `--reconnect-wait` | float  | `2.0`        | Seconds to wait before RTSP reconnect                     |

---

## Dev notes

- Default container ENTRYPOINT is headless: `license-plate-analysis.py`.
- Local GUI script remains separate: `license-plate-analysis-gui.py`.
- Keep GUI-only code out of the headless path to avoid accidental imports.
