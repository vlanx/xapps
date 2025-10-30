# Truck Detection (Ultralytics YOLO)

RTSP truck detector with two run modes:
- **Headless** for servers/containers (no windows, logs only)
- **GUI** for local debugging (OpenCV windows)

Ultralytics YOLOv11 + CPU PyTorch. Designed for Docker/Kubernetes.

## TL;DR

#### Docker container

```bash
# Run (headless)
docker run --rm --name truc_detection truck-detection:latest --source=rtsp://10.255.35.86/stream2 --model=yolo11m.pt --conf=0.8
```

> If using Kubernetes, set flags via `args:` (example below).

---

## Two Modes:

- **Headless**: production-style, no GUI calls, safe in containers, logs to stdout.
- **GUI**: for local tuning/visual checks (`cv2.imshow`), not meant for servers.

Repo entries :

```
truck-detection.py        # headless entrypoint (ENTRYPOINT in Dockerfile)
truck-detection-gui.py    # local debugging (uses cv2 windows)
```

---

## Requirements

- Python ≥ 3.9 (or Docker)
- Models: YOLO11 (e.g., `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`)

### Runtime GUI(lean)

Install with:

```bash
pip install -r requirements_gui.txt
```

---

### Run (GUI, local only)**

```bash
python truck-detection-gui.py   --source=rtsp://10.255.35.86/stream2   --model=yolo11m.pt   --conf=0.8
```

---

## Kubernetes

Use `args:` for flags; you can also map envs to args if you prefer.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truck-detection
spec:
  replicas: 1
  selector:
    matchLabels: { app: truck-detection }
  template:
    metadata:
      labels: { app: truck-detection }
    spec:
      containers:
      - name: detector
        image: ghcr.io/vlanx/truck-detection:lateset
        # image ENTRYPOINT runs truck-detection.py
        args:
          - "--source=rtsp://10.255.35.86/stream2"
          - "--model=yolo11m.pt"
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

- Default container ENTRYPOINT is headless: `truck-detection.py`.
- Local GUI script remains separate: `truck-detection-gui.py`.
- Keep GUI-only code out of the headless path to avoid accidental imports.
