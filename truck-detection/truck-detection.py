#!/usr/bin/env python3
# Continuous RTSP truck detector using Ultralytics YOLO and OpenCV.
#
# Features:
# - Streams frames from an RTSP (or file) source.
# - Shows a live window of annotated detections (if GUI available).
# - Opens/updates a secondary window with the latest frame that contains a 'truck'.
# - Robust loop with reconnect on failure.
# - Headless-friendly
#
# Usage examples:
#   python truck_detector.py --source "rtsp://user:pass@host/path"
#   python truck_detector.py --source sample.mp4 --conf 0.8

import time
import argparse
from datetime import datetime
from ultralytics import YOLO
import logging

# Window names
MAIN_WND = "YOLO view"
TRUCK_WND = "Truck detected"

# Truck index in the COCO dataset
TRUCKS = 7

# Logs
logging.basicConfig(level=logging.INFO)
logging.setLoggerClass(logging.Logger)
logger = logging.getLogger()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def has_truck(result) -> bool:
    # Use the model's class-name mapping to check for 'truck'
    if result.boxes is None or result.boxes.cls is None:
        return False

    # If there's a Truck (7th index of class names)
    detected_classes = result.boxes.cls.tolist()
    if TRUCKS in detected_classes:
        logger.info("Truck Detected")
        return True

    return False


def annotate_image(result):
    """Return an image with boxes/labels drawn (BGR for OpenCV)."""
    return result.plot()


def main():
    parser = argparse.ArgumentParser(
        description="RTSP truck detector (Ultralytics YOLO + OpenCV)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="RTSP URL, video file path, or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics model to use (e.g. yolo11n.pt, yolo11s.pt, custom.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument(
        "--reconnect-wait",
        type=float,
        default=2.0,
        help="Seconds to wait before reconnecting the stream on failure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for Ultralytics (e.g., 'cpu', '0' for first CUDA GPU).",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Whether to output the verbose log of the prediction method",
    )

    args = parser.parse_args()

    # Stream source
    source = args.source

    # Load model
    model = YOLO(args.model)

    # Continuous loop with reconnect on error/end
    while True:
        try:
            # stream=True yields results per frame
            gen = model.predict(
                source=source,
                stream=True,
                conf=args.conf,
                device=args.device,
                verbose=args.verbose,
            )

            for result in gen:
                # Truck logic
                if has_truck(result):
                    logger.info("sending network command")

            # If generator finishes (file ends or stream breaks), reconnect.
            logger.warning("[stream] ended or interrupted; attempting reconnect...")
            time.sleep(args.reconnect_wait)

        except KeyboardInterrupt:
            logger.info("[exit] interrupted by user")
            break
        except Exception as ex:
            logger.error(f"[warn] exception: {ex}")
            time.sleep(args.reconnect_wait)
            logger.error("[retry] attempting to reconnect...")


if __name__ == "__main__":
    main()
