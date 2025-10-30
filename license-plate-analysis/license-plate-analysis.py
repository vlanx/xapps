#!/usr/bin/env python3
# Continuous RTSP License plate detector and analysis using Ultralytics YOLO and EasyOCR.
#
# Features:
# - Streams frames from an RTSP (or file) source.
# - Shows a live window of annotated detections.
# - Opens/updates a secondary window with the latest frame that contains a 'license plate'.
# - Robust loop with reconnect on failure.
# - Headless-friendly
# - Detect and reads the license plate captured
#
# Usage examples:
#   python license-plate-analysis-gui.py --source "rtsp://user:pass@host/path"
#   python license-plate-analysis-gui.py --source sample.mp4 --conf 0.8

from collections import defaultdict
from dotenv import load_dotenv
from ultralytics import YOLO
from rich.logging import RichHandler
import time
import argparse
import logging, os
import cv2
import aiohttp, asyncio, threading
import easyocr

# ---- OTel setup (code-only) -------------------------------------------------
from opentelemetry.sdk.resources import Resource

from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter


# Load credentials.
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASS = os.getenv("PASS")
ENDPOINT = os.getenv("ENDPOINT")
LOKI = os.getenv("LOKI")


def setup_otel_log():
    # Machine logger (OTLP -> Loki)
    resource = Resource.create(
        {
            "service.name": "license-plate-analysis",
            "service.namespace": "nexus",
            "deployment.environment": "dev",
        }
    )
    provider = LoggerProvider(resource=resource)
    set_logger_provider(provider)

    exporter = OTLPLogExporter(  # HTTP exporter
        endpoint=f"http://{LOKI}/otlp/v1/logs",
        headers={"X-Scope-OrgID": "IT"},  # harmless in single-tenant
        timeout=10_000,
    )
    provider.add_log_record_processor(
        BatchLogRecordProcessor(
            exporter,
            max_queue_size=4096,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
            export_timeout_millis=10_000,
        )
    )

    otlp_handler = LoggingHandler(level=logging.INFO, logger_provider=provider)

    telemetry = logging.getLogger("telemetry")
    telemetry.setLevel(logging.INFO)
    telemetry.propagate = False
    telemetry.handlers[:] = [otlp_handler]

    return telemetry


# Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
    handlers=[RichHandler(show_time=False, rich_tracebacks=True)],
)
logging.setLoggerClass(logging.Logger)
logger = logging.getLogger(__name__)
logger.propagate = True

otel_logs = setup_otel_log()

# Easy OCR config
reader = easyocr.Reader(
    ["en"], gpu=False, download_enabled=False, model_storage_directory="easy-ocr-models"
)
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "  # Only possible characters


# Create one shared session for network commands
async def _make_session():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))


def fire_network_comm(coroutine, label):
    fut = asyncio.run_coroutine_threadsafe(coroutine, loop)
    fut.add_done_callback(
        lambda f: f.exception()
        and logger.warning("%s failed: %r", label, f.exception())
    )
    return fut


# Save the image to be displayed.
def save_image(image):
    tmp = "/frames/latest.jpg"
    cv2.imwrite(str(tmp), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    logger.info(f"Saved image to {str(tmp)}")


# API Client session to send requests
loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()

# Create session
session = asyncio.run_coroutine_threadsafe(_make_session(), loop).result()

# License plate class index:
LIC_PLATE = 0

# Plate text dict to get the highest detected string
plate_hits: dict[str, int] = defaultdict(int)  # auto-starts at 0
HITS_TO_LOCK = 3


def has_license_plate(result) -> bool:
    # Use the model's class-name mapping to check for 'truck'
    if result.boxes is None or result.boxes.cls is None:
        return False

    # If there's a license plate
    detected_classes = result.boxes.cls.tolist()
    if LIC_PLATE in detected_classes:
        return True

    return False


def get_bounding_box(result):
    img = result.orig_img  # original BGR image (np.ndarray)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy().astype(int)
    pad = 0.06  # ~6% padding around the box helps OCR
    dx = int((x2 - x1) * pad)
    dy = int((y2 - y1) * pad)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    # Generate the cropped image
    crop = img[y1:y2, x1:x2]

    return crop


def get_text_from_lp(image) -> str:
    crop_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)  # gentle denoise, preserves strokes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Gray scale
    gray = clahe.apply(gray)

    # 2) Use beam search (less greedy)
    r1 = reader.readtext(
        gray,
        detail=0,
        allowlist=ALLOW,
        decoder="beamsearch",
        contrast_ths=0.05,
        adjust_contrast=0.5,
    )

    # Concatenate all results of the text detection array
    text = "".join(r1)

    return text


def update_votes(plate_str: str) -> tuple[str | None, int]:
    """Increment hit count and optionally return (locked_plate, count)."""
    if not plate_str:
        return None, 0
    plate_hits[plate_str] += 1
    if plate_hits[plate_str] >= HITS_TO_LOCK:
        return plate_str, plate_hits[plate_str]
    return None, plate_hits[plate_str]


def current_winner() -> str | None:
    if not plate_hits:
        return None
    return max(plate_hits.items(), key=lambda kv: kv[1])[0]


def annotate_image(result):
    """Return an image with boxes/labels drawn (BGR for OpenCV)."""
    return result.plot()


async def deactivate_antennas(session):
    # Deactivate Antenna 352.
    async with session.delete(
        url=f"http://{ENDPOINT}/antena/253",
        json={"antenasIDs": [352]},
        auth=aiohttp.BasicAuth(login=str(USERNAME), password=str(PASS)),
    ) as response:
        if response.status == 204:
            logger.info("Disabled Antenna Cell 352 (204 No Content)")
        else:
            # Drain whatever came back (empty) and ignore it
            await response.read()
            logger.info("Disabled Antenna Cell 352 (status %s)", response.status)
            otel_logs.info("Disabled Antenna Cell 352 (status %s)", response.status)


async def downgrade_power(session):
    # Downgrade Antenna 351 to minimum TX/RX power.
    async with session.patch(
        url=f"http://{ENDPOINT}/antena/253/control",
        json={"antenasIDs": [351], "Power": 70, "SU - MIMO": False},
        auth=aiohttp.BasicAuth(login=str(USERNAME), password=str(PASS)),
    ) as response:
        r = await response.json()
        logger.info(
            "Downgraded Atenna Cell 351 to minimum power (status %s)", r["description"]
        )
        otel_logs.info(
            "Downgraded Atenna Cell 351 to minimum power (status %s)", r["description"]
        )


def main():
    parser = argparse.ArgumentParser(
        description="RTSP truck detector (Ultralytics YOLO)."
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
        default="license_plate_detector.pt",
        help="Ultralytics model to use (e.g. yolo11n.pt, yolo11s.pt, custom.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument(
        "--antenna",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to manage and activate the antennas",
    )
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
        action=argparse.BooleanOptionalAction,
        help="Whether to output the verbose log of the prediction method",
    )

    args = parser.parse_args()

    # Stream source
    source = args.source

    # Load model
    model = YOLO(args.model)

    # Number of missing frames to discard this detection phase.
    misses = 0
    max_misses = 50
    processing_done = False

    # Antenna flag. When disabled, the program will not make requests to alter the antenna state.
    antenna = args.antenna
    logger.info(
        f"[bold magenta]Antenna management: {antenna}[/]",
        extra={"markup": True},
    )

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
                # Detect license plate
                if has_license_plate(result):
                    if not processing_done:
                        # Retrieves the license plate crop
                        bb = get_bounding_box(result)
                        plate = get_text_from_lp(bb)
                        locked, cnt = update_votes(plate)
                        save_image(bb)

                        if locked:
                            logger.info(
                                f"[bold blue]--Locked License plate: {locked}--[/]",
                                extra={"markup": True},
                            )
                            otel_logs.info(f"License plate: {locked}")
                            plate_hits.clear()  # simple reset for the next vehicle
                            processing_done = True

                        # Reset the misses counter
                        misses = 0
                else:
                    misses += 1
                    if misses >= max_misses:
                        # Processing is done. We can disable the antennas.
                        if antenna and processing_done:
                            fire_network_comm(
                                deactivate_antennas(session), "deactivate"
                            )
                            fire_network_comm(downgrade_power(session), "downgrade")
                        if misses % 50 == 0:
                            logger.info(
                                f"No license plate in the last {misses} frames. Assuming no truck in sight."
                            )
                            otel_logs.info(
                                f"No license plate in the last {misses} frames. Assuming no truck in sight."
                            )
                        # Assume we're on to the next truck. Clear the dictionary of license plates
                        plate_hits.clear()  # simple reset for the next vehicle
                        processing_done = False

            # If generator finishes (file ends or stream breaks), reconnect.
            time.sleep(args.reconnect_wait)
            logger.warning("[stream] ended or interrupted; attempting reconnect...")

        except KeyboardInterrupt:
            logger.info("[exit] interrupted by user")
            break
        except Exception as ex:
            logger.warning(f"[warn] exception: {ex}")
            time.sleep(args.reconnect_wait)
            logger.warning("[retry] attempting to reconnect...")


if __name__ == "__main__":
    main()
    asyncio.run_coroutine_threadsafe(session.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop)
    logger.info("[exit] Closed asyncio and threads")
