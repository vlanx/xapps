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
from ultralytics import YOLO
import logging, sys, os
import cv2
import aiohttp, asyncio, threading
from dotenv import load_dotenv

# Truck index in the COCO dataset
TRUCKS = 7

# Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(name)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logging.setLoggerClass(logging.Logger)
logger = logging.getLogger(__name__)
logger.propagate = True


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


# API Client session to send requests
loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()

# Create session
session = asyncio.run_coroutine_threadsafe(_make_session(), loop).result()

# Load credentials.
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASS = os.getenv("PASS")
ENDPOINT = os.getenv("ENDPOINT")


def has_truck(result) -> bool:
    # Use the model's class-name mapping to check for 'truck'
    if result.boxes is None or result.boxes.cls is None:
        return False

    # If there's a Truck (7th index of class names)
    detected_classes = result.boxes.cls.tolist()
    if TRUCKS in detected_classes:
        return True

    return False


def annotate_image(result):
    """Return an image with boxes/labels drawn (BGR for OpenCV)."""
    return result.plot()


def save_image(image):
    tmp = "/frames/latest.jpg"
    cv2.imwrite(str(tmp), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    logger.info(f"Saved image to {str(tmp)}")


async def activate_antennas(session):
    # Activate Antenna 352. It's already at max TX/RX power.
    async with session.post(
        url=f"http://{ENDPOINT}/antena/253",
        json={"antenasIDs": [352]},
        auth=aiohttp.BasicAuth(login=str(USERNAME), password=str(PASS)),
    ) as response:
        r = await response.json()
        logger.info("Enabled Antenna Cell 352 status(%s)", r["description"])


async def bump_power(session):
    # Bump Antenna 351 to max TX/RX power.
    async with session.patch(
        url=f"http://{ENDPOINT}/antena/253/control",
        json={"antenasIDs": [351], "Power": 242, "SU - MIMO": True},
        auth=aiohttp.BasicAuth(login=str(USERNAME), password=str(PASS)),
    ) as response:
        r = await response.json()
        logger.info(
            "Bumped Antenna Cell 351 to max power (status %s)", r["description"]
        )


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

    # This is to help debounce the truck detection algorithm.
    # When we detect a truck a send a network command, we enable the `detected` flag and do not send
    # the command again while we see the same truck. What happens is sometimes, even when detecting the same truck,
    # not all frames are detected as containing a truck, hence the debounce limit.
    detected = False
    misses = 0
    max_misses = 25

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
                    # Annotate & save
                    if not detected:
                        logger.info("Truck Detected")
                        logger.info("Sending network command")
                        img_annot = annotate_image(result)
                        save_image(img_annot)
                        fire_network_comm(activate_antennas(session), "activate")
                        # Can't do this now, Slice manager cannot handle multiple requests
                        fire_network_comm(bump_power(session), "bump")
                        detected = True
                    else:
                        logger.info("Same truck still in frame")
                    misses = 0
                else:
                    misses += 1
                    if misses >= max_misses:
                        if misses % 50 == 0:
                            logger.info(
                                f"No truck in the last {misses} frames. Assuming no truck in sight."
                            )
                            # If we have previously detected a truck, now it is gone. we can deactivate the antennas.
                            # This blocks sending the deactivation command every time there isn't a truck in sight.
                        if detected:
                            fire_network_comm(
                                deactivate_antennas(session), "deactivate"
                            )
                            # Can't do this now, Slice manager cannot handle multiple requests
                            fire_network_comm(downgrade_power(session), "downgrade")
                        # Reset the detected flag
                        detected = False

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
    asyncio.run_coroutine_threadsafe(session.close(), loop).result()
    loop.call_soon_threadsafe(loop.stop)
    logger.info("[exit] Closed asyncio and threads")
