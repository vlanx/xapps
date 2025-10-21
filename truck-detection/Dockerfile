FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \ 
    QT_QPA_PLATFORM=offscreen

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

# Copy the rest of the directory. Dockerignore makes it so that no junk is passed.
COPY truck-detection.py yolo11m.pt .

# run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python3", "truck-detection.py"]
