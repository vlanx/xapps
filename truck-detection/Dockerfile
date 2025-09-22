FROM amd64/python:3.9-slim-buster
WORKDIR /app

COPY . /app

ENV STREAM_URL=rtsp://192.168.94.142:554/stream2
ENV NETWORK_APP_URL=localhost:3000


RUN apt update && apt install ffmpeg libsm6 libxext6  -y
RUN apt install net-tools

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "video.py"]