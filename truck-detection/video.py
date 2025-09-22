import cv2
import numpy as np
import time
import requests
from dotenv import load_dotenv

import queue, threading, time
import os
import aiohttp

start_time = time.time()

load_dotenv()

stream_url = os.getenv("STREAM_URL")

NETWORK_APP_URL = os.getenv("NETWORK_APP_URL")

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  def _reader(self):
    while True:
        ret, frame = self.cap.read()
        if not ret:
            break
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(frame)

  def read(self):
    return self.q.get()

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.config')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f]

cap = VideoCapture(stream_url)

end_time = time.time()

elapsed_time = end_time - start_time

data = {'log': f'Started capturing video stream ({elapsed_time})'}

url = NETWORK_APP_URL + "/log"
print(f'Started capturing video stream ({elapsed_time})')
#response = requests.post(url, json=data)

detected = False

def process_image(image):
    print('Analyzing frame')
    global detected
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layer_names)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'car':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                color = (0, 255, 0) 
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                label = f'{classes[class_id]}: {confidence:.2f}'
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if detected == False:
                    print('Making request')
                    response = requests.get(NETWORK_APP_URL)
                    detected = True
                    
    return image

while True:
    frame = cap.read()

    frame = process_image(frame)

    print('received')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
