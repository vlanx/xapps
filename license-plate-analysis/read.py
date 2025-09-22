import easyocr
import cv2
import requests
from dotenv import load_dotenv
import os
import time

start_time = time.time()

load_dotenv()

reader = easyocr.Reader(['en'])
    
STREAM_URL = os.getenv("STREAM_URL")

NETWORK_APP_URL = os.getenv("NETWORK_APP_URL")

cap = cv2.VideoCapture(STREAM_URL)

end_time = time.time()

elapsed_time = end_time - start_time

data = {'log': f'Started capturing video stream ({elapsed_time})'}

url = NETWORK_APP_URL + "/log"
print(f'Started capturing video stream ({elapsed_time})')
response = requests.post(url, json=data)

def process_image(image):
    print('Getting text...')
    result = reader.readtext(image)

    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        print(f'Text: {text}, Probability: {prob}')
        
    return image

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

ret, frame = cap.read()

start_time = time.time()
text = process_image(frame)

end_time = time.time()
elapsed_time = end_time - start_time

data = {'log': f'Analyzed text in {elapsed_time}.'}

url = NETWORK_APP_URL + "/log"
print(f'Analyzed text in {elapsed_time}.')
response = requests.post(url, json=data)

data = {'log': f'Extracted text {text}.'}

url = NETWORK_APP_URL + "/log"
print(f'Extracted text in {text}.')
response = requests.post(url, json=data)
    
response = requests.get(NETWORK_APP_URL)

if not ret:
    print("Error: Could not read frame.")

print(text)

cap.release()
