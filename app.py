from fastapi import FastAPI, Request
from pydantic import BaseModel
import cv2
from ultralytics import YOLO
import requests

app = FastAPI()
model = YOLO("best.pt")

class StreamInput(BaseModel):
    stream_url: str

@app.get("/")
def root():
    return {"message": "Camera detection is live!"}

@app.post("/detect")
def detect(input: StreamInput):
    cap = cv2.VideoCapture(input.stream_url)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "فشل في قراءة الفيديو من الكاميرا"}

    results = model(frame)
    names = model.names
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "class": names[cls],
                "confidence": round(conf, 3)
            })

    try:
        backend_url = "http://farmsmanagement.runasp.net/api/Notifiactions/CreateNotification"
        payload = {
            "detections": detections
        }
        headers = {'Content-Type': 'application/json'}
        requests.post(backend_url, json=payload, headers=headers)
    except Exception as e:
        print(f"Error sending to backend: {e}")

    return {"detections": detections}
