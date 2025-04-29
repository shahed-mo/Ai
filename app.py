from fastapi import FastAPI
import asyncio
import cv2
from ultralytics import YOLO
import requests

app = FastAPI()

# تحميل موديل YOLO
model = YOLO("best.pt")

# هنا رابط الكاميرا بعد ngrok
stream_url = "https://0c4a-154-183-138-118.ngrok-free.app/video"

@app.on_event("startup")
async def start_detection():
    while True:
        cap = cv2.VideoCapture(stream_url)
        ret, frame = cap.read()
        cap.release()

        if ret:
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

        await asyncio.sleep(5)  # كل 5 ثواني هيبعت الداتا

@app.get("/")
def root():
    return {"message": "Camera detection is live!"}
