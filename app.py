from fastapi import FastAPI
import asyncio
import cv2
from ultralytics import YOLO
import httpx
import logging

app = FastAPI()

# إعداد اللوجات
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# تحميل الموديل
model = YOLO("best.pt")

# روابط
stream_url = "https://0c4a-154-183-138-118.ngrok-free.app/video"
backend_url = "http://farmsmanagement.runasp.net/api/Notifiactions/CreateNotification"
headers = {'Content-Type': 'application/json'}

async def detect_and_notify():
    while True:
        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logging.error("❌ Failed to connect to the camera stream. Retrying in 5 seconds...")
                await asyncio.sleep(5)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("⚠️ Lost connection to the camera stream. Reconnecting...")
                    cap.release()
                    await asyncio.sleep(2)
                    break  # نطلع من اللوب الداخلي ونحاول نعيد فتح الكاميرا

                results = model.predict(frame)
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

                if detections:
                    logging.info(f"🔍 Detections found: {detections}")
                    payload = {
                        "detections": detections
                    }
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(backend_url, json=payload, headers=headers)
                            if response.status_code == 200:
                                logging.info("✅ Notification sent successfully.")
                            else:
                                logging.error(f"❌ Failed to send notification. Status code: {response.status_code}")
                    except Exception as e:
                        logging.error(f"❌ Error sending to backend: {e}")

                await asyncio.sleep(5)  # كل 5 ثواني

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def start_detection():
    asyncio.create_task(detect_and_notify())

@app.get("/")
def root():
    return {"message": "Camera detection is live!"}
