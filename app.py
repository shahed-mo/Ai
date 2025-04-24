import cv2
import requests
from ultralytics import YOLO
import time

# تحميل النموذج المدرب
model = YOLO(r"E:\Ai\best.pt")

# IP لكاميرا الموبايل (مثال IP Webcam app)
ip_camera_url = "http://192.168.1.3:8080/video"  # غيّره حسب شبكتك

# رابط API للباك اند
api_url = "http://farmsmanagement.runasp.net/api/Notifiactions/CreateNotification"

# ترجمة الفئات
label_translation = {
    "Sick": "مريضة",
    "Dead": "ميتة",
    "Healthy": "سليمة"
}

# كاش لمنع تكرار نفس الإشعار
last_label = None
last_sent_time = 0

# افتح الكاميرا
cap = cv2.VideoCapture(ip_camera_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ فشل في الاتصال بالكاميرا")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        classes = result.boxes.cls

        for i in range(len(boxes)):
            conf = float(scores[i])
            class_id = int(classes[i])
            label = model.names[class_id]
            arabic_label = label_translation.get(label, label)

            print(f"تم اكتشاف: {arabic_label} بثقة: {conf:.2f}")

            # إرسال إشعار في حالة ثقة عالية وتغيير في النوع
            if conf > 0.6 and (label != last_label or time.time() - last_sent_time > 10):
                if label in label_translation:
                    body = f"فرخة {arabic_label}"
                    data = {
                        "body": body,
                        "userId": 24,
                        "barnId": 3,
                        "isRead": False
                    }
                    try:
                        response = requests.post(api_url, json=data)
                        print(f"🚨 تم إرسال إشعار: {body} ✅")
                        print(response.json())
                        last_label = label
                        last_sent_time = time.time()
                    except Exception as e:
                        print(f"❌ خطأ في الإرسال: {e}")

    time.sleep(0.5)  # تقليل الضغط

cap.release()
