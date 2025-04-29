import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# تحميل النموذج
model = YOLO("best.pt")

@app.route('/')
def home():
    return "YOLO Flask API is running!"

@app.route('/detect', methods=['GET'])
def detect():
    # الحصول على رابط كاميرا الموبايل
    stream_url = request.args.get('stream_url')
    if not stream_url:
        return jsonify({"error": "يرجى توفير رابط الكاميرا (stream_url)"}), 400

    # فتح الفيديو باستخدام الرابط المقدم
    cap = cv2.VideoCapture(stream_url)

    # محاولة التقاط أول إطار من الفيديو
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "فشل في قراءة الفيديو من الكاميرا"}), 500

    # تحليل الصورة باستخدام YOLO
    results = model(frame)
    names = model.names
    detections = []

    # استخراج النتائج من YOLO
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "class": names[cls],
                "confidence": round(conf, 3)
            })

    # إرسال النتائج إلى API الباك إند
    try:
        backend_url = "http://farmsmanagement.runasp.net/api/Notifiactions/CreateNotification"
        payload = {
            "detections": detections
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(backend_url, json=payload, headers=headers)
        
        # التحقق من الرد من الباك إند
        if response.status_code == 200:
            print("تم إرسال البيانات بنجاح للباك إند!")
        else:
            print(f"فشل في إرسال البيانات للباك إند. حالة الرد: {response.status_code}")
        
    except Exception as e:
        print(f"حدث خطأ أثناء إرسال البيانات للباك إند: {e}")

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(debug=True)
