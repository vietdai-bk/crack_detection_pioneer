import cv2
import requests
import time
video_path = 'demo/video_demo.mp4' 

# Địa chỉ API
url = "http://192.168.43.159:5000/predict"
# url = "http://160.191.50.68:5000/predict"

class_names = ['crack'] 
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_times = 0
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS của video: {fps_video}")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(url, files={"image": img_encoded.tobytes()})

    if response.status_code == 200:
        detections = response.json()["detections"]
        for detection in detections:
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            if confidence > 0.5:
                x1, y1, x2, y2 = bbox

                color = (0, 255, 0)
                thickness = 2
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                label = f"{class_name}: {confidence*100:.2f}%"

                frame = cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, (255, 255, 255), 2)

    elapsed_time = time.time() - start_time
    total_times += elapsed_time
    fps_estimate = 1 / elapsed_time
    frame = cv2.putText(frame, f"{fps_estimate:.2f}", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f'avg fps: {total_frames/total_times}')
cap.release()
cv2.destroyAllWindows()