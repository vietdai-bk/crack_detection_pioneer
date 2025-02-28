import cv2
import time
import numpy as np
import os
from sort.sort import Sort
from ultralytics import YOLO

video_path = "demo/video_2.mp4"
class_names = ['crack']
font = cv2.FONT_HERSHEY_SIMPLEX

tracker = Sort()
saved_cracks = {}
tracking_start = {}
os.makedirs("crack_images", exist_ok=True)
os.makedirs("demo",exist_ok=True)

model = YOLO("models/crack_yolov11n.pt")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(f'demo/video2_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_times = 0
frame_count = 0

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS của video: {fps_video}")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    out.write(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break

    frame_count += 1

    results = model(frame)
    boxes = results[0].boxes
    detections = []
    for i in range(len(boxes.xyxy)):
        box = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = box.astype(int)
        class_id = boxes.cls[i].item()
        confidence = boxes.conf[i].item()
        if confidence > 0.5:
            detections.append([x1, y1, x2, y2, confidence])

    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    tracked_objects = tracker.update(detections)

    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Crack {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, (255, 255, 255), 2)

        if track_id not in tracking_start:
            tracking_start[track_id] = frame_count

        if frame_count - tracking_start[track_id] >= 20:
            if track_id not in saved_cracks or (frame_count - saved_cracks[track_id] >= 40):
                saved_cracks[track_id] = frame_count
                cv2.imwrite(f"crack_images/crack_{track_id}_frame_{frame_count}.jpg", frame)

    elapsed_time = time.time() - start_time
    total_times += elapsed_time
    fps_estimate = 1 / elapsed_time
    cv2.putText(frame, f"{fps_estimate:.2f} FPS", (10, 30), font, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f'avg fps: {frame_count / total_times}')
cap.release()
out.release()
cv2.destroyAllWindows()
