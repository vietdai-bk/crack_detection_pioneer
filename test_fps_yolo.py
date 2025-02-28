import cv2
import time
from ultralytics import YOLO

model = YOLO("models/crack_yolov11n.pt")
# model.to('cpu')
video_path = "video_demo.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print("Không thể mở video:", video_path)
    exit()

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(annotated_frame, f"{fps:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('window', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()