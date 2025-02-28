from ultralytics import YOLO
import cv2
import time

model = YOLO('models/crack_yolov11n.pt')
cap = cv2.VideoCapture('demo/video.mp4')
fps = 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('demo/video_output_fix_25fps.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.4)
    anno = results[0].plot()
    end = time.time()
    fps = 1/(end-start)
    cv2.putText(anno, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    out.write(anno)
    cv2.imshow('windows', anno)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()