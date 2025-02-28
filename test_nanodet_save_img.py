import cv2
import numpy as np
from nanodet_clone.demo.demo import Predictor
from nanodet_clone.nanodet.util import cfg, load_config, Logger
import torch
import os 
from sort.sort import Sort
import time

video_path = 'video_demo.mp4'
class_names = ['crack']
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(f'demo/video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_times = 0
frame_count = 0

tracker = Sort()
saved_cracks = {}
tracking_start = {}
os.makedirs("crack_images_nanodet", exist_ok=True)
os.makedirs("demo",exist_ok=True)

cfg_path = "nanodet_clone/config/shufflenet_custom.yml"
model_path = "models/nanodet_custom_025x_loss.pth"
device = torch.device('cpu')

load_config(cfg, cfg_path)
logger = Logger(-1, use_tensorboard=False)
predictor = Predictor(cfg, model_path, logger, device=device)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS của video: {fps_video}")
while True:
    start_time = time.time()
    ret, frame = cap.read()
    out.write(frame)
    if not ret:
        break

    frame_count += 1
    meta, res = predictor.inference(frame)
    detections = []
    for label in res[0]:
        for bbox in res[0][label]:
            score = bbox[-1]
            if score > 0.5:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                detections.append([x0,y0,x1,y1,score])

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
                cv2.imwrite(f"crack_images_nanodet/crack_{track_id}_frame_{frame_count}.jpg", frame)

    elapsed_time = time.time() - start_time
    total_times += elapsed_time
    fps_estimate = 1 / elapsed_time
    cv2.putText(frame, f"{fps_estimate:.2f} FPS", (10, 30), font, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f'avg fps: {frame_count / total_times}')
cap.release()
out.release()
cv2.destroyAllWindows()