import cv2
import time
from nanodet_clone.nanodet.util import overlay_bbox_cv
from nanodet_clone.demo.demo import Predictor
from nanodet_clone.nanodet.util import cfg, load_config, Logger
import torch

cfg_path = "nanodet_clone/config/shufflenet_custom.yml"
model_path = "models/nanodet_custom_025x_loss.pth"
device = torch.device('cpu')

load_config(cfg, cfg_path)
logger = Logger(-1, use_tensorboard=False)
predictor = Predictor(cfg, model_path, logger, device=device)

video_path = "demo/video_demo.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print("Không thể mở video:", video_path)
    exit()
total_times = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    meta, res = predictor.inference(frame)
    result_frame = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.5)
    elapsed_time = time.time() - start_time
    total_times += elapsed_time
    fps_estimate = 1 / elapsed_time
    frame = cv2.putText(frame, f"{fps_estimate:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    out.write(result_frame)
    cv2.imshow('window', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('\n')
print(f"avg fps: {total_frame/total_times}")

out.release()
cap.release()