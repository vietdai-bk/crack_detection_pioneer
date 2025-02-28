import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("../models/crack_yolov11n.pt")

# Hàm nhận diện ảnh
def predict_image(image):
    # Chuyển ảnh sang định dạng YOLO
    results = model(image)

    # Vẽ bounding box lên ảnh
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Lấy toạ độ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

# Hàm nhận diện video
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])  # Lấy toạ độ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    
    return "output.mp4"

# Tạo giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## YOLOv8 Object Detection")
    
    with gr.Tab("Detect Image"):
        image_input = gr.Image(type="numpy")
        image_output = gr.Image()
        image_button = gr.Button("Detect")
        image_button.click(predict_image, inputs=image_input, outputs=image_output)
    
    with gr.Tab("Detect Video"):
        video_input = gr.Video()
        video_output = gr.Video()
        video_button = gr.Button("Detect")
        video_button.click(predict_video, inputs=video_input, outputs=video_output)

# Chạy ứng dụng Gradio
if __name__ == "__main__":
    demo.launch(share=True)
