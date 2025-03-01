import cv2
import numpy as np
import os
from sort.sort import Sort
from ultralytics import YOLO
from tkinter import Tk, Label, Canvas, Frame, Scrollbar, Entry, Button, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class_names = ['crack']
font = cv2.FONT_HERSHEY_SIMPLEX
frame_count = 0
tracking_start = {}
tracker = Sort()
saved_cracks = {}

os.makedirs("crack_images", exist_ok=True)
for file in os.listdir("crack_images"):
    os.remove(os.path.join("crack_images", file))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = resource_path("model/crack_yolov11n.pt")
model = YOLO(model_path).to(device)

DISPLAY_WIDTH = 1080
DISPLAY_HEIGHT = 720
IMAGE_SIZE = 200

def update_frame():
    global frame_count
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        original_h, original_w = frame.shape[:2]
        
        results = model(frame)
        boxes = results[0].boxes
        detections = []

        for i in range(len(boxes.xyxy)):
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            confidence = boxes.conf[i].item()
            if confidence > 0.5:
                detections.append([x1, y1, x2, y2, confidence])

        detections = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = tracker.update(detections)

        new_crack_saved = False

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Crack {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, (255, 255, 255), 2)
            
            if track_id not in tracking_start:
                tracking_start[track_id] = frame_count

            if frame_count - tracking_start[track_id] >= 20:
                if track_id not in saved_cracks or (frame_count - saved_cracks[track_id][1] >= 40):
                    pad = 10
                    crop_x1 = max(x1 - pad, 0)
                    crop_y1 = max(y1 - pad, 0)
                    crop_x2 = min(x2 + pad, original_w)
                    crop_y2 = min(y2 + pad, original_h)
                    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    image_path = f"crack_images/crack_{track_id}_frame_{frame_count}.jpg"
                    cv2.imwrite(image_path, cropped)
                    saved_cracks[track_id] = (image_path, frame_count)
                    new_crack_saved = True

        if new_crack_saved:
            update_crack_images()
        
        aspect_ratio = original_w / original_h
        if aspect_ratio > DISPLAY_WIDTH / DISPLAY_HEIGHT:
            new_w = DISPLAY_WIDTH
            new_h = int(DISPLAY_WIDTH / aspect_ratio)
        else:
            new_h = DISPLAY_HEIGHT
            new_w = int(DISPLAY_HEIGHT * aspect_ratio)

        frame_resized = cv2.resize(frame, (new_w, new_h))
        display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        y_offset = (DISPLAY_HEIGHT - new_h) // 2
        x_offset = (DISPLAY_WIDTH - new_w) // 2
        display_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = frame_resized

        frame_tk = Image.fromarray(display_frame)
        frame_tk = ImageTk.PhotoImage(frame_tk)
        lbl_video.imgtk = frame_tk
        lbl_video.config(image=frame_tk)
        lbl_video.after(10, update_frame)
    else:
        cap.release()

def update_crack_images():
    for widget in frame_inner.winfo_children():
        widget.destroy()

    crack_images = sorted(os.listdir("crack_images"), 
                         key=lambda x: os.path.getctime(os.path.join("crack_images", x)), 
                         reverse=True)
    
    for image_name in crack_images:
        image_path = os.path.join("crack_images", image_name)
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img = ImageTk.PhotoImage(img)
                lbl = Label(frame_inner, image=img, bg='white', bd=2, relief='solid')
                lbl.image = img
                lbl.pack(pady=5)
            except Exception as e:
                print(f"Lỗi khi mở ảnh {image_path}: {e}")

def on_close():
    if 'cap' in globals():
        cap.release()
    root.destroy()

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        entry_source.delete(0, 'end')
        entry_source.insert(0, file_path)

def start_processing():
    video_source = entry_source.get()
    if not video_source:
        messagebox.showerror("Error", "Vui lòng nhập đường dẫn hoặc chọn file.")
        return
    global cap
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Không thể mở nguồn video. Vui lòng kiểm tra lại đường dẫn hoặc link.")
        return
    update_frame()

if __name__ == '__main__':
    root = Tk()
    root.title("Crack Detection")
    root.geometry("1400x800")
    root.configure(bg='#f0f0f0')
    frame_input = Frame(root, bg='#f0f0f0')
    frame_input.pack(pady=10)

    Label(frame_input, text="Nhập đường dẫn video hoặc link camera IP:", 
          font=("Arial", 12), bg='#f0f0f0').pack(side='left', padx=5)
    
    entry_source = Entry(frame_input, width=40)
    entry_source.pack(side='left', padx=5)
    entry_source.insert(0, "demo/video_demo.mp4")
    Button(frame_input, text="Duyệt", command=browse_file, font=("Arial", 10)).pack(side='left', padx=5)
    Button(frame_input, text="Bắt đầu", command=start_processing, font=("Arial", 10), bg="#4CAF50", fg="white").pack(side='left', padx=5)

    frame_left = Frame(root, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, bg='black', bd=3, relief='ridge')
    frame_left.pack(side='left', padx=10, pady=10)

    lbl_video = Label(frame_left, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, bg='black')
    lbl_video.pack()
    frame_right = Frame(root, width=250, height=DISPLAY_HEIGHT, bg='#d9d9d9', bd=3, relief='ridge')
    frame_right.pack(side='right', fill='y', padx=10, pady=10)

    lbl_title = Label(frame_right, text="Detected Cracks", font=("Arial", 14, "bold"), bg='#d9d9d9')
    lbl_title.pack(pady=5)

    canvas = Canvas(frame_right, bg='#d9d9d9')
    scrollbar = Scrollbar(frame_right, orient="vertical", command=canvas.yview)
    frame_inner = Frame(canvas, bg='#d9d9d9')
    frame_inner.bind(
        "<Configure>", 
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=frame_inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()