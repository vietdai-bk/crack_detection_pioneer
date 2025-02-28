from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

model = YOLO("models/crack_yolov11n.pt")
# model.to('cpu')

@app.get("/")
async def home():
    return {"message": "Welcome to the YOLOv11 model prediction API!"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        results = model(image)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return JSONResponse(content={"detections": detections})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=5000)