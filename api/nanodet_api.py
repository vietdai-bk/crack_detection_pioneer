from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from demo.demo import Predictor
from nanodet.util import cfg, load_config, Logger
import torch

cfg_path = "nanodet_clone/config/shufflenet_custom.yml"
model_path = "models/nanodet_custom_025x_loss.pth"
device = torch.device('cpu')

load_config(cfg, cfg_path)
logger = Logger(-1, use_tensorboard=False)
predictor = Predictor(cfg, model_path, logger, device=device)

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Welcome to the NanoDet model prediction API!"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        meta, res = predictor.inference(image)
        print("Hello world")
        detections = []
        all_box = []
        for label in res[0]:
            for bbox in res[0][label]:
                score = bbox[-1]
                if score > 0.3:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    detections.append({
                        "class_id" : label,
                        "confidence": float(score),
                        "bbox": [int(x0), int(y0), int(x1), int(y1)]
                    })

        return JSONResponse(content={"detections": detections})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("nanodet_api:app", host='0.0.0.0', port=5000)