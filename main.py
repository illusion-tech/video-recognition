import os
import cv2
import uuid
import numpy as np
from enum import Enum
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

model = YOLO("models/best.pt")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DetectionType(Enum):
    CoarseScreen = 0
    BiologicalPoolFirst= 1
    BiologicalPoolSecond = 2
    EfficientPoolsFirst = 3
    EfficientPoolsSecond = 4 

class DetectionInput(BaseModel):
    detection_type: DetectionType

@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        result = model(image)

        date = datetime.now().strftime("%Y%m%d")
        predict_path = os.path.join("data/predict/", date)
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        filename = str(uuid.uuid4()) + '.jpg'
        predict_path = os.path.join(predict_path, filename)
        img = result[0].plot(font="Arial", line_width=2, font_size=15)
        cv2.imwrite(predict_path, img)
        return FileResponse(predict_path, media_type='image/jpeg')
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/detect/multiple")
async def predict_multiple(files: List[UploadFile] = File(...)):
    try:
        response_files = []
        for file in files:
            image_bytes = await file.read()
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            result = model(image)
        
            date = datetime.now().strftime("%Y%m%d")
            predict_path = os.path.join("data/predict/", date)
            if not os.path.exists(predict_path):
                os.makedirs(predict_path)
            filename = str(uuid.uuid4()) + '.jpg'
            predict_path = os.path.join(predict_path, filename)
            img = result[0].plot(font="Arial", line_width=2, font_size=15)
            cv2.imwrite(predict_path, img)

            response_files.append(predict_path)
        
        responses = [FileResponse(file_path, media_type='image/jpeg') for file_path in response_files]
        return responses
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
