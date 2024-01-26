import os
import cv2
import uuid
import time
import numpy as np
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from obs import ObsClient
from obs import PutObjectHeader
from dotenv import load_dotenv

model = YOLO("models/best.pt")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
ak = os.getenv("AccessKeyId")
sk = os.getenv("SecretAccessKey")
server = os.getenv("EndPoint")
bucketName = os.getenv("BucketName")
obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)

class DetectBody(BaseModel):
    imageUrl: str

@app.post("/detect")
async def predict(body: DetectBody):
    try:
        result = model(body.imageUrl)
        # 数据集的全部类名
        classNames = result[0].names
        # 检测到的类名
        detectNames = []
        for box in result[0].boxes:
            name = classNames[int(box.cls)]
            detectNames.append(name)

        # 检测到的类名如果为空或包含异物，检测结果为 False
        detection = False if len(detectNames) == 0 or 'matter' in detectNames else True

        date = datetime.now().strftime("%Y%m%d")
        timestamp = int(time.time())
        predict_path = os.path.join("data/predict/", date if detection else ('matter/' + date))
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        filename = str(timestamp) + '.jpg'
        predict_path = os.path.join(predict_path, filename)

        img = result[0].plot(font="Arial", line_width=2, font_size=10)
        cv2.imwrite(predict_path, img)

        # 上传到华为云 OBS, 并返回图片地址
        headers = PutObjectHeader()
        headers.contentType = 'image/jpeg'
        objectKey = 'yolov8-predict/' + (date if detection else ('matter/' + date)) + '/' + filename
        resp = obsClient.putFile(
            bucketName,
            objectKey,
            file_path=predict_path,
            headers=headers
        )
        if resp.status < 300:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "content": {
                        "url": resp.body.objectUrl,
                        "detection": detection,
                        "info": {
                            "names": detectNames,
                        }
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": resp.errorMessage
                }
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # 删除当前目录下保存的 .jpg/.png 文件
        for file in os.listdir('./'):
            if file.endswith('.jpg') or file.endswith('.png'):
                os.remove(file)


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
