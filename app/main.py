import cv2
from starlette.responses import Response
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from fastapi.responses import FileResponse

from segmentation import  get_image_from_bytes

model = YOLO("settling-tank.pt")
app = FastAPI()


@app.post("/detect/")
async def detect_objects(files: list[UploadFile] = File(...)):
    try:
        result_images = []
        for index, file in enumerate(files):

            image_bytes = await file.read()
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            results = model.predict(image, save=True, save_txt=True)
            print(results)


            for detection in results.pred:
                image = cv2.rectangle(
                    image, (detection[0], detection[1]), (detection[2], detection[3]), (255, 0, 0), 2)

            image_pil = Image.fromarray(image)
            img_io = BytesIO()
            image_pil.save(img_io, format='JPEG')
            img_io.seek(0)
            result_images.append((f"image_{index}.jpg", img_io))

        return {'successful'}
    except Exception as e:
        return {"error": str(e)}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: UploadFile):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv5
    results = model.predict(image, save=True)
    print(results)
    #results.render()  # updates results.imgs with boxes and labels

    file_path = results.path  # 替换为您要返回的文件路径
    return FileResponse(file_path, media_type='image/jpeg')