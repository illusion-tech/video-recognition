import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO

model = YOLO("settling-tank.pt")
app = FastAPI()


@app.post("/detect/")
async def detect_objects(files: list[UploadFile] = File(...)):
    try:
        result_images = []
        for index, file in enumerate(files):
            # Process the uploaded image for object detection
            image_bytes = await file.read()
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Perform object detection with YOLOv5
            results = model(image, save=True)

            # Draw bounding boxes on the image
            for detection in results.pred:
                image = cv2.rectangle(
                    image, (detection[0], detection[1]), (detection[2], detection[3]), (255, 0, 0), 2)

            # Convert the OpenCV image to PIL format
            image_pil = Image.fromarray(image)
            img_io = BytesIO()
            image_pil.save(img_io, format='JPEG')
            img_io.seek(0)
            result_images.append((f"image_{index}.jpg", img_io))

        return StreamingResponse(result_images, media_type='image/jpeg')
    except Exception as e:
        return {"error": str(e)}
