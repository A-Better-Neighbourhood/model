from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import numpy as np
from typing import List

app = FastAPI(
    title="Pothole Detection API",
    description="YOLOv8 pothole detection",
    version="1.0.0"
)

# Load model
print("Loading model...")
model = YOLO('best.pt')
print("Model loaded!")

@app.get("/")
def root():
    return {
        "name": "Pothole Detection API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "YOLOv8"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Predict
        results = model.predict(
            image,
            conf=conf_threshold,
            imgsz=416,
            verbose=False
        )
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            det = {
                "confidence": round(float(box.conf[0]), 3),
                "class": "pothole",
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 2),
                    "y1": round(float(box.xyxy[0][1]), 2),
                    "x2": round(float(box.xyxy[0][2]), 2),
                    "y2": round(float(box.xyxy[0][3]), 2)
                }
            }
            detections.append(det)
        
        # Get annotated image
        annotated = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)