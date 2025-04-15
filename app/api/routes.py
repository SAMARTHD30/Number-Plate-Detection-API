from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import io
import uuid
from typing import Dict, List

router = APIRouter(prefix="/api/v1")

@router.get("/ping", response_model=Dict[str, str])
async def ping():
    """Check API status"""
    return {
        "status": "ok",
        "message": "API is running",
        "version": "1.0.0"
    }

@router.post("/detect")
async def detect_plate(image: UploadFile = File(...)):
    """Detect license plate in image"""
    try:
        contents = await image.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Placeholder for detection logic
        # In production, you would call your YOLO model here
        return {"message": "Detection endpoint working", "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_image(
    car_image: UploadFile = File(...),
    custom_text: str = Form(None)
):
    """Process car image with optional custom text"""
    try:
        contents = await car_image.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Generate a unique ID for the processed image
        image_id = str(uuid.uuid4())

        return {"image_id": image_id, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve a processed image by ID"""
    try:
        # Placeholder image response
        placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", placeholder)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))