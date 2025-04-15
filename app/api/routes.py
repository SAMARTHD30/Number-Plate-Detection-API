from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import io
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")

class ProcessImageRequest(BaseModel):
    custom_text: Optional[str] = None

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

@router.post("/process-image")
async def process_and_return_image(
    car_image: UploadFile = File(...),
    custom_text: Optional[str] = Form(None),
    return_type: Optional[str] = Form("image", description="Return type: 'image' or 'json'")
):
    """Process car image and return the processed image directly"""
    try:
        if not car_image:
            raise HTTPException(status_code=400, detail="Car image is required")

        # Read and validate image
        contents = await car_image.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process the image (placeholder for your actual processing logic)
        # Example: Add text to image if custom_text is provided
        if custom_text:
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, custom_text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Generate unique ID
        image_id = str(uuid.uuid4())

        # Encode the processed image
        _, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer.tobytes())

        # Return based on return_type preference
        if return_type.lower() == "json":
            return {
                "image_id": image_id,
                "status": "success",
                "custom_text": custom_text if custom_text else None,
                "message": "Image processed successfully"
            }
        else:
            return StreamingResponse(io_buf, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep the old endpoints for backward compatibility
@router.post("/process")
async def process_image(
    car_image: UploadFile = File(...),
    custom_text: Optional[str] = Form(None)
):
    """Legacy endpoint - use /process-image instead"""
    result = await process_and_return_image(car_image, custom_text, return_type="json")
    return result

@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Legacy endpoint - use /process-image instead"""
    try:
        # Placeholder image response
        placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", placeholder)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))