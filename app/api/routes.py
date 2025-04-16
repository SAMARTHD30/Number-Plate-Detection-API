from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np
import io
import uuid
import base64
import logging
from typing import Dict, List, Optional
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
import traceback
from app.core.model import model  # Import the cached model
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure maximum upload size (100MB)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB in bytes

class ImageResponse(BaseModel):
    image_id: str
    status: str
    custom_text: Optional[str] = None
    message: str
    image_url: str
    timestamp: str
    image_data: Dict[str, str]

class DetectionResponse(BaseModel):
    detection_found: bool
    plate_text: str
    confidence: float
    bounding_box: Optional[dict] = None

# Fix the router prefix to avoid double /api
router = APIRouter()

class ProcessImageRequest(BaseModel):
    custom_text: Optional[str] = None

async def validate_file_size(file: UploadFile) -> None:
    """Validate file size before processing"""
    try:
        # Get file size from content length header
        content_length = int(file.headers.get("content-length", "0"))
        if content_length > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size allowed is {MAX_UPLOAD_SIZE/1024/1024}MB"
            )

        # If content-length header is missing, read file in chunks
        if content_length == 0:
            size = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    await file.seek(0)  # Reset file pointer
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size allowed is {MAX_UPLOAD_SIZE/1024/1024}MB"
                    )
            await file.seek(0)  # Reset file pointer for subsequent reads
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.error(f"Error validating file size: {e}")
            raise HTTPException(status_code=500, detail="Error validating file size")
        raise e

async def process_image(image: UploadFile, custom_text: Optional[str] = None) -> DetectionResponse:
    try:
        # Read image efficiently
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference with optimized settings
        results = model.predict(
            img_rgb,
            conf=float(settings.CONFIDENCE_THRESHOLD),
            iou=0.45,
            max_det=int(settings.MAX_DETECTIONS)
        )

        if len(results) == 0:
            return DetectionResponse(
                detection_found=False,
                plate_text="",
                confidence=0.0,
                bounding_box=None
            )

        # Get the first result
        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            return DetectionResponse(
                detection_found=False,
                plate_text="",
                confidence=0.0,
                bounding_box=None
            )

        # Get the detection with highest confidence
        box = boxes[0]
        confidence = float(box.conf)
        plate_text = result.names[int(box.cls)]

        # Convert bounding box to required format
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bounding_box = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }

        return DetectionResponse(
            detection_found=True,
            plate_text=plate_text,
            confidence=confidence,
            bounding_box=bounding_box
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ping", response_model=Dict[str, str])
async def ping():
    """Check API status"""
    return {
        "status": "ok",
        "message": "API is running",
        "version": "1.0.0"
    }

@router.post("/detect", response_model=DetectionResponse)
async def detect_plate(
    file: UploadFile = File(...),
    return_type: str = "json",
    custom_text: Optional[str] = None
):
    try:
        # Process the image
        detection = await process_image(file, custom_text)

        if return_type == "json":
            return detection

        # For image return type, draw bounding box and return image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if detection.detection_found:
            # Draw bounding box
            bbox = detection.bounding_box
            cv2.rectangle(
                img,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                (0, 255, 0),
                2
            )

            # Add text
            text = f"{detection.plate_text} ({detection.confidence:.2f})"
            if custom_text:
                text = f"{text} - {custom_text}"

            cv2.putText(
                img,
                text,
                (bbox["x1"], bbox["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Convert image to bytes
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()

        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg"
        )

    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-image", response_model=Optional[ImageResponse])
async def process_and_return_image(
    car_image: UploadFile = File(...),
    custom_text: Optional[str] = Form(None),
    return_type: Optional[str] = Form("image", description="Return type: 'image' or 'json'")
):
    """Process car image and return the processed image directly"""
    try:
        # Validate file size
        await validate_file_size(car_image)

        if not car_image:
            raise HTTPException(status_code=400, detail="Car image is required")

        # Read and validate image
        contents = await car_image.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process the image
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add text if provided
        if custom_text:
            # Parameters for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 3
            text_color = (0, 0, 255)  # Red color in RGB

            # Get text size
            text_size = cv2.getTextSize(custom_text, font, font_scale, font_thickness)[0]

            # Calculate text position (centered)
            text_x = (image_rgb.shape[1] - text_size[0]) // 2
            text_y = (image_rgb.shape[0] + text_size[1]) // 2

            # Add white background for text
            padding = 10
            cv2.rectangle(
                image_rgb,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (255, 255, 255),  # White background
                -1  # Filled rectangle
            )

            # Add text
            cv2.putText(
                image_rgb,
                custom_text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )

        # Generate unique ID and timestamp
        image_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Convert back to BGR for encoding
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Encode the processed image with high quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, buffer = cv2.imencode(".jpg", image_bgr, encode_params)
        io_buf = io.BytesIO(buffer.tobytes())

        # Convert image to base64 for JSON response
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Generate image URL (you can modify this based on your setup)
        image_url = f"/api/v1/image/{image_id}"

        # Return based on return_type preference
        if return_type.lower() == "json":
            return ImageResponse(
                image_id=image_id,
                status="success",
                custom_text=custom_text,
                message="Image processed successfully",
                image_url=image_url,
                timestamp=timestamp,
                image_data={
                    "content_type": "image/jpeg",
                    "base64_data": f"data:image/jpeg;base64,{base64_image}"
                }
            )
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
    """Retrieve a processed image by ID"""
    try:
        # Here you would typically fetch the image from storage
        # For now, returning a placeholder
        placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", placeholder)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))