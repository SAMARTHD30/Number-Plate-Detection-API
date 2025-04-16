from fastapi import APIRouter, UploadFile, File, Form, HTTPException
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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
    status: str
    plate_text: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[List[int]] = None
    image_data: Optional[Dict[str, str]] = None

router = APIRouter(prefix="/api/v1")

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

# Load your model (add your model path)
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='app/models/best.pt', force_reload=True)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

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
    image: UploadFile = File(...),
    return_processed: Optional[bool] = Form(False)
):
    """Detect license plate in image"""
    try:
        # Validate file size
        await validate_file_size(image)

        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")

        logger.info(f"Processing image: {image.filename}")

        # Read image
        contents = await image.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Invalid image file or format")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Log image shape and type
        logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Convert to RGB for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Perform detection
            logger.info("Running model inference")
            results = model(image_rgb)
            logger.info(f"Detection complete. Found {len(results.xyxy[0])} objects")

            # Process results
            if len(results.xyxy[0]) > 0:
                # Get the first detection (highest confidence)
                detection = results.xyxy[0][0]
                bbox = [int(x) for x in detection[:4]]  # x1, y1, x2, y2
                confidence = float(detection[4])

                logger.info(f"Detection confidence: {confidence}")

                # Convert bbox format from x1,y1,x2,y2 to x,y,w,h
                x1, y1, x2, y2 = bbox
                bbox = [x1, y1, x2-x1, y2-y1]

                # Extract plate text (if available in your model's output)
                plate_text = results.names[int(detection[5])] if hasattr(results, 'names') else "DETECTED"
                logger.info(f"Detected plate text: {plate_text}")

                # Draw bounding box if return_processed is True
                if return_processed:
                    x, y, w, h = bbox
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(image, f"{plate_text} {confidence:.2f}",
                               (int(x), int(y - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Encode the processed image
                    _, buffer = cv2.imencode(".jpg", image)
                    base64_image = base64.b64encode(buffer).decode('utf-8')

                    return DetectionResponse(
                        status="success",
                        plate_text=plate_text,
                        confidence=confidence,
                        bbox=bbox,
                        image_data={
                            "content_type": "image/jpeg",
                            "base64_data": f"data:image/jpeg;base64,{base64_image}"
                        }
                    )

                return DetectionResponse(
                    status="success",
                    plate_text=plate_text,
                    confidence=confidence,
                    bbox=bbox
                )
            else:
                logger.info("No detection found in image")
                return DetectionResponse(
                    status="no_detection",
                    plate_text=None,
                    confidence=None,
                    bbox=None
                )

        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing request: {e}")
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