from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np
import io
import uuid
import base64
from typing import Dict, List, Optional
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image

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

# Load your model (add your model path)
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='app/models/best.pt')
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
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
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Read image
        contents = await image.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to RGB for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(image_rgb)

        # Process results
        if len(results.xyxy[0]) > 0:
            # Get the first detection (highest confidence)
            detection = results.xyxy[0][0]
            bbox = [int(x) for x in detection[:4]]  # x1, y1, x2, y2
            confidence = float(detection[4])

            # Convert bbox format from x1,y1,x2,y2 to x,y,w,h
            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2-x1, y2-y1]

            # Extract plate text (if available in your model's output)
            plate_text = results.names[int(detection[5])] if hasattr(results, 'names') else "DETECTED"

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
            return DetectionResponse(
                status="no_detection",
                plate_text=None,
                confidence=None,
                bbox=None
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-image", response_model=Optional[ImageResponse])
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