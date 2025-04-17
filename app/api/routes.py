from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np
import io
import uuid
import base64
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback
from pathlib import Path
from app.core.model import get_model, preprocess_image
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use settings.MAX_FILE_SIZE from config.py instead of defining it here
# MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB in bytes

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
    plate_text: Optional[str] = None
    confidence: Optional[float] = None
    bounding_box: Optional[dict] = None

# Fix the router prefix to avoid double /api
router = APIRouter()

class ProcessImageRequest(BaseModel):
    car_image: UploadFile = File(...)
    custom_text: Optional[str] = None
    return_type: Optional[str] = "image"

async def validate_file_size(file: UploadFile) -> None:
    """Validate file size before processing"""
    try:
        content_length = int(file.headers.get("content-length", "0"))
        if content_length > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size allowed is {settings.MAX_FILE_SIZE/1024/1024}MB"
            )

        if content_length == 0:
            size = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.MAX_FILE_SIZE:
                    await file.seek(0)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size allowed is {settings.MAX_FILE_SIZE/1024/1024}MB"
                    )
            await file.seek(0)
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.error(f"Error validating file size: {str(e)}")
            raise HTTPException(status_code=500, detail="Error validating file size")
        raise e

@router.get("/ping", response_model=Dict[str, str])
async def ping():
    """Check API status"""
    return {
        "status": "ok",
        "message": "API is running",
        "version": "1.0.0"
    }

@router.post("/detect", status_code=200)
async def detect(request: Request,
                car_image: UploadFile = File(...),
                conf_threshold: float = Form(0.2),
                return_type: str = Form("json"),
                custom_text: Optional[str] = Form(None)):
    """
    Detect and read license plate from an image.

    Args:
        car_image (UploadFile): The image containing the license plate.
        conf_threshold (float): Confidence threshold for detection. Defaults to 0.2.
        return_type (str): Return type. Either "json" or "image". Defaults to "json".
        custom_text (str, optional): Custom text to add to the image. Defaults to None.
    """
    try:
        # Start timing
        start_time = datetime.now()

        # Validate confidence threshold
        if conf_threshold < 0.0:
            conf_threshold = 0.0

        # Validate file size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {settings.MAX_FILE_SIZE/(1024*1024):.1f} MB"
            )

        # Read and process the image
        contents = await car_image.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {settings.MAX_FILE_SIZE/(1024*1024):.1f} MB"
            )

        # Convert the image from bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )

        # Log image dimensions
        logger.info(f"Image dimensions: {image.shape}")

        # Get the YOLO model
        model = get_model()
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model"
            )

        # Extract regions where license plates are likely to be found
        regions = focus_license_plate_regions(image)
        logger.info(f"Found {len(regions)} regions to analyze for license plates")

        # Initialize response
        detection_result = {
            "detection_found": False,
            "plate_text": None,
            "confidence": None,
            "bounding_box": None
        }

        best_confidence = 0.0
        best_plate = None
        best_box = None

        # Process each region for license plate detection
        for i, region in enumerate(regions):
            logger.info(f"Processing region {i+1}/{len(regions)}")

            # Preprocess the image for the model
            processed_img = preprocess_image(region)

            # Run inference
            results = model.predict(
                source=processed_img,
                conf=conf_threshold,
                iou=0.3,  # Lower IOU for license plates
                verbose=False
            )

            # Check if there are any detections
            if not results or len(results) == 0:
                logger.info(f"No detections in region {i+1}")
                continue

            # Get the first result
            result = results[0]

            # Check if there are any boxes
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                logger.info(f"No boxes in region {i+1}")
                continue

            # Get all detections
            boxes = result.boxes

            # Filter by aspect ratio (license plates are typically wider than tall)
            filtered_indices = []
            for j, box in enumerate(boxes):
                try:
                    # Get coordinates
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    width = x2 - x1
                    height = y2 - y1

                    # Skip if dimensions are invalid
                    if width <= 0 or height <= 0:
                        continue

                    # Calculate aspect ratio (width/height)
                    aspect_ratio = width / height

                    # License plates typically have aspect ratios between 1:1 and 8:1 (more inclusive)
                    if 1.0 <= aspect_ratio <= 8.0:
                        # Calculate relative area
                        img_area = region.shape[0] * region.shape[1]
                        box_area = width * height
                        area_percentage = (box_area / img_area) * 100

                        # License plates typically occupy 0.2-30% of a focused region (more inclusive)
                        if 0.2 <= area_percentage <= 30.0:
                            filtered_indices.append(j)
                            logger.info(f"Candidate plate: aspect={aspect_ratio:.2f}, area={area_percentage:.2f}%, conf={float(box.conf[0]):.2f}")
                    else:
                        logger.debug(f"Rejected by aspect ratio: {aspect_ratio:.2f}")

                except Exception as e:
                    logger.error(f"Error filtering box {j}: {str(e)}")

            # Process filtered boxes
            for j in filtered_indices:
                box = boxes[j]
                confidence = float(box.conf[0])

                # If confidence is higher than previous best, update
                if confidence > best_confidence:
                    # Extract text from box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Get the license plate text (simulated)
                    plate_text = "ABC123"  # Replace with actual OCR in production

                    # Update best detection
                    best_confidence = confidence
                    best_plate = plate_text
                    best_box = {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "region_index": i
                    }

                    logger.info(f"Found better plate: {plate_text} with conf={confidence:.2f}")

        # Check if we found any valid license plate
        if best_plate is not None:
            detection_result["detection_found"] = True
            detection_result["plate_text"] = best_plate
            detection_result["confidence"] = best_confidence
            detection_result["bounding_box"] = best_box

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        detection_result["processing_time"] = processing_time

        logger.info(f"Detection completed in {processing_time:.2f} seconds")
        logger.info(f"Detection result: {detection_result}")

        # Return the result based on return_type
        if return_type.lower() == "json":
            return detection_result
        else:
            # If return type is image, create an annotated image
            annotated_img = image.copy()

            if detection_result["detection_found"]:
                # Get the coordinates
                box = detection_result["bounding_box"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # Draw rectangle on the image
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put text on the image
                plate_text = detection_result["plate_text"]
                confidence = detection_result["confidence"]
                text = f"{plate_text} ({confidence:.2f})"
                cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add custom text if provided
            if custom_text:
                cv2.putText(annotated_img, custom_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert the image to bytes
            _, buffer = cv2.imencode(".jpg", annotated_img)
            img_str = base64.b64encode(buffer).decode()

            return {"image": img_str, "detection": detection_result}

    except Exception as e:
        logger.error(f"Error during license plate detection: {str(e)}")
        logger.error(traceback.format_exc())
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

# Add vehicle detector for two-stage approach (could be moved to model.py in production)
vehicle_detector = None

def load_vehicle_detector():
    """
    Load YOLOv8 model for vehicle detection to narrow down license plate search areas

    Returns:
        YOLO model instance for vehicle detection, or None if not available
    """
    global vehicle_detector

    if vehicle_detector is None:
        try:
            # Try to import YOLO
            from ultralytics import YOLO

            # Use a standard YOLOv8 model (comes with vehicle classes)
            vehicle_detector = YOLO("yolov8n.pt")
            logger.info("Vehicle detector model loaded successfully")
            return vehicle_detector
        except Exception as e:
            logger.warning(f"Could not load vehicle detector: {str(e)}")
            logger.warning("Will continue with direct license plate detection")
            return None
    else:
        return vehicle_detector

def detect_vehicles(image: np.ndarray) -> List[Dict]:
    """
    Detect vehicles in the image to focus license plate detection

    Args:
        image: Original RGB image

    Returns:
        List of vehicle bounding boxes or empty list if no vehicles found
    """
    vehicle_classes = [2, 3, 5, 7]  # Standard COCO classes: car, motorcycle, bus, truck

    try:
        # Try to load vehicle detector
        detector = load_vehicle_detector()
        if detector is None:
            return []

        # Run detection
        results = detector.predict(
            source=image,
            conf=0.3,  # Lower threshold for vehicles
            verbose=False
        )

        if not results or len(results) == 0:
            return []

        result = results[0]

        # Check if we have any boxes
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return []

        # Filter for vehicle classes
        vehicle_boxes = []

        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Expand slightly to ensure we capture the license plate
                height, width = image.shape[:2]
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(width, x2 + 10)
                y2 = min(height, y2 + 10)

                vehicle_boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(box.conf[0]),
                    "class": cls
                })

        return vehicle_boxes

    except Exception as e:
        logger.error(f"Error in vehicle detection: {str(e)}")
        return []

def focus_license_plate_regions(image: np.ndarray) -> List[np.ndarray]:
    """
    Extract regions of interest where license plates are likely to be found

    Args:
        image: Original BGR image

    Returns:
        List of image regions (RGB) to focus license plate detection on
    """
    try:
        # Convert to RGB for detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get vehicle boxes
        vehicle_boxes = detect_vehicles(rgb_image)

        if not vehicle_boxes:
            # If no vehicles detected, return whole image
            logger.info("No vehicles detected, using whole image for license plate detection")
            return [rgb_image]

        # Extract regions with focus on likely license plate locations
        regions = []

        for box in vehicle_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            vehicle_img = rgb_image[y1:y2, x1:x2].copy()

            # Vehicle detected, focus on front/rear regions where plates are typically found
            height, width = y2 - y1, x2 - x1

            # Add full vehicle region
            regions.append(vehicle_img)

            # Add lower third of vehicle (common location for license plates)
            lower_region_y = y1 + int(height * 0.6)
            if lower_region_y < y2:
                lower_region = rgb_image[lower_region_y:y2, x1:x2].copy()
                if lower_region.size > 0:
                    regions.append(lower_region)

            # Add front region (first 40% of vehicle width)
            front_region_x = x1 + int(width * 0.4)
            if front_region_x > x1:
                front_region = rgb_image[y1:y2, x1:front_region_x].copy()
                if front_region.size > 0:
                    regions.append(front_region)

            # Add rear region (last 40% of vehicle width)
            rear_region_x = x2 - int(width * 0.4)
            if rear_region_x < x2:
                rear_region = rgb_image[y1:y2, rear_region_x:x2].copy()
                if rear_region.size > 0:
                    regions.append(rear_region)

            logger.info(f"Extracted {len(regions)} regions from vehicle at {box}")

        # If no valid regions extracted, return original image
        if not regions:
            return [rgb_image]

        return regions

    except Exception as e:
        logger.error(f"Error focusing on license plate regions: {str(e)}")
        # Return original image if anything fails
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]