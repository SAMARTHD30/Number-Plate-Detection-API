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
from app.core.model import get_model, preprocess_image, map_coordinates_to_original
from app.core.config import settings
import time
import os

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
async def detect(
    request: Request,
    car_image: UploadFile = File(...),
    conf_threshold: float = Form(0.1),  # Lowered default threshold
    return_type: str = Form("json"),
    custom_text: Optional[str] = Form(None),
    skip_enhancement: bool = Form(False),  # Changed default to False to enable enhancement
    iou_threshold: float = Form(0.1)
):
    """
    Detect license plates in the provided car image.

    Parameters:
    - car_image: Image file
    - conf_threshold: Confidence threshold for detections (0.0-1.0)
    - return_type: Return format ("json" or "image")
    - custom_text: Optional text to add to image output
    - skip_enhancement: Skip image enhancement if True
    - iou_threshold: IoU threshold for NMS

    Returns:
    - JSON: Detection results
    - or
    - Image: Annotated image with detection
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

        # Perform license plate detection directly to ensure consistent results
        detection_result = detect_license_plate(
            image=image,
            conf_threshold=conf_threshold,
            skip_enhancement=skip_enhancement
        )

        # Transform the result into the expected format
        api_result = {
            "detection_found": detection_result["success"],
            "plate_text": None,  # No OCR implemented yet
            "confidence": detection_result["confidence"] if detection_result["success"] else None,
            "bounding_box": None
        }

        # Extract the best detection if available
        if detection_result["success"] and detection_result["detections"]:
            best_detection = None
            for detection in detection_result["detections"]:
                if best_detection is None or detection["confidence"] > best_detection["confidence"]:
                    best_detection = detection

            if best_detection:
                x1, y1, x2, y2 = best_detection["box"]
                api_result["bounding_box"] = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        api_result["processing_time"] = processing_time

        logger.info(f"Detection completed in {processing_time:.2f} seconds")
        logger.info(f"Detection result: {api_result}")

        # Return the result based on return_type
        if return_type.lower() == "json":
            return api_result
        else:
            # If return type is image, create an annotated image
            annotated_img = image.copy()

            # If detection found, draw bounding box
            if api_result["detection_found"] and api_result["bounding_box"]:
                box = api_result["bounding_box"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # Draw rectangle around license plate
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add confidence text
                conf_text = f"Conf: {api_result['confidence']:.2f}"
                cv2.putText(annotated_img, conf_text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add custom text if provided
            if custom_text:
                # Place at top-left corner
                cv2.putText(annotated_img, custom_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Convert the image to bytes
            _, buffer = cv2.imencode(".jpg", annotated_img)
            img_str = base64.b64encode(buffer).decode()

            return {"image": img_str, "detection": api_result}

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

@router.post("/detect-and-process")
async def detect_and_process(
    request: Request,
    car_image: UploadFile = File(...),
    custom_text: Optional[str] = Form(None),
    text_x: Optional[int] = Form(None),
    text_y: Optional[int] = Form(None),
    conf_threshold: float = Form(0.1),  # Lowered threshold for better detection
    return_type: str = Form("image"),  # Default to "image" for direct image viewing
    skip_enhancement: bool = Form(False)  # Changed default to enable enhancement
):
    """
    Detect license plate and return processed annotated image.

    Parameters:
    - car_image: Image file
    - custom_text: Optional text to add to the processed image
    - text_x: X coordinate for custom text
    - text_y: Y coordinate for custom text
    - conf_threshold: Confidence threshold (0.0-1.0)
    - return_type: Return format ("json", "image", or "base64")
    - skip_enhancement: Skip image enhancement if True

    Returns:
    - When return_type is "image": Direct image response (StreamingResponse)
    - When return_type is "json" or "base64": JSON with detection results and optionally base64 image
    """
    try:
        # Start timing
        start_time = datetime.now()

        # Read image file
        contents = await car_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get image dimensions
        height, width = image.shape[:2]
        logger.info(f"Image dimensions: {width}x{height}")

        # Perform detection using the common detection function
        detection_result = detect_license_plate(
            image=image,
            conf_threshold=conf_threshold,
            skip_enhancement=skip_enhancement
        )

        # Create an annotated image
        annotated_img = image.copy()

        # Draw license plate detection if found
        if detection_result["success"] and detection_result["detections"]:
            # Find the best detection (already determined in detect_license_plate)
            best_detection = None
            for detection in detection_result["detections"]:
                if best_detection is None or detection["confidence"] > best_detection["confidence"]:
                    best_detection = detection

            if best_detection:
                # Get the coordinates
                x1, y1, x2, y2 = best_detection["box"]

                # Ensure coordinates are valid
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # If no custom text is provided, use a default value
                display_text = custom_text if custom_text else "License Plate"

                # Calculate text dimensions with larger, bolder font
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8  # Increased from 0.6
                font_thickness = 3  # Increased from 2
                text_size, baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)

                # Create a white filled rectangle directly over the license plate
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 255, 255), -1)  # White filled rectangle

                # Calculate text position to center it within the rectangle
                text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center horizontally
                text_y = y1 + (y2 - y1 + text_size[1]) // 2  # Center vertically

                # Add the text in black color with improved visibility
                cv2.putText(annotated_img, display_text, (text_x, text_y),
                            font, font_scale, (0, 0, 0), font_thickness)

        # Generate a unique image ID
        image_id = str(uuid.uuid4())
        timestamp = int(time.time())

        # Save processed image to disk
        output_dir = os.path.join(settings.MEDIA_ROOT, "processed")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{image_id}.jpg")
        cv2.imwrite(output_file, annotated_img)

        # Generate URL for the processed image
        host = request.headers.get("host", f"{settings.API_HOST}:{settings.API_PORT}")
        image_url = f"http://{host}/media/processed/{image_id}.jpg"

        # Check if return_type is "image" - return a StreamingResponse with the image
        if return_type.lower() == "image":
            # Encode the image for streaming
            _, buffer = cv2.imencode(".jpg", annotated_img)
            io_buf = io.BytesIO(buffer.tobytes())
            io_buf.seek(0)

            # Return the image directly
            logger.info("Returning image as StreamingResponse")
            return StreamingResponse(io_buf, media_type="image/jpeg")

        # Otherwise, prepare JSON response (for "json" or "base64" return types)
        # Check if base64 encoding is requested
        include_base64 = return_type.lower() in ["base64", "both"]
        base64_data = None

        if include_base64:
            # Convert the processed image to base64
            _, buffer = cv2.imencode(".jpg", annotated_img)
            base64_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        # Build the JSON response
        response = {
            "image_id": image_id,
            "status": "success",
            "custom_text": custom_text,
            "custom_text_position": {"x": text_x, "y": text_y} if text_x and text_y else None,
            "message": "Image processed successfully",
            "image_url": image_url,
            "timestamp": timestamp,
            "detection": {
                "detection_found": detection_result["success"],
                "plate_text": None,  # No OCR implemented yet
                "confidence": detection_result["confidence"] if detection_result["success"] else None,
                "bounding_box": best_detection if detection_result["success"] and best_detection else None,
                "detections": detection_result["detections"],
                "detection_time": detection_result["detection_time"]
            }
        }

        # Include image data if requested
        if include_base64:
            response["image_data"] = {
                "content_type": "image/jpeg",
                "base64_data": base64_data
            }

        # Return the JSON response
        logger.info("Returning JSON response")
        return response

    except Exception as e:
        logger.error(f"Error in detect_and_process: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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
    Extract regions of interest where license plates are likely to be found.
    Now returning only the whole image for better resolution and to avoid boundary issues.

    Args:
        image: Original BGR image

    Returns:
        List containing only the whole image in RGB format
    """
    try:
        # Convert to RGB for detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Return only the whole image
        logger.info("Processing the whole image at higher resolution for license plate detection")
        regions = [rgb_image]

        return regions

    except Exception as e:
        logger.error(f"Error creating license plate regions: {str(e)}")
        # Return original image if anything fails
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]

def detect_license_plate(image, conf_threshold=0.1, skip_enhancement=False):
    """
    Process regions of interest for license plate detection.

    Args:
        image: Image as numpy array (BGR)
        conf_threshold: Confidence threshold for detections (default: 0.1)
        skip_enhancement: Whether to skip image enhancement (default: False)

    Returns:
        Dictionary with detection results
    """
    try:
        # Initialize response
        response = {
            "success": False,
            "license_plate": None,
            "confidence": 0,
            "detections": [],
            "detection_time": 0,
            "error": None
        }

        if image is None or image.size == 0 or len(image.shape) < 2:
            response["error"] = "Invalid image provided to detector"
            logger.error(response["error"])
            return response

        # Get the YOLO model
        model = get_model()
        if model is None:
            response["error"] = "Failed to load detection model"
            logger.error(response["error"])
            return response

        # Get focus regions (potential license plate areas)
        regions = focus_license_plate_regions(image)

        if not regions:
            logger.warning("No regions found for license plate detection")
            regions = [image]  # Fallback to full image

        start_time = time.time()
        highest_conf = 0
        best_detection = None

        # Process each region
        for idx, region in enumerate(regions):
            try:
                # Process image with or without enhancement based on flag
                processed_img, transform_params = preprocess_image(region, skip_enhancement=skip_enhancement)

                if processed_img is None:
                    logger.warning(f"Region {idx} preprocessing failed")
                    continue

                # Perform detection on this region
                results = model.predict(
                    source=processed_img,
                    conf=conf_threshold,  # Use lower confidence threshold
                    verbose=False
                )

                # Process results for this region
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()

                    for i, (box, conf) in enumerate(zip(boxes, confs)):
                        x1, y1, x2, y2 = box.astype(int)

                        # Map coordinates back to original image space using transform_params
                        orig_coords = map_coordinates_to_original([x1, y1, x2, y2], transform_params)
                        ox1, oy1, ox2, oy2 = orig_coords

                        # Calculate aspect ratio
                        width = ox2 - ox1
                        height = oy2 - oy1
                        aspect_ratio = width / height if height > 0 else 0

                        # Calculate area percentage
                        region_height, region_width = region.shape[:2]
                        region_area = region_height * region_width
                        box_area = width * height
                        area_percentage = (box_area / region_area) * 100

                        # Filter based on combined criteria
                        valid_aspect_ratio = 1.0 <= aspect_ratio <= 8.0
                        valid_area = 0.1 <= area_percentage <= 60.0

                        if valid_aspect_ratio and valid_area:
                            # Add to detections list with mapped coordinates
                            detection = {
                                "box": [int(coord) for coord in [ox1, oy1, ox2, oy2]],
                                "confidence": float(conf),
                                "region_index": idx,
                                "aspect_ratio": float(aspect_ratio),
                                "area_percentage": float(area_percentage)
                            }
                            response["detections"].append(detection)

                            # Update best detection if this one has higher confidence
                            if conf > highest_conf:
                                highest_conf = conf
                                best_detection = detection
                        else:
                            if not valid_aspect_ratio:
                                logger.debug(f"Filtered out detection with aspect ratio: {aspect_ratio:.2f}")
                            if not valid_area:
                                logger.debug(f"Filtered out detection with area percentage: {area_percentage:.2f}%")

            except Exception as e:
                logger.error(f"Error processing region {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Calculate detection time
        detection_time = time.time() - start_time
        response["detection_time"] = detection_time

        # Set highest confidence detection as the result
        if best_detection:
            response["success"] = True
            response["confidence"] = best_detection["confidence"]

            # Extract region from the original image
            region_idx = best_detection["region_index"]
            region_img = regions[region_idx]

            # Extract license plate from region using detected box
            box = best_detection["box"]
            x1, y1, x2, y2 = box

            # Ensure coordinates are within bounds
            h, w = region_img.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            # Extract license plate image
            if x1 < x2 and y1 < y2:  # Valid box
                license_plate_img = region_img[y1:y2, x1:x2]

                # Convert to RGB and encode as base64
                license_plate_rgb = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', license_plate_rgb)
                license_plate_b64 = base64.b64encode(buffer).decode('utf-8')

                response["license_plate"] = license_plate_b64

        logger.info(f"License plate detection completed in {detection_time:.2f}s with {len(response['detections'])} detections")
        return response

    except Exception as e:
        error_msg = f"Error in license plate detection: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        response["error"] = error_msg
        return response