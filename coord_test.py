import os
import cv2
import numpy as np
from pathlib import Path
from app.core.model import get_model, preprocess_image
from app.api.routes import focus_license_plate_regions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def map_coordinates_to_original(box, original_shape, processed_shape):
    """
    Map coordinates from processed image space back to original image space

    Args:
        box: [x1, y1, x2, y2] in processed image space
        original_shape: (height, width) of original image
        processed_shape: (height, width) of processed image

    Returns:
        [x1, y1, x2, y2] in original image space
    """
    orig_h, orig_w = original_shape[:2]
    proc_h, proc_w = processed_shape[:2]

    # Calculate scaling factor
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    # Map coordinates
    x1, y1, x2, y2 = box
    mapped_x1 = int(x1 * scale_x)
    mapped_y1 = int(y1 * scale_y)
    mapped_x2 = int(x2 * scale_x)
    mapped_y2 = int(y2 * scale_y)

    return [mapped_x1, mapped_y1, mapped_x2, mapped_y2]

def test_coordinate_mapping(image_path):
    """
    Test coordinate mapping between processed and original images

    Args:
        image_path: Path to test image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    # Store original image dimensions
    original_height, original_width = image.shape[:2]
    logger.info(f"Original image dimensions: {original_width}x{original_height}")

    # Create a copy for drawing results
    result_img = image.copy()

    # Extract regions
    regions = focus_license_plate_regions(image)
    logger.info(f"Found {len(regions)} regions")

    # Get the model
    model = get_model()
    if model is None:
        logger.error("Failed to load model")
        return

    # Process each region
    for i, region in enumerate(regions):
        logger.info(f"Processing region {i+1}/{len(regions)}")

        # Store original region dimensions
        region_height, region_width = region.shape[:2]

        # Create a copy for drawing on this region
        region_result = region.copy()

        # Preprocess the region
        processed_region = preprocess_image(region, skip_enhancement=True)
        if processed_region is None:
            logger.warning(f"Failed to preprocess region {i+1}")
            continue

        # Store processed region dimensions
        processed_height, processed_width = processed_region.shape[:2]
        logger.info(f"Region {i+1} dimensions: Original {region_width}x{region_height}, Processed {processed_width}x{processed_height}")

        # Calculate the scaling factor and offsets used in preprocessing
        scale = min(640 / region_width, 640 / region_height)
        new_width = int(region_width * scale)
        new_height = int(region_height * scale)
        x_offset = (640 - new_width) // 2
        y_offset = (640 - new_height) // 2

        logger.info(f"Region {i+1} scaling: scale={scale:.4f}, offsets=({x_offset}, {y_offset})")

        # Create a copy of processed region for visualization
        processed_visual = processed_region.copy()

        # Run detection on processed region
        results = model.predict(source=processed_region, conf=0.1, verbose=False)

        if not results or len(results) == 0:
            logger.info(f"No detections in region {i+1}")
            continue

        # Get the first result
        result = results[0]

        # Check if there are any boxes
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            logger.info(f"No boxes in region {i+1}")
            continue

        # Process each detection
        for j, box in enumerate(result.boxes):
            # Get coordinates in processed image space
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            # Draw on processed image
            cv2.rectangle(processed_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_visual, f"{confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log detection in processed space
            logger.info(f"Detection {j+1} in region {i+1}: "
                       f"Processed coords: ({x1}, {y1}, {x2}, {y2}), conf: {confidence:.2f}")

            # Map coordinates back from processed to region space
            # First, remove the offset
            reg_x1 = (x1 - x_offset) / scale if x1 >= x_offset else 0
            reg_y1 = (y1 - y_offset) / scale if y1 >= y_offset else 0
            reg_x2 = (x2 - x_offset) / scale if x2 >= x_offset else 0
            reg_y2 = (y2 - y_offset) / scale if y2 >= y_offset else 0

            # Ensure coordinates are within region bounds
            reg_x1 = max(0, min(int(reg_x1), region_width - 1))
            reg_y1 = max(0, min(int(reg_y1), region_height - 1))
            reg_x2 = max(0, min(int(reg_x2), region_width - 1))
            reg_y2 = max(0, min(int(reg_y2), region_height - 1))

            # Draw on region image
            cv2.rectangle(region_result, (reg_x1, reg_y1), (reg_x2, reg_y2), (0, 0, 255), 2)

            # Log mapped coordinates
            logger.info(f"Mapped to region space: ({reg_x1}, {reg_y1}, {reg_x2}, {reg_y2})")

            # If this is not the full image region, we need to map coordinates to original image
            if i > 0:  # Regions after the first one are subregions
                # Get region's position in the original image
                if i == 1:  # Bottom half
                    orig_x1 = reg_x1
                    orig_y1 = original_height // 2 + reg_y1
                    orig_x2 = reg_x2
                    orig_y2 = original_height // 2 + reg_y2
                elif i == 2:  # Center region
                    center_x1 = max(0, original_width // 4)
                    center_y1 = max(0, original_height // 3)
                    orig_x1 = center_x1 + reg_x1
                    orig_y1 = center_y1 + reg_y1
                    orig_x2 = center_x1 + reg_x2
                    orig_y2 = center_y1 + reg_y2
                elif i == 3:  # Lower center region
                    center_x1 = max(0, original_width // 4)
                    lower_center_y1 = max(0, int(original_height * 0.6))
                    orig_x1 = center_x1 + reg_x1
                    orig_y1 = lower_center_y1 + reg_y1
                    orig_x2 = center_x1 + reg_x2
                    orig_y2 = lower_center_y1 + reg_y2
            else:  # First region is the full image
                orig_x1, orig_y1, orig_x2, orig_y2 = reg_x1, reg_y1, reg_x2, reg_y2

            # Ensure coordinates are within original image bounds
            orig_x1 = max(0, min(orig_x1, original_width - 1))
            orig_y1 = max(0, min(orig_y1, original_height - 1))
            orig_x2 = max(0, min(orig_x2, original_width - 1))
            orig_y2 = max(0, min(orig_y2, original_height - 1))

            # Draw on result image
            cv2.rectangle(result_img, (orig_x1, orig_y1), (orig_x2, orig_y2), (255, 0, 0), 2)
            cv2.putText(result_img, f"R{i+1}D{j+1}: {confidence:.2f}", (orig_x1, orig_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Log mapped coordinates to original image
            logger.info(f"Mapped to original image: ({orig_x1}, {orig_y1}, {orig_x2}, {orig_y2})")

        # Save processed visual
        cv2.imwrite(f"region_{i+1}_processed.jpg", processed_visual)
        # Save region result
        cv2.imwrite(f"region_{i+1}_result.jpg", cv2.cvtColor(region_result, cv2.COLOR_RGB2BGR))

    # Save final result
    cv2.imwrite("final_result.jpg", result_img)
    logger.info(f"Results saved to final_result.jpg")

def test_multiple_images(count=3):
    """Test coordinate mapping on multiple images"""
    # Get sample images
    dataset_path = Path("Dataset")
    image_files = list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.jpg"))

    if not image_files:
        logger.error("No image files found in Dataset folder")
        return

    # Select a subset of images
    import random
    if len(image_files) > count:
        test_images = random.sample(image_files, count)
    else:
        test_images = image_files

    # Test each image
    for i, img_path in enumerate(test_images):
        logger.info(f"\n\n=== TESTING IMAGE {i+1}/{len(test_images)}: {img_path} ===")
        test_coordinate_mapping(str(img_path))

if __name__ == "__main__":
    logger.info("=== COORDINATE MAPPING TEST ===")
    test_multiple_images(count=3)
    logger.info("Test completed.")