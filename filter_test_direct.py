from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_with_new_filters():
    """Test detection with expanded aspect ratio and area percentage filters"""
    model_path = Path("app/models/best.pt")

    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return

    # Load the model
    logger.info("Loading model...")
    model = YOLO(model_path)

    # Test on multiple images
    image_files = list(Path("Dataset").glob("*.jpeg"))[:5]  # Test first 5 images
    if not image_files:
        logger.error("No images found in Dataset directory")
        return

    # Test each image
    for idx, img_path in enumerate(image_files):
        logger.info(f"\n[{idx+1}/{len(image_files)}] Testing {img_path}")

        # Load the image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            continue

        # Get image dimensions
        height, width = img.shape[:2]
        logger.info(f"Image dimensions: {width}x{height}")

        # Convert to RGB for YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run prediction with lower confidence threshold
        results = model.predict(
            source=img_rgb,
            conf=0.2,  # Our new threshold
            verbose=True
        )

        logger.info(f"Detection results: {len(results)} items")

        # Process results
        for i, result in enumerate(results):
            logger.info(f"Result {i}:")

            if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
                logger.warning("No detections found for this result")
                continue

            # Get all boxes
            boxes = result.boxes
            filtered_boxes = []

            # Apply our new filtering logic
            for j, box in enumerate(boxes):
                try:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    box_width = x2 - x1
                    box_height = y2 - y1

                    # Skip if dimensions are invalid
                    if box_width <= 0 or box_height <= 0:
                        continue

                    # Calculate aspect ratio (width/height)
                    aspect_ratio = box_width / box_height

                    # Calculate relative area
                    img_area = width * height
                    box_area = box_width * box_height
                    area_percentage = (box_area / img_area) * 100

                    # Apply our expanded filters: aspect_ratio between 1.0-8.0, area_percentage between 0.2-30%
                    if 1.0 <= aspect_ratio <= 8.0 and 0.2 <= area_percentage <= 30.0:
                        confidence = float(box.conf[0])
                        filtered_boxes.append({
                            "box_idx": j,
                            "confidence": confidence,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "aspect_ratio": aspect_ratio,
                            "area_percentage": area_percentage
                        })
                        logger.info(f"  Candidate plate {j}: confidence={confidence:.4f}, aspect={aspect_ratio:.2f}, area={area_percentage:.2f}%")
                    else:
                        logger.debug(f"  Rejected box {j}: aspect={aspect_ratio:.2f}, area={area_percentage:.2f}%")
                except Exception as e:
                    logger.error(f"Error processing box {j}: {str(e)}")

            # Sort by confidence
            filtered_boxes.sort(key=lambda x: x["confidence"], reverse=True)

            # Report results
            if filtered_boxes:
                logger.info(f"Found {len(filtered_boxes)} license plates after filtering")
                for idx, box_info in enumerate(filtered_boxes):
                    logger.info(f"  Plate {idx+1}: confidence={box_info['confidence']:.4f}, " +
                              f"aspect={box_info['aspect_ratio']:.2f}, area={box_info['area_percentage']:.2f}%")

                # Draw on the image
                output_img = img.copy()
                for box_info in filtered_boxes:
                    # Draw rectangle
                    cv2.rectangle(output_img,
                                (box_info["x1"], box_info["y1"]),
                                (box_info["x2"], box_info["y2"]),
                                (0, 255, 0), 2)

                    # Draw text
                    cv2.putText(output_img,
                              f"{box_info['confidence']:.2f}",
                              (box_info["x1"], box_info["y1"] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the annotated image
                output_path = f"new_filter_result_{os.path.basename(str(img_path))}"
                cv2.imwrite(output_path, output_img)
                logger.info(f"Saved annotated image to {output_path}")
            else:
                logger.warning("No license plates found after applying filters")

    logger.info("\nTesting with new filters completed")

if __name__ == "__main__":
    test_with_new_filters()