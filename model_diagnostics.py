from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import logging
import inspect

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model():
    """Analyze the YOLOv8 model to understand what it's trained for"""
    model_path = Path("app/models/best.pt")

    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return

    logger.info(f"Model file found: {model_path} (Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB)")

    try:
        # Load the model
        logger.info("Loading model...")
        model = YOLO(model_path)

        # Get model info
        logger.info("Model architecture:")
        logger.info(f"Model type: {type(model).__name__}")

        # Check model attributes
        logger.info("Model attributes:")
        # Filter out callable methods and private attributes
        attributes = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith('_')]
        for attr in attributes[:20]:  # Limit to first 20 to avoid too much output
            try:
                value = getattr(model, attr)
                if not callable(value):
                    logger.info(f"  {attr}: {value}")
            except Exception as e:
                logger.warning(f"  {attr}: [Error accessing: {str(e)}]")

        # Check model classes
        if hasattr(model, 'names'):
            logger.info("Model classes:")
            for idx, name in model.names.items():
                logger.info(f"  Class {idx}: {name}")
        else:
            logger.warning("Model doesn't have a 'names' attribute - can't determine classes")

        # Safely check model parameters
        try:
            conf = model.overrides.get('conf', None) or getattr(model, 'conf', None)
            logger.info(f"Default confidence threshold: {conf}")
        except:
            logger.info("Could not determine default confidence threshold")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def test_detection(model, image_path=None):
    """Test detection on a sample image"""
    if image_path is None:
        # Find first image in Dataset folder
        dataset_dir = Path("Dataset")
        image_files = list(dataset_dir.glob("*.jpeg")) + list(dataset_dir.glob("*.jpg"))
        if not image_files:
            logger.error("No images found in Dataset directory")
            return
        image_path = image_files[0]

    logger.info(f"Testing detection on image: {image_path}")

    # Load and preprocess the image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    logger.info(f"Image shape: {img.shape}")

    # Run detection with very low confidence threshold
    try:
        results = model.predict(
            source=img,
            conf=0.2,  # Using our new API threshold instead of very low threshold
            verbose=True
        )

        logger.info(f"Detection results: {len(results)} items")

        # Process results
        for i, result in enumerate(results):
            logger.info(f"Result {i}:")

            if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
                logger.warning("No detections found for this result")
                continue

            # Show all detections
            for j, box in enumerate(result.boxes):
                try:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = model.names.get(cls, f"Unknown-{cls}")

                    logger.info(f"  Detection {j}: class={cls_name}, confidence={conf:.4f}")

                    # Get bounding box
                    if hasattr(box, 'xyxy'):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        logger.info(f"    Bounding box: ({x1}, {y1}) - ({x2}, {y2})")

                        # Draw bounding box for visualization
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{cls_name}: {conf:.2f}",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    logger.error(f"Error processing detection {j}: {str(e)}")

        # Save the annotated image
        output_path = f"detection_result_{os.path.basename(str(image_path))}"
        cv2.imwrite(output_path, img)
        logger.info(f"Saved annotated image to {output_path}")

    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")

def main():
    logger.info("=== YOLOv8 Model Diagnostics ===")

    # Check model
    model = check_model()
    if model is None:
        logger.error("Failed to load model. Please check model file.")
        return

    # Test on multiple images
    logger.info("\n=== Testing Detection on Sample Images ===")

    # Test with a smaller batch of images
    try:
        image_files = list(Path("Dataset").glob("*.jpeg"))[:3]
        if image_files:
            for img_path in image_files:
                test_detection(model, img_path)
        else:
            logger.warning("No images found in Dataset directory")
    except Exception as e:
        logger.error(f"Error testing detection: {str(e)}")

    logger.info("\n=== Diagnosis and Recommendations ===")
    if hasattr(model, 'names'):
        has_license_plate = any('plate' in name.lower() or 'license' in name.lower()
                              for name in model.names.values())

        if has_license_plate:
            logger.info("✓ Model contains license plate related classes")
            logger.info("Since the model has license plate classes but detection is failing, possible issues:")
            logger.info("1. The model may not be effectively trained on data similar to your test images")
            logger.info("2. The model could be expecting different preprocessing steps")
            logger.info("3. There may be issues with how detection results are interpreted in the code")
            logger.info("\nRecommendations:")
            logger.info("1. Try using images with very clear license plates for testing")
            logger.info("2. Check if any preprocessing steps might be incorrect")
            logger.info("3. Verify detection code by examining any successfully detected objects")
        else:
            logger.warning("✗ Model does NOT contain license plate related classes")
            logger.info("Recommendation: You need a model specifically trained for license plate detection.")
            logger.info("Options:")
            logger.info("1. Use a pre-trained license plate detection model")
            logger.info("2. Train your own YOLOv8 model on license plate data")
            logger.info("3. Fine-tune this model on license plate data")

    logger.info("\nDiagnostic complete.")

if __name__ == "__main__":
    main()