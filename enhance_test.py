#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import logging
import time
from pathlib import Path
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path=None):
    """
    Load YOLOv8 model directly
    """
    if model_path is None:
        # Default model path
        model_path = os.path.join("app", "models", "best.pt")

    model_path = os.path.abspath(model_path)
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None

    try:
        # Log model file size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {model_size_mb:.2f} MB")

        # Load model
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def enhance_image(image):
    """
    Apply image enhancement techniques with less aggressive processing
    to better preserve license plate features.

    Args:
        image: Image to enhance (RGB format)

    Returns:
        Enhanced image
    """
    try:
        # Make a copy of the input image
        enhanced = image.copy()

        # Calculate image quality metrics
        # Convert to grayscale for analysis only
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate contrast as standard deviation of grayscale pixels
        contrast = np.std(gray)

        # Calculate brightness as mean of grayscale pixels
        brightness = np.mean(gray)

        # Calculate noise level (using Laplacian)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Apply enhancements more selectively and with milder settings
        # Only apply brightness enhancement if image is very dark
        if brightness < 50:  # More strict threshold (was 80)
            # Use CLAHE on the value channel of HSV with milder settings
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            # Reduced clip limit from 3.0 to 2.0
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Only apply contrast enhancement if contrast is very low
        if contrast < 40:  # More strict threshold (was 60)
            # Use CLAHE on the value channel of HSV with milder settings
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            # Reduced clip limit from 3.0 to 2.0
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Only apply noise reduction if image is very noisy
        if noise_level > 1000:  # More strict threshold (was 500)
            # Use bilateral filter with smaller kernel for less aggressive smoothing
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)  # Was (9, 75, 75)

        # Apply milder sharpening to all images to enhance edges
        # Changed from [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]] to:
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced, {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'noise_level': float(noise_level)
        }
    except Exception as e:
        logger.error(f"Error in image enhancement: {str(e)}")
        return image, None  # Return original image if enhancement fails

def preprocess_image(image, target_size=1024, apply_enhancement=False):
    """
    Preprocess image for model inference
    """
    try:
        if image is None:
            logger.error("Received None image in preprocess_image")
            return None, None

        # Store original dimensions
        original_height, original_width = image.shape[:2]

        # Convert BGR to RGB (YOLOv8 expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Apply enhancement if requested
        enhancement_stats = None
        if apply_enhancement:
            rgb_image, enhancement_stats = enhance_image(rgb_image)

        # Calculate scale to maintain aspect ratio
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a square canvas with black padding
        square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Center the resized image on the square canvas
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        # Create transform parameters dictionary for coordinate mapping
        transform_params = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "original_size": (original_height, original_width),
            "enhancement_applied": apply_enhancement,
            "enhancement_stats": enhancement_stats
        }

        return square_img, transform_params

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def map_coordinates_to_original(box, transform_params):
    """
    Map coordinates from processed image space back to original image space
    """
    try:
        if not transform_params:
            return box

        scale = transform_params["scale"]
        x_offset = transform_params["x_offset"]
        y_offset = transform_params["y_offset"]

        # Unpack box coordinates
        x1, y1, x2, y2 = box

        # Remove offset (account for padding in the square canvas)
        x1 = (x1 - x_offset) if x1 >= x_offset else 0
        y1 = (y1 - y_offset) if y1 >= y_offset else 0
        x2 = (x2 - x_offset) if x2 >= x_offset else 0
        y2 = (y2 - y_offset) if y2 >= y_offset else 0

        # Reverse the scaling
        if scale > 0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        # Ensure coordinates are within original image bounds
        original_height, original_width = transform_params["original_size"]
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        return [x1, y1, x2, y2]

    except Exception as e:
        logger.error(f"Error mapping coordinates: {str(e)}")
        return box

def detect_plate(image_path, model, apply_enhancement=False, conf_threshold=0.2, filter_params=None):
    """
    Detect license plates with or without enhancement
    """
    if filter_params is None:
        filter_params = {
            'min_aspect_ratio': 1.0,
            'max_aspect_ratio': 8.0,
            'min_area_percentage': 0.1,
            'max_area_percentage': 60.0
        }

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Preprocess image (with or without enhancement)
        processed_img, transform_params = preprocess_image(image, apply_enhancement=apply_enhancement)
        if processed_img is None:
            logger.error(f"Failed to preprocess image: {image_path}")
            return None

        # Run inference
        start_time = time.time()
        results = model.predict(
            source=processed_img,
            conf=conf_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time

        # Initialize results dict
        result_dict = {
            "image_path": image_path,
            "enhancement_applied": apply_enhancement,
            "inference_time": inference_time,
            "detection_found": False,
        }

        # Add enhancement stats if available
        if transform_params and transform_params.get("enhancement_stats"):
            result_dict["enhancement_stats"] = transform_params["enhancement_stats"]

        # Process results
        all_detections = []

        if not results or len(results) == 0:
            result_dict["detections"] = []
            return result_dict

        result = results[0]

        # Check if there are any boxes
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            result_dict["detections"] = []
            return result_dict

        # Process all boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            # Get coordinates in model space
            x1, y1, x2, y2 = map(int, box)

            # Map back to original image space
            orig_coords = map_coordinates_to_original([x1, y1, x2, y2], transform_params)
            ox1, oy1, ox2, oy2 = orig_coords

            # Calculate metrics
            width = ox2 - ox1
            height = oy2 - oy1
            aspect_ratio = width / height if height > 0 else 0

            img_area = image.shape[0] * image.shape[1]
            box_area = width * height
            area_percentage = (box_area / img_area) * 100

            # Check if detection passes filters
            passes_filters = (
                filter_params['min_aspect_ratio'] <= aspect_ratio <= filter_params['max_aspect_ratio'] and
                filter_params['min_area_percentage'] <= area_percentage <= filter_params['max_area_percentage']
            )

            # Create detection dict
            detection = {
                "confidence": float(conf),
                "box": [int(c) for c in orig_coords],
                "aspect_ratio": float(aspect_ratio),
                "area_percentage": float(area_percentage),
                "passes_filters": passes_filters
            }
            all_detections.append(detection)

        # Sort by confidence
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Find best detection that passes filters
        valid_detections = [d for d in all_detections if d["passes_filters"]]
        best_detection = valid_detections[0] if valid_detections else None

        # Update result dict
        result_dict["detection_found"] = best_detection is not None
        result_dict["detections"] = all_detections
        result_dict["valid_detections"] = valid_detections
        if best_detection:
            result_dict["best_detection"] = best_detection
            result_dict["confidence"] = best_detection["confidence"]
            result_dict["bounding_box"] = best_detection["box"]

        return result_dict

    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def compare_enhancement_impact(image_path, model, conf_threshold=0.2, filter_params=None):
    """
    Compare detection with and without enhancement
    """
    # Run detection both ways
    with_enhancement = detect_plate(image_path, model, apply_enhancement=True, conf_threshold=conf_threshold, filter_params=filter_params)
    without_enhancement = detect_plate(image_path, model, apply_enhancement=False, conf_threshold=conf_threshold, filter_params=filter_params)

    if with_enhancement is None or without_enhancement is None:
        logger.error(f"Detection failed for {image_path}")
        return None

    # Compare results
    result = {
        "image_path": image_path,
        "with_enhancement": with_enhancement,
        "without_enhancement": without_enhancement
    }

    # Check for discrepancies
    with_detected = with_enhancement["detection_found"]
    without_detected = without_enhancement["detection_found"]
    result["detection_difference"] = with_detected != without_detected

    # Compare confidences if both detected
    if with_detected and without_detected:
        with_conf = with_enhancement["confidence"]
        without_conf = without_enhancement["confidence"]
        result["confidence_difference"] = with_conf - without_conf
        result["confidence_ratio"] = with_conf / without_conf if without_conf > 0 else float('inf')

        # Calculate IoU between detections
        with_box = with_enhancement["bounding_box"]
        without_box = without_enhancement["bounding_box"]

        # Calculate intersection
        x1 = max(with_box[0], without_box[0])
        y1 = max(with_box[1], without_box[1])
        x2 = min(with_box[2], without_box[2])
        y2 = min(with_box[3], without_box[3])

        if x2 >= x1 and y2 >= y1:
            intersection_area = (x2 - x1) * (y2 - y1)
            with_area = (with_box[2] - with_box[0]) * (with_box[3] - with_box[1])
            without_area = (without_box[2] - without_box[0]) * (without_box[3] - without_box[1])
            union_area = with_area + without_area - intersection_area

            iou = intersection_area / union_area if union_area > 0 else 0
            result["iou"] = iou

    return result

def visualize_enhancement_comparison(image_path, comparison_result):
    """
    Visualize detection results with and without enhancement
    """
    if comparison_result is None:
        logger.error(f"No comparison results for {image_path}")
        return

    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image for visualization: {image_path}")
        return

    # Create enhanced and non-enhanced versions for visualization
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_img, _ = enhance_image(rgb_image)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Original image
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    # Enhanced image
    axs[0, 1].imshow(enhanced_img)
    axs[0, 1].set_title("Enhanced Image")
    axs[0, 1].axis('off')

    # Original image with detection
    axs[1, 0].imshow(rgb_image)
    axs[1, 0].set_title("Without Enhancement")
    axs[1, 0].axis('off')

    # Enhanced image with detection
    axs[1, 1].imshow(enhanced_img)
    axs[1, 1].set_title("With Enhancement")
    axs[1, 1].axis('off')

    # Add detection boxes if found
    without_result = comparison_result.get("without_enhancement", {})
    with_result = comparison_result.get("with_enhancement", {})

    # Add box to without enhancement subplot
    if without_result.get("detection_found", False):
        box = without_result["bounding_box"]
        confidence = without_result["confidence"]
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor='g', facecolor='none')
        axs[1, 0].add_patch(rect)
        axs[1, 0].text(box[0], box[1]-10, f"Conf: {confidence:.4f}",
                     color='white', fontsize=10,
                     bbox=dict(facecolor='g', alpha=0.7))
    else:
        axs[1, 0].text(10, 30, "No detection", color='red', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))

    # Add box to with enhancement subplot
    if with_result.get("detection_found", False):
        box = with_result["bounding_box"]
        confidence = with_result["confidence"]
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor='g', facecolor='none')
        axs[1, 1].add_patch(rect)
        axs[1, 1].text(box[0], box[1]-10, f"Conf: {confidence:.4f}",
                     color='white', fontsize=10,
                     bbox=dict(facecolor='g', alpha=0.7))
    else:
        axs[1, 1].text(10, 30, "No detection", color='red', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))

    # Add enhancement metrics if available
    if with_result.get("enhancement_stats"):
        stats = with_result["enhancement_stats"]
        metrics_text = f"Brightness: {stats['brightness']:.1f}\nContrast: {stats['contrast']:.1f}\nNoise: {stats['noise_level']:.1f}"
        axs[0, 1].text(10, 30, metrics_text, color='black', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7))

    # Add comparison info
    if comparison_result.get("detection_difference", False):
        diff_text = "Detection discrepancy!"
        fig.text(0.5, 0.04, diff_text, ha='center', fontsize=14, color='red',
               bbox=dict(facecolor='yellow', alpha=0.5))
    elif comparison_result.get("confidence_difference"):
        diff = comparison_result["confidence_difference"]
        ratio = comparison_result.get("confidence_ratio", 1.0)
        diff_text = f"Confidence diff: {diff:.4f} (ratio: {ratio:.2f}x)"
        color = 'green' if diff > 0 else 'red'
        fig.text(0.5, 0.04, diff_text, ha='center', fontsize=12, color=color,
               bbox=dict(facecolor='white', alpha=0.7))

    # Save visualization
    out_filename = f"enhance_comparison_{Path(image_path).stem}.jpg"
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close(fig)

    logger.info(f"Saved visualization to {out_filename}")

def analyze_enhancement_impact(results):
    """
    Analyze enhancement impact across multiple images
    """
    if not results:
        logger.error("No results to analyze")
        return

    total_images = len(results)

    # Detection counts
    with_enhancement_detections = sum(1 for r in results if r.get("with_enhancement", {}).get("detection_found", False))
    without_enhancement_detections = sum(1 for r in results if r.get("without_enhancement", {}).get("detection_found", False))

    # Detection differences
    detection_differences = sum(1 for r in results if r.get("detection_difference", False))
    improved_with_enhancement = sum(1 for r in results if
                                  r.get("with_enhancement", {}).get("detection_found", False) and
                                  not r.get("without_enhancement", {}).get("detection_found", False))
    worsened_with_enhancement = sum(1 for r in results if
                                  not r.get("with_enhancement", {}).get("detection_found", False) and
                                  r.get("without_enhancement", {}).get("detection_found", False))

    # Confidence differences
    both_detected = [r for r in results if
                   r.get("with_enhancement", {}).get("detection_found", False) and
                   r.get("without_enhancement", {}).get("detection_found", False)]

    confidence_diffs = [r.get("confidence_difference", 0) for r in both_detected]
    positive_diffs = [d for d in confidence_diffs if d > 0]
    negative_diffs = [d for d in confidence_diffs if d < 0]

    avg_confidence_diff = sum(confidence_diffs) / len(confidence_diffs) if confidence_diffs else 0
    avg_positive_diff = sum(positive_diffs) / len(positive_diffs) if positive_diffs else 0
    avg_negative_diff = sum(negative_diffs) / len(negative_diffs) if negative_diffs else 0

    # Create summary
    summary = {
        "total_images": total_images,
        "with_enhancement_detections": with_enhancement_detections,
        "without_enhancement_detections": without_enhancement_detections,
        "detection_differences": detection_differences,
        "improved_with_enhancement": improved_with_enhancement,
        "worsened_with_enhancement": worsened_with_enhancement,
        "both_detected_count": len(both_detected),
        "avg_confidence_difference": avg_confidence_diff,
        "avg_positive_confidence_difference": avg_positive_diff,
        "avg_negative_confidence_difference": avg_negative_diff,
        "improved_confidence_count": len(positive_diffs),
        "worsened_confidence_count": len(negative_diffs)
    }

    # Calculate percentages
    if total_images > 0:
        summary["with_enhancement_detection_rate"] = (with_enhancement_detections / total_images) * 100
        summary["without_enhancement_detection_rate"] = (without_enhancement_detections / total_images) * 100
        summary["detection_difference_rate"] = (detection_differences / total_images) * 100

    # List images with notable differences
    improved_images = [r["image_path"] for r in results if
                     r.get("with_enhancement", {}).get("detection_found", False) and
                     not r.get("without_enhancement", {}).get("detection_found", False)]

    worsened_images = [r["image_path"] for r in results if
                     not r.get("with_enhancement", {}).get("detection_found", False) and
                     r.get("without_enhancement", {}).get("detection_found", False)]

    significant_improvement = [r["image_path"] for r in both_detected if r.get("confidence_difference", 0) > 0.1]
    significant_worsening = [r["image_path"] for r in both_detected if r.get("confidence_difference", 0) < -0.1]

    summary["improved_images"] = improved_images
    summary["worsened_images"] = worsened_images
    summary["significant_improvement_images"] = significant_improvement
    summary["significant_worsening_images"] = significant_worsening

    return summary

def main():
    """
    Main entry point
    """
    # Get image files
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Use default paths - sample images
        dataset_dir = Path("Dataset")
        all_images = list(dataset_dir.glob("*.jpeg"))

        # Use a random sample
        import random
        random.seed(42)  # For reproducibility
        sample_size = min(15, len(all_images))
        image_paths = [str(p) for p in random.sample(all_images, sample_size)]

    # Load model once
    model = load_model()
    if model is None:
        logger.error("Failed to load model")
        return

    # Run comparisons
    results = []

    for path in tqdm(image_paths):
        logger.info(f"Processing {path}")
        result = compare_enhancement_impact(path, model)

        if result:
            results.append(result)

            # Log result
            with_detected = result["with_enhancement"]["detection_found"]
            without_detected = result["without_enhancement"]["detection_found"]
            detection_diff = result.get("detection_difference", False)

            with_conf = result["with_enhancement"].get("confidence", 0) if with_detected else 0
            without_conf = result["without_enhancement"].get("confidence", 0) if without_detected else 0

            if detection_diff:
                if with_detected and not without_detected:
                    logger.info(f"Enhancement IMPROVED detection: {path}")
                elif without_detected and not with_detected:
                    logger.info(f"Enhancement WORSENED detection: {path}")
            elif with_detected and without_detected:
                conf_diff = with_conf - without_conf
                if abs(conf_diff) > 0.05:  # Only log significant differences
                    if conf_diff > 0:
                        logger.info(f"Enhancement IMPROVED confidence by {conf_diff:.4f}: {path}")
                    else:
                        logger.info(f"Enhancement WORSENED confidence by {conf_diff:.4f}: {path}")

            # Create visualization
            visualize_enhancement_comparison(path, result)

    # Analyze results
    summary = analyze_enhancement_impact(results)

    # Print summary
    logger.info("\n=== Enhancement Impact Summary ===")
    logger.info(f"Total images: {summary['total_images']}")
    logger.info(f"Detection rate with enhancement: {summary['with_enhancement_detection_rate']:.1f}%")
    logger.info(f"Detection rate without enhancement: {summary['without_enhancement_detection_rate']:.1f}%")
    logger.info(f"Detection differences: {summary['detection_differences']} ({summary['detection_difference_rate']:.1f}%)")
    logger.info(f"Improved with enhancement: {summary['improved_with_enhancement']}")
    logger.info(f"Worsened with enhancement: {summary['worsened_with_enhancement']}")

    if summary['both_detected_count'] > 0:
        logger.info(f"Average confidence difference: {summary['avg_confidence_difference']:.4f}")
        logger.info(f"Improved confidence: {summary['improved_confidence_count']}/{summary['both_detected_count']}")
        logger.info(f"Worsened confidence: {summary['worsened_confidence_count']}/{summary['both_detected_count']}")

    # Save results to JSON
    full_results = {
        "summary": summary,
        "detailed_results": results
    }

    with open("enhancement_impact_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info("Results saved to enhancement_impact_results.json")

if __name__ == "__main__":
    main()