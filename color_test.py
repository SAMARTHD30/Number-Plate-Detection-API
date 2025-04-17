import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def simulate_pipeline_with_issue(image_path):
    """
    Simulate the color conversion pipeline with the potential issue

    Args:
        image_path: Path to test image

    Returns:
        Tuple of original and processed images for comparison
    """
    # Load image in BGR (as OpenCV normally does)
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Save original for comparison
    original = bgr_image.copy()

    print("Original image shape:", bgr_image.shape)

    # Step 1: focus_license_plate_regions converts BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print("After BGR->RGB conversion (region extraction):", rgb_image.shape)

    # Step 2: preprocess_image assumes BGR input but gets RGB, converts "BGR" to RGB
    # This is a potential issue - we're converting RGB to RGB
    rgb_image_again = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print("After RGB->RGB conversion (preprocess):", rgb_image_again.shape)

    # Step 3: enhance_image operates on this wrongly converted image
    # Simulate basic enhancement by converting to HSV and back
    hsv = cv2.cvtColor(rgb_image_again, cv2.COLOR_RGB2HSV)
    enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    print("After RGB->HSV->RGB conversion (enhance):", enhanced_rgb.shape)

    # To visualize, convert everything back to BGR for saving with OpenCV
    result_with_issue = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    return original, result_with_issue

def simulate_pipeline_fixed(image_path):
    """
    Simulate the color conversion pipeline with the fix for the issue

    Args:
        image_path: Path to test image

    Returns:
        Tuple of original and processed images for comparison
    """
    # Load image in BGR (as OpenCV normally does)
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Save original for comparison
    original = bgr_image.copy()

    # Step 1: focus_license_plate_regions converts BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Step 2: preprocess_image knows input is RGB, skips additional conversion
    rgb_image_no_double_conversion = rgb_image.copy()

    # Step 3: enhance_image operates on the correctly formatted RGB image
    hsv = cv2.cvtColor(rgb_image_no_double_conversion, cv2.COLOR_RGB2HSV)
    enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # To visualize, convert everything back to BGR for saving with OpenCV
    result_fixed = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    return original, result_fixed

def simulate_extreme_case(image_path):
    """
    Simulate an extreme case with multiple incorrect color conversions

    Args:
        image_path: Path to test image
    """
    # Load image in BGR
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Save original
    original = bgr_image.copy()

    # Create a series of 10 improper conversions (RGB -> BGR treated as BGR -> RGB)
    result = bgr_image.copy()
    for i in range(10):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # Treat RGB as BGR and convert "BGR" to RGB again
        result = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Save intermediate result every few steps
        if i in [0, 2, 5, 9]:
            cv2.imwrite(f"extreme_conversion_{i+1}.jpg", result)

    # Final comparison
    return original, result

def test_image(image_path):
    """Run all tests on an image"""
    # Create output folder if it doesn't exist
    output_dir = Path("color_test_output")
    output_dir.mkdir(exist_ok=True)

    # Get base filename
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)

    print(f"Testing color conversions on: {basename}")

    # Test 1: Current pipeline with potential color issue
    print("\nTesting current pipeline...")
    original, result_with_issue = simulate_pipeline_with_issue(image_path)
    cv2.imwrite(str(output_dir / f"{name}_original.jpg"), original)
    cv2.imwrite(str(output_dir / f"{name}_with_issue.jpg"), result_with_issue)

    # Test 2: Fixed pipeline
    print("\nTesting fixed pipeline...")
    _, result_fixed = simulate_pipeline_fixed(image_path)
    cv2.imwrite(str(output_dir / f"{name}_fixed.jpg"), result_fixed)

    # Test 3: Extreme case showing impact of multiple incorrect conversions
    print("\nTesting extreme case...")
    _, result_extreme = simulate_extreme_case(image_path)
    cv2.imwrite(str(output_dir / f"{name}_extreme.jpg"), result_extreme)

    # Calculate difference images to highlight the impact
    diff_issue_vs_fixed = cv2.absdiff(result_with_issue, result_fixed)
    cv2.imwrite(str(output_dir / f"{name}_diff_issue_vs_fixed.jpg"), diff_issue_vs_fixed)

    diff_original_vs_issue = cv2.absdiff(original, result_with_issue)
    cv2.imwrite(str(output_dir / f"{name}_diff_original_vs_issue.jpg"), diff_original_vs_issue)

    diff_original_vs_extreme = cv2.absdiff(original, result_extreme)
    cv2.imwrite(str(output_dir / f"{name}_diff_original_vs_extreme.jpg"), diff_original_vs_extreme)

    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print(f"  - {name}_original.jpg (Original image)")
    print(f"  - {name}_with_issue.jpg (Current pipeline with potential color issue)")
    print(f"  - {name}_fixed.jpg (Fixed pipeline)")
    print(f"  - {name}_extreme.jpg (Extreme case with multiple conversions)")
    print(f"  - {name}_diff_issue_vs_fixed.jpg (Difference between current and fixed)")
    print(f"  - {name}_diff_original_vs_issue.jpg (Difference between original and current)")
    print(f"  - {name}_diff_original_vs_extreme.jpg (Difference between original and extreme)")

def analyze_histogram_differences(image_path):
    """
    Analyze color histogram differences between correct and incorrect conversion pipelines

    Args:
        image_path: Path to test image
    """
    # Load image
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Apply current pipeline with issue
    _, result_with_issue = simulate_pipeline_with_issue(image_path)

    # Apply fixed pipeline
    _, result_fixed = simulate_pipeline_fixed(image_path)

    # Convert to HSV for better color analysis
    hsv_original = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_with_issue = cv2.cvtColor(result_with_issue, cv2.COLOR_BGR2HSV)
    hsv_fixed = cv2.cvtColor(result_fixed, cv2.COLOR_BGR2HSV)

    # Calculate histograms
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]  # Use hue and saturation channels

    hist_original = cv2.calcHist([hsv_original], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_original, hist_original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_with_issue = cv2.calcHist([hsv_with_issue], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_with_issue, hist_with_issue, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_fixed = cv2.calcHist([hsv_fixed], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_fixed, hist_fixed, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Compare histograms
    comparison_original_vs_issue = cv2.compareHist(hist_original, hist_with_issue, cv2.HISTCMP_CORREL)
    comparison_original_vs_fixed = cv2.compareHist(hist_original, hist_fixed, cv2.HISTCMP_CORREL)
    comparison_issue_vs_fixed = cv2.compareHist(hist_with_issue, hist_fixed, cv2.HISTCMP_CORREL)

    print("\nHistogram Comparison Results (correlation, higher is more similar):")
    print(f"  - Original vs Current Pipeline: {comparison_original_vs_issue:.4f}")
    print(f"  - Original vs Fixed Pipeline: {comparison_original_vs_fixed:.4f}")
    print(f"  - Current vs Fixed Pipeline: {comparison_issue_vs_fixed:.4f}")

    # Calculate HSV channel means to see color shifts
    h_mean_original = np.mean(hsv_original[:,:,0])
    s_mean_original = np.mean(hsv_original[:,:,1])
    v_mean_original = np.mean(hsv_original[:,:,2])

    h_mean_with_issue = np.mean(hsv_with_issue[:,:,0])
    s_mean_with_issue = np.mean(hsv_with_issue[:,:,1])
    v_mean_with_issue = np.mean(hsv_with_issue[:,:,2])

    h_mean_fixed = np.mean(hsv_fixed[:,:,0])
    s_mean_fixed = np.mean(hsv_fixed[:,:,1])
    v_mean_fixed = np.mean(hsv_fixed[:,:,2])

    print("\nHSV Channel Means:")
    print(f"  - Original: H={h_mean_original:.2f}, S={s_mean_original:.2f}, V={v_mean_original:.2f}")
    print(f"  - Current Pipeline: H={h_mean_with_issue:.2f}, S={s_mean_with_issue:.2f}, V={v_mean_with_issue:.2f}")
    print(f"  - Fixed Pipeline: H={h_mean_fixed:.2f}, S={s_mean_fixed:.2f}, V={v_mean_fixed:.2f}")

    # Calculate HSV differences
    h_diff_issue = abs(h_mean_original - h_mean_with_issue)
    s_diff_issue = abs(s_mean_original - s_mean_with_issue)
    v_diff_issue = abs(v_mean_original - v_mean_with_issue)

    h_diff_fixed = abs(h_mean_original - h_mean_fixed)
    s_diff_fixed = abs(s_mean_original - s_mean_fixed)
    v_diff_fixed = abs(v_mean_original - v_mean_fixed)

    print("\nHSV Channel Differences from Original:")
    print(f"  - Current Pipeline: H={h_diff_issue:.2f}, S={s_diff_issue:.2f}, V={v_diff_issue:.2f}")
    print(f"  - Fixed Pipeline: H={h_diff_fixed:.2f}, S={s_diff_fixed:.2f}, V={v_diff_fixed:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test color conversion issues in license plate detection")
    parser.add_argument("image_path", help="Path to the test image")
    args = parser.parse_args()

    # Run tests
    test_image(args.image_path)

    # Analyze histogram differences
    analyze_histogram_differences(args.image_path)