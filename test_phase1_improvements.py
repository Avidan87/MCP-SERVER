"""
Test Script for Phase 1 MiDaS Accuracy Improvements
Validates reference detection, food heights, and portion estimation

Usage:
    python test_phase1_improvements.py

Requirements:
    - OpenCV (cv2)
    - NumPy
    - Pillow (PIL)
    - Test images in test_images/ directory
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import logging
from typing import Dict, Any

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import our improved modules
from reference_detector import ReferenceObjectDetector, detect_reference_object
from nigerian_food_heights import get_food_height, NIGERIAN_FOOD_HEIGHTS, SHAPE_DEFAULT_HEIGHTS
from portion_calculator import PortionCalculator, estimate_portion_from_depth

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def test_reference_detector():
    """Test 1: Reference Object Detection (Plate/Bowl Detection)"""
    print_section("TEST 1: Reference Object Detection")

    detector = ReferenceObjectDetector()

    # Test 1a: Verify Hough Circle detection works
    print("1a. Testing Hough Circle Detection Algorithm\n")

    # Create synthetic test image with a white circle (simulated plate)
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(test_img, (400, 300), 150, (255, 255, 255), -1)

    circles = detector.detect_circular_objects(test_img, min_radius=100, max_radius=200)

    if len(circles) > 0:
        print(f"   ✓ Circle detection WORKS: Found {len(circles)} circle(s)")
        x, y, r = circles[0]
        print(f"   Circle: center=({x}, {y}), radius={r}px")
    else:
        print("   ✗ FAILED: No circles detected in synthetic image")
        return False

    # Test 1b: Classification logic
    print("\n1b. Testing Reference Object Classification\n")

    test_cases = [
        (150, 800, "plate_small"),   # 300px diameter / 800px width = 37.5% → plate_small (0.25-0.4)
        (250, 800, "plate_large"),   # 500px diameter / 800px width = 62.5% → plate_large (>0.6)
        (80, 800, "bowl_medium"),    # 160px diameter / 800px width = 20% → bowl_medium (0.15-0.25)
        (50, 800, "bowl_small"),     # 100px diameter / 800px width = 12.5% → bowl_small (<0.15)
    ]

    all_passed = True
    for radius, width, expected_type in test_cases:
        obj_type, size_cm = detector.classify_reference_object(radius, width)
        status = "✓" if expected_type in obj_type else "✗"
        print(f"   {status} Radius {radius}px → {obj_type} ({size_cm}cm)")
        if expected_type not in obj_type:
            all_passed = False

    # Test 1c: Full pipeline
    print("\n1c. Testing Full Reference Detection Pipeline\n")

    reference_info = detector.find_best_reference(test_img)

    if reference_info:
        print(f"   ✓ Reference detection: {reference_info['object_type']}")
        print(f"   Real size: {reference_info['real_size_cm']}cm")
        print(f"   Pixel-to-cm ratio: {reference_info['pixel_to_cm_ratio']:.4f}")
        print(f"   Confidence: {reference_info['confidence']:.2f}")
    else:
        print("   ✗ FAILED: No reference detected")
        all_passed = False

    print("\n" + "─" * 70)
    print(f"TEST 1 RESULT: {'✓ PASSED' if all_passed else '✗ FAILED'}\n")
    return all_passed


def test_nigerian_food_heights():
    """Test 2: Nigerian Food Heights Database"""
    print_section("TEST 2: Nigerian Food Heights Database")

    # Test 2a: Known foods
    print("2a. Testing Known Nigerian Foods\n")

    test_foods = [
        ("fufu", "typical", 9.0, "mound"),
        ("jollof_rice", "typical", 5.0, "mound"),
        ("egusi_soup", "typical", 4.0, "bowl"),
        ("fried_plantain", "typical", 2.0, "flat_stack"),
        ("pounded_yam", "large", 14.0, "mound"),
    ]

    all_passed = True
    for food_name, portion, expected_height, expected_shape in test_foods:
        height, shape = get_food_height(food_name, portion_size=portion)
        status = "✓" if height == expected_height and shape == expected_shape else "✗"
        print(f"   {status} {food_name} ({portion}): {height}cm ({shape})")
        if height != expected_height or shape != expected_shape:
            all_passed = False

    # Test 2b: Portion size variations
    print("\n2b. Testing Portion Size Variations\n")

    min_h, typ_h, max_h = 6.0, 9.0, 12.0

    small_height, _ = get_food_height("fufu", "small")
    typical_height, _ = get_food_height("fufu", "typical")
    large_height, _ = get_food_height("fufu", "large")

    portion_test = (small_height == min_h and typical_height == typ_h and large_height == max_h)

    print(f"   {'✓' if portion_test else '✗'} Fufu heights: small={small_height}cm, typical={typical_height}cm, large={large_height}cm")

    if not portion_test:
        all_passed = False

    # Test 2c: Unknown food fallback
    print("\n2c. Testing Unknown Food Fallback\n")

    unknown_height, unknown_shape = get_food_height("unknown_weird_food", "typical")

    fallback_test = unknown_shape == "mound" and unknown_height == 7.0
    print(f"   {'✓' if fallback_test else '✗'} Unknown food defaults to: {unknown_height}cm ({unknown_shape})")

    if not fallback_test:
        all_passed = False

    # Test 2d: Database completeness
    print("\n2d. Testing Database Completeness\n")

    total_foods = len(NIGERIAN_FOOD_HEIGHTS)
    print(f"   Foods in database: {total_foods}")

    # Check we have all major categories
    categories = {
        "swallows": ["fufu", "pounded_yam", "eba", "amala"],
        "rice": ["jollof_rice", "fried_rice", "white_rice"],
        "soups": ["egusi_soup", "okra_soup", "ogbono_soup", "efo_riro"],
        "proteins": ["grilled_chicken", "fried_fish", "suya"],
    }

    category_completeness = True
    for category, foods in categories.items():
        missing = [f for f in foods if f not in NIGERIAN_FOOD_HEIGHTS]
        if missing:
            print(f"   ✗ Missing {category}: {', '.join(missing)}")
            category_completeness = False
        else:
            print(f"   ✓ {category.capitalize()}: All present")

    if not category_completeness:
        all_passed = False

    print("\n" + "─" * 70)
    print(f"TEST 2 RESULT: {'✓ PASSED' if all_passed else '✗ FAILED'}\n")
    return all_passed


def test_portion_calculator_integration():
    """Test 3: Portion Calculator Integration"""
    print_section("TEST 3: Portion Calculator Integration")

    # Create synthetic test scenario
    print("3a. Testing Scale Calibration Integration\n")

    # Synthetic image with plate
    img_height, img_width = 600, 800
    test_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    cv2.circle(test_img, (400, 300), 120, (255, 255, 255), -1)

    calculator = PortionCalculator()

    # Test calibration
    calibrated = calculator.calibrate_scale(test_img, reference_object="plate")

    print(f"   Reference detected: {calculator.reference_detected}")
    print(f"   Pixel-to-cm ratio: {calculator.pixel_to_cm_ratio:.4f}")
    print(f"   Calibration confidence: {calculator.calibration_confidence:.2f}")

    calibration_test = calculator.pixel_to_cm_ratio is not None
    print(f"   {'✓' if calibration_test else '✗'} Calibration {'successful' if calibration_test else 'failed'}\n")

    # Test 3b: Volume calculation with food-specific heights
    print("3b. Testing Volume Calculation with Food Heights\n")

    # Create synthetic depth map (mound shape)
    depth_map = np.zeros((img_height, img_width), dtype=np.float32)
    y, x = np.ogrid[:img_height, :img_width]

    # Create circular mound
    center_x, center_y = 400, 300
    radius = 100
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    depth_map = np.maximum(0, 1 - distance / radius)

    # Create food mask
    food_mask = (depth_map > 0).astype(np.uint8) * 255

    # Test with different food types
    test_foods = [
        ("fufu", 9.0),
        ("jollof_rice", 5.0),
        ("egusi_soup", 4.0),
    ]

    volume_test = True
    for food_type, expected_height in test_foods:
        volume = calculator.calculate_volume_from_depth(
            depth_map,
            food_mask,
            calculator.pixel_to_cm_ratio,
            food_type=food_type
        )

        print(f"   {food_type}: {volume:.1f}ml (using {expected_height}cm height)")

        if volume <= 0:
            volume_test = False

    print(f"\n   {'✓' if volume_test else '✗'} Volume calculation {'works' if volume_test else 'failed'}\n")

    # Test 3c: Full portion estimation pipeline
    print("3c. Testing Full Portion Estimation Pipeline\n")

    result = calculator.estimate_portion(
        test_img,
        depth_map,
        food_type="jollof_rice",
        reference_object="plate"
    )

    print(f"   Volume: {result['volume_ml']:.1f}ml")
    print(f"   Weight: {result['weight_grams']:.1f}g")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reference detected: {result['reference_detected']}")
    print(f"   Food pixels: {result['food_pixels']}")

    pipeline_test = (
        result['volume_ml'] > 0 and
        result['weight_grams'] > 0 and
        0 <= result['confidence'] <= 1.0
    )

    print(f"\n   {'✓' if pipeline_test else '✗'} Pipeline {'complete' if pipeline_test else 'failed'}\n")

    all_passed = calibration_test and volume_test and pipeline_test

    print("─" * 70)
    print(f"TEST 3 RESULT: {'✓ PASSED' if all_passed else '✗ FAILED'}\n")
    return all_passed


def test_confidence_scoring():
    """Test 4: Confidence Scoring Accuracy"""
    print_section("TEST 4: Confidence Scoring Accuracy")

    calculator = PortionCalculator()

    # Test 4a: High confidence scenario (plate detected + food type specified)
    print("4a. High Confidence Scenario\n")

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(img, (400, 300), 150, (255, 255, 255), -1)

    depth_map = np.random.rand(600, 800).astype(np.float32)

    high_conf_result = calculator.estimate_portion(
        img, depth_map,
        food_type="jollof_rice",
        reference_object="plate"
    )

    print(f"   Confidence: {high_conf_result['confidence']:.2f}")
    print(f"   Reference detected: {high_conf_result['reference_detected']}")

    high_conf_test = high_conf_result['confidence'] >= 0.5
    print(f"   {'✓' if high_conf_test else '✗'} High confidence {'achieved' if high_conf_test else 'failed'}\n")

    # Test 4b: Low confidence scenario (no plate detected)
    print("4b. Low Confidence Scenario (No Reference)\n")

    no_plate_img = np.zeros((600, 800, 3), dtype=np.uint8)  # No circles

    low_conf_result = calculator.estimate_portion(
        no_plate_img, depth_map,
        food_type="jollof_rice",
        reference_object="plate"
    )

    print(f"   Confidence: {low_conf_result['confidence']:.2f}")
    print(f"   Reference detected: {low_conf_result['reference_detected']}")

    low_conf_test = low_conf_result['confidence'] <= 0.5
    print(f"   {'✓' if low_conf_test else '✗'} Low confidence {'correct' if low_conf_test else 'incorrect'}\n")

    # Test 4c: Confidence capped at 85%
    print("4c. Confidence Cap at 85%\n")

    cap_test = high_conf_result['confidence'] <= 0.85
    print(f"   Max confidence: {high_conf_result['confidence']:.2f}")
    print(f"   {'✓' if cap_test else '✗'} Confidence {'capped correctly' if cap_test else 'EXCEEDS 85%!'}\n")

    all_passed = high_conf_test and low_conf_test and cap_test

    print("─" * 70)
    print(f"TEST 4 RESULT: {'✓ PASSED' if all_passed else '✗ FAILED'}\n")
    return all_passed


def test_edge_cases():
    """Test 5: Edge Cases and Error Handling"""
    print_section("TEST 5: Edge Cases and Error Handling")

    calculator = PortionCalculator()

    # Test 5a: Empty image
    print("5a. Empty Image (All Black)\n")

    empty_img = np.zeros((600, 800, 3), dtype=np.uint8)
    empty_depth = np.zeros((600, 800), dtype=np.float32)

    try:
        result = calculator.estimate_portion(empty_img, empty_depth)
        print(f"   ✓ Handled gracefully: volume={result['volume_ml']:.1f}ml")
        empty_test = True
    except Exception as e:
        print(f"   ✗ FAILED with error: {e}")
        empty_test = False

    # Test 5b: Very small food region
    print("\n5b. Very Small Food Region\n")

    small_depth = np.zeros((600, 800), dtype=np.float32)
    small_depth[300:305, 400:405] = 1.0  # Tiny 5x5 region

    try:
        small_result = calculator.estimate_portion(empty_img, small_depth)
        print(f"   ✓ Handled gracefully: volume={small_result['volume_ml']:.1f}ml")
        small_test = True
    except Exception as e:
        print(f"   ✗ FAILED with error: {e}")
        small_test = False

    # Test 5c: No reference object (fallback calibration)
    print("\n5c. No Reference Object Detected (Fallback)\n")

    no_ref_result = calculator.estimate_portion(empty_img, empty_depth)

    fallback_test = (
        calculator.pixel_to_cm_ratio is not None and
        not calculator.reference_detected and
        no_ref_result['confidence'] < 0.5
    )

    print(f"   Reference detected: {calculator.reference_detected}")
    print(f"   Fallback ratio: {calculator.pixel_to_cm_ratio:.4f}")
    print(f"   Confidence: {no_ref_result['confidence']:.2f}")
    print(f"   {'✓' if fallback_test else '✗'} Fallback {'works correctly' if fallback_test else 'failed'}\n")

    # Test 5d: Unknown food type
    print("5d. Unknown Food Type\n")

    try:
        unknown_result = calculator.estimate_portion(
            empty_img, empty_depth,
            food_type="totally_unknown_food_12345"
        )
        print(f"   ✓ Handled gracefully: used fallback height")
        unknown_test = True
    except Exception as e:
        print(f"   ✗ FAILED with error: {e}")
        unknown_test = False

    all_passed = empty_test and small_test and fallback_test and unknown_test

    print("\n" + "─" * 70)
    print(f"TEST 5 RESULT: {'✓ PASSED' if all_passed else '✗ FAILED'}\n")
    return all_passed


def run_all_tests():
    """Run all Phase 1 validation tests"""
    print("\n" + "=" * 70)
    print("  PHASE 1 IMPROVEMENTS - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    results = {}

    try:
        results['test_1'] = test_reference_detector()
    except Exception as e:
        print(f"\n✗ TEST 1 CRASHED: {e}\n")
        results['test_1'] = False

    try:
        results['test_2'] = test_nigerian_food_heights()
    except Exception as e:
        print(f"\n✗ TEST 2 CRASHED: {e}\n")
        results['test_2'] = False

    try:
        results['test_3'] = test_portion_calculator_integration()
    except Exception as e:
        print(f"\n✗ TEST 3 CRASHED: {e}\n")
        results['test_3'] = False

    try:
        results['test_4'] = test_confidence_scoring()
    except Exception as e:
        print(f"\n✗ TEST 4 CRASHED: {e}\n")
        results['test_4'] = False

    try:
        results['test_5'] = test_edge_cases()
    except Exception as e:
        print(f"\n✗ TEST 5 CRASHED: {e}\n")
        results['test_5'] = False

    # Final summary
    print_section("FINAL TEST SUMMARY")

    passed = sum(results.values())
    total = len(results)

    print(f"Test 1 - Reference Detection:       {'✓ PASSED' if results['test_1'] else '✗ FAILED'}")
    print(f"Test 2 - Food Heights Database:     {'✓ PASSED' if results['test_2'] else '✗ FAILED'}")
    print(f"Test 3 - Portion Calculator:        {'✓ PASSED' if results['test_3'] else '✗ FAILED'}")
    print(f"Test 4 - Confidence Scoring:        {'✓ PASSED' if results['test_4'] else '✗ FAILED'}")
    print(f"Test 5 - Edge Cases:                {'✓ PASSED' if results['test_5'] else '✗ FAILED'}")

    print("\n" + "─" * 70)
    print(f"\nRESULTS: {passed}/{total} tests passed ({passed/total*100:.0f}%)\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Phase 1 implementation is ready for deployment.\n")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review errors above.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
