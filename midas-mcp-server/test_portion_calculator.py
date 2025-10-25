"""
Test script for Portion Calculator
Tests volume calculation from depth maps with Nigerian food density integration
"""

import numpy as np
import cv2
from portion_calculator import PortionCalculator, estimate_portion_from_depth
from nigerian_food_densities import get_density, NIGERIAN_FOOD_DENSITIES


def test_density_lookup():
    """Test Nigerian food density lookup"""
    print("=== Testing Food Density Lookup ===\n")
    
    test_foods = [
        "jollof-rice",
        "eba",
        "egusi-soup",
        "suya",
        "moi-moi",
        "unknown-food"
    ]
    
    for food in test_foods:
        density = get_density(food)
        print(f"{food:20s} -> {density:.2f} g/ml")
    
    print()


def test_portion_calculator_with_synthetic_data():
    """Test portion calculator with synthetic depth map"""
    print("=== Testing Portion Calculator with Synthetic Data ===\n")
    
    # Create synthetic image and depth map
    height, width = 480, 640
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Create synthetic depth map with a circular food region
    depth_map = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    radius = 100
    
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Create a dome-shaped depth profile (higher in center)
    for i in range(height):
        for j in range(width):
            if mask[i, j]:
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                depth_map[i, j] = 100 * (1 - dist / radius)
    
    # Test with different food types
    test_cases = [
        ("jollof-rice", "plate"),
        ("eba", "hand"),
        ("egusi-soup", None),
    ]
    
    calculator = PortionCalculator()
    
    for food_type, reference in test_cases:
        print(f"Testing: {food_type} with reference: {reference}")
        results = calculator.estimate_portion(
            image=image,
            depth_map=depth_map,
            food_type=food_type,
            reference_object=reference
        )
        
        print(f"  Volume: {results['volume_ml']:.2f} ml")
        print(f"  Weight: {results['weight_grams']:.2f} g")
        print(f"  Confidence: {results['confidence']:.2%}")
        print(f"  Reference detected: {results['reference_detected']}")
        print(f"  Food pixels: {results['food_pixels']}")
        print(f"  Scale: {results['pixel_to_cm_ratio']:.4f} cm/pixel")
        print()


def test_volume_calculation():
    """Test volume calculation logic"""
    print("=== Testing Volume Calculation ===\n")
    
    # Create a simple depth map
    depth_map = np.array([
        [0, 0, 0, 0, 0],
        [0, 50, 100, 50, 0],
        [0, 100, 150, 100, 0],
        [0, 50, 100, 50, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)
    
    # Create mask for food region (center 3x3)
    food_mask = np.zeros((5, 5), dtype=np.uint8)
    food_mask[1:4, 1:4] = 255
    
    calculator = PortionCalculator()
    pixel_to_cm = 0.1  # 1 pixel = 0.1 cm
    
    volume = calculator.calculate_volume_from_depth(
        depth_map=depth_map,
        food_mask=food_mask,
        pixel_to_cm=pixel_to_cm
    )
    
    print(f"Calculated volume: {volume:.2f} ml")
    print(f"Food pixels: {np.sum(food_mask == 255)}")
    print()


def test_reference_sizes():
    """Test reference object sizes"""
    print("=== Testing Reference Object Sizes ===\n")
    
    calculator = PortionCalculator()
    
    print("Available reference objects:")
    for obj, size in calculator.REFERENCE_SIZES.items():
        print(f"  {obj:10s}: {size:.1f} cm")
    print()


def test_all_nigerian_foods():
    """Test density values for all Nigerian foods in database"""
    print("=== All Nigerian Food Densities ===\n")
    
    print(f"{'Food Name':<30s} {'Density (g/ml)'}")
    print("-" * 50)
    
    for food, density in sorted(NIGERIAN_FOOD_DENSITIES.items()):
        print(f"{food:<30s} {density:.2f}")
    
    print()


if __name__ == "__main__":
    print("🍽️  KAI Portion Calculator Test Suite\n")
    print("=" * 60)
    print()
    
    # Run all tests
    test_density_lookup()
    test_reference_sizes()
    test_volume_calculation()
    test_portion_calculator_with_synthetic_data()
    test_all_nigerian_foods()
    
    print("=" * 60)
    print("✅ All tests completed!")
