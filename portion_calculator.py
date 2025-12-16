"""
Portion Calculator - Volume estimation from depth maps
Integrates MiDaS depth estimation with Nigerian food density values

ENHANCED: Includes Nigerian food shape priors for improved accuracy
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging
from nigerian_food_densities import get_density, estimate_weight_from_volume
from nigerian_food_priors import NigerianFoodPriors
from reference_detector import detect_reference_object
from nigerian_food_heights import get_food_height, estimate_portion_size_category

logger = logging.getLogger(__name__)


class PortionCalculator:
    """Calculate portion sizes from depth maps and food images"""
    
    # Reference object sizes in cm (for scale calibration)
    REFERENCE_SIZES = {
        "hand": 18.0,  # Average adult hand width in cm
        "plate": 26.0,  # Standard dinner plate diameter in cm
        "spoon": 15.0,  # Standard tablespoon length in cm
        "fork": 18.0,   # Standard fork length in cm
        "phone": 14.0,  # Average smartphone width in cm
    }
    
    def __init__(self):
        self.pixel_to_cm_ratio = None
        self.reference_detected = False
    
    def detect_food_region(self, image: np.ndarray, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the food region in the image using depth and color information
        
        Args:
            image: RGB image array
            depth_map: Depth map from MiDaS
        
        Returns:
            Tuple of (food_mask, food_depth_region)
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Normalize depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply adaptive thresholding to find foreground objects
        # Assuming food is closer to camera (higher depth values in inverse depth)
        _, depth_thresh = cv2.threshold(depth_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_CLOSE, kernel)
        depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(depth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No food region detected, estimating center region")
            # CRITICAL FIX: Instead of using full image, estimate food is in center 40% of image
            # This is much more realistic than treating entire image (including table/background) as food
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 5  # 40% diameter circle in center

            food_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(food_mask, (center_x, center_y), radius, 255, -1)

            return food_mask, np.where(food_mask == 255, depth_map, 0)
        
        # Get the largest contour (assumed to be the food)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask from largest contour
        food_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(food_mask, [largest_contour], -1, 255, -1)
        
        # Extract food depth region
        food_depth = np.where(food_mask == 255, depth_map, 0)
        
        return food_mask, food_depth
    
    def calibrate_scale(self, image: np.ndarray, reference_object: Optional[str] = None) -> bool:
        """
        Calibrate pixel-to-cm ratio using REAL reference object detection

        Args:
            image: RGB image array
            reference_object: Type of reference object hint (e.g., "plate")

        Returns:
            True if calibration successful
        """
        # Use real reference detection (plate/bowl detection via Hough Circles)
        reference_size_cm = None
        if reference_object and reference_object.lower() in self.REFERENCE_SIZES:
            reference_size_cm = self.REFERENCE_SIZES[reference_object.lower()]

        # Detect actual reference object in image
        calibration = detect_reference_object(
            image,
            reference_object=reference_object,
            reference_size_cm=reference_size_cm
        )

        # Extract calibration results
        self.pixel_to_cm_ratio = calibration["pixel_to_cm_ratio"]
        self.reference_detected = calibration.get("detected", False)
        self.calibration_confidence = calibration.get("confidence", 0.3)

        if self.reference_detected:
            logger.info(
                f"✅ Reference detected: {calibration.get('object_type', 'unknown')} "
                f"({calibration.get('real_size_cm', 0)}cm), "
                f"ratio={self.pixel_to_cm_ratio:.4f} cm/pixel, "
                f"confidence={self.calibration_confidence:.2f}"
            )
        else:
            logger.warning(
                f"⚠️ No reference detected, using fallback calibration: "
                f"{self.pixel_to_cm_ratio:.4f} cm/pixel (low confidence)"
            )

        return self.reference_detected
    
    def calculate_volume_from_depth(
        self,
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        pixel_to_cm: float,
        food_type: Optional[str] = None
    ) -> float:
        """
        Calculate volume from depth map using numerical integration
        WITH FOOD-SPECIFIC HEIGHT ESTIMATION

        Args:
            depth_map: Depth map array
            food_mask: Binary mask of food region
            pixel_to_cm: Pixel to centimeter conversion ratio
            food_type: Type of food for height estimation

        Returns:
            Estimated volume in milliliters (ml)
        """
        # Extract food depth values
        food_depths = depth_map[food_mask == 255]

        if len(food_depths) == 0:
            logger.warning("No food region found for volume calculation")
            return 0.0

        # Normalize depth values (inverse depth from MiDaS)
        # Higher values = closer to camera = higher elevation
        depth_normalized = (food_depths - food_depths.min()) / (food_depths.max() - food_depths.min() + 1e-8)

        # Calculate pixel area in cm²
        pixel_area_cm2 = (pixel_to_cm ** 2)

        # Use FOOD-SPECIFIC height instead of hardcoded 5cm!
        if food_type:
            max_height_cm, shape_type = get_food_height(food_type, portion_size="typical")
            logger.info(f"Using food-specific height for '{food_type}': {max_height_cm}cm ({shape_type})")
        else:
            # Fallback to conservative default
            max_height_cm = 5.0
            logger.warning("No food type provided, using default height: 5cm")

        heights_cm = depth_normalized * max_height_cm

        # Calculate volume: sum of (pixel_area × height) for all pixels
        volume_cm3 = np.sum(heights_cm) * pixel_area_cm2

        # Convert cm³ to ml (1 cm³ = 1 ml)
        volume_ml = volume_cm3

        # SANITY CHECK: Cap volume at reasonable maximum
        # Typical meal portions: 200-800ml
        # Large servings: 800-1500ml
        # Anything over 1500ml is likely an error (segmentation failure, scale miscalibration)
        MAX_REASONABLE_VOLUME = 1500.0  # ml
        if volume_ml > MAX_REASONABLE_VOLUME:
            logger.warning(
                f"⚠️ Volume {volume_ml:.0f}ml exceeds reasonable maximum ({MAX_REASONABLE_VOLUME}ml). "
                f"Capping to {MAX_REASONABLE_VOLUME}ml. This suggests calibration or segmentation error."
            )
            volume_ml = MAX_REASONABLE_VOLUME

        logger.info(f"Calculated volume: {volume_ml:.2f} ml from {len(food_depths)} pixels (max_height={max_height_cm}cm)")

        return volume_ml
    
    def estimate_portion(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        food_type: Optional[str] = None,
        reference_object: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Complete portion estimation pipeline
        
        Args:
            image: RGB image array
            depth_map: Depth map from MiDaS
            food_type: Type of Nigerian food for density lookup
            reference_object: Reference object for scale calibration
        
        Returns:
            Dictionary with volume, weight, and confidence estimates
        """
        # Step 1: Calibrate scale
        reference_detected = self.calibrate_scale(image, reference_object)

        # Step 2: Detect food region
        food_mask, food_depth = self.detect_food_region(image, depth_map)

        # Step 3: Calculate volume WITH food-specific height
        volume_ml = self.calculate_volume_from_depth(
            depth_map,
            food_mask,
            self.pixel_to_cm_ratio,
            food_type=food_type  # Pass food type for height estimation
        )
        
        # Step 4: Convert to weight using food density
        if food_type:
            weight_grams = estimate_weight_from_volume(volume_ml, food_type)
        else:
            # Use default density if food type not specified
            weight_grams = volume_ml * 0.90  # Default density
        
        # Step 5: Calculate confidence
        # Use calibration confidence as base, then adjust
        if reference_detected:
            confidence = self.calibration_confidence
        else:
            confidence = 0.3  # Low confidence without reference

        # Boost confidence if food region is clear
        if np.sum(food_mask) > (food_mask.size * 0.1):  # Food covers >10% of image
            confidence += 0.1
        if food_type:  # Food type specified (enables height calibration)
            confidence += 0.1

        confidence = min(confidence, 0.85)  # Cap at 85% (we're honest now!)
        
        return {
            "volume_ml": float(volume_ml),
            "weight_grams": float(weight_grams),
            "confidence": float(confidence),
            "reference_detected": reference_detected,
            "food_pixels": int(np.sum(food_mask == 255)),
            "pixel_to_cm_ratio": float(self.pixel_to_cm_ratio)
        }


def estimate_portion_from_depth(
    image: np.ndarray,
    depth_map: np.ndarray,
    food_type: Optional[str] = None,
    reference_object: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function for portion estimation with Nigerian food priors

    Args:
        image: RGB image array
        depth_map: Depth map from MiDaS (already enhanced with refinement)
        food_type: Type of Nigerian food
        reference_object: Reference object for scale

    Returns:
        Portion estimation results
    """
    calculator = PortionCalculator()

    # Apply food-specific shape priors if food type provided
    enhanced_depth = depth_map
    if food_type:
        logger.info(f"Applying shape priors for food type: {food_type}")
        # Detect food region first
        food_mask, _ = calculator.detect_food_region(image, depth_map)

        # Apply Nigerian food shape constraints
        prior_engine = NigerianFoodPriors()
        enhanced_depth = prior_engine.apply_shape_prior(
            depth_map,
            food_mask,
            food_type
        )

    # Calculate portion with enhanced depth map
    return calculator.estimate_portion(image, enhanced_depth, food_type, reference_object)
