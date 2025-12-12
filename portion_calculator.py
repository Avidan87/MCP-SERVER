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
            logger.warning("No food region detected, using full image")
            return np.ones(image.shape[:2], dtype=np.uint8) * 255, depth_map
        
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
        Calibrate pixel-to-cm ratio using reference object
        
        Args:
            image: RGB image array
            reference_object: Type of reference object (e.g., "hand", "plate")
        
        Returns:
            True if calibration successful
        """
        if reference_object and reference_object.lower() in self.REFERENCE_SIZES:
            # TODO: Implement actual reference object detection
            # For now, using estimated pixel size based on image dimensions
            # This should be replaced with actual object detection
            
            image_width = image.shape[1]
            reference_size_cm = self.REFERENCE_SIZES[reference_object.lower()]
            
            # Rough estimation: assume reference object takes ~30% of image width
            estimated_pixels = image_width * 0.3
            self.pixel_to_cm_ratio = reference_size_cm / estimated_pixels
            self.reference_detected = True
            
            logger.info(f"Scale calibrated using {reference_object}: {self.pixel_to_cm_ratio:.4f} cm/pixel")
            return True
        else:
            # Default calibration based on typical food photography
            # Assume image represents ~40cm width (typical plate + margins)
            self.pixel_to_cm_ratio = 40.0 / image.shape[1]
            self.reference_detected = False
            
            logger.info(f"Using default scale calibration: {self.pixel_to_cm_ratio:.4f} cm/pixel")
            return False
    
    def calculate_volume_from_depth(
        self,
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        pixel_to_cm: float
    ) -> float:
        """
        Calculate volume from depth map using numerical integration
        
        Args:
            depth_map: Depth map array
            food_mask: Binary mask of food region
            pixel_to_cm: Pixel to centimeter conversion ratio
        
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
        
        # Estimate height of each pixel in cm
        # Assume max height is ~5cm for typical Nigerian food portions
        max_height_cm = 5.0
        heights_cm = depth_normalized * max_height_cm
        
        # Calculate volume: sum of (pixel_area × height) for all pixels
        volume_cm3 = np.sum(heights_cm) * pixel_area_cm2
        
        # Convert cm³ to ml (1 cm³ = 1 ml)
        volume_ml = volume_cm3
        
        logger.info(f"Calculated volume: {volume_ml:.2f} ml from {len(food_depths)} pixels")
        
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
        
        # Step 3: Calculate volume
        volume_ml = self.calculate_volume_from_depth(
            depth_map,
            food_mask,
            self.pixel_to_cm_ratio
        )
        
        # Step 4: Convert to weight using food density
        if food_type:
            weight_grams = estimate_weight_from_volume(volume_ml, food_type)
        else:
            # Use default density if food type not specified
            weight_grams = volume_ml * 0.90  # Default density
        
        # Step 5: Calculate confidence
        # Higher confidence if reference object detected and food region is clear
        confidence = 0.5  # Base confidence
        if reference_detected:
            confidence += 0.2
        if np.sum(food_mask) > (food_mask.size * 0.1):  # Food covers >10% of image
            confidence += 0.2
        if food_type:  # Food type specified
            confidence += 0.1
        
        confidence = min(confidence, 0.95)  # Cap at 95%
        
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
