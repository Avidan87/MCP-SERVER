"""
Nigerian Food Height Database
Measured typical heights for Nigerian foods to improve volume estimation accuracy

Heights are based on:
- Traditional serving styles
- Typical portion presentations
- Nigerian culinary practices
"""

from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Food height database (in centimeters)
# Format: food_name -> (min_height_cm, typical_height_cm, max_height_cm, shape_type)

NIGERIAN_FOOD_HEIGHTS: Dict[str, Tuple[float, float, float, str]] = {
    # STAPLES - Swallows (Mounded shapes)
    "fufu": (6.0, 9.0, 12.0, "mound"),
    "pounded_yam": (6.0, 10.0, 14.0, "mound"),
    "eba": (4.0, 7.0, 10.0, "mound"),
    "garri": (4.0, 7.0, 10.0, "mound"),
    "amala": (5.0, 8.0, 11.0, "mound"),
    "semovita": (4.0, 7.0, 10.0, "mound"),
    "tuwo_shinkafa": (5.0, 8.0, 11.0, "mound"),
    "tuwo_masara": (5.0, 8.0, 11.0, "mound"),

    # RICE DISHES (Mounded or layered)
    "jollof_rice": (3.0, 5.0, 7.0, "mound"),
    "fried_rice": (3.0, 5.0, 7.0, "mound"),
    "white_rice": (3.0, 5.0, 7.0, "mound"),
    "coconut_rice": (3.0, 5.0, 7.0, "mound"),
    "ofada_rice": (3.0, 5.0, 7.0, "mound"),
    "rice_and_stew": (3.0, 5.0, 7.0, "layered"),

    # SOUPS (Bowl-contained, liquid)
    "egusi_soup": (2.0, 4.0, 6.0, "bowl"),
    "okra_soup": (2.0, 4.0, 6.0, "bowl"),
    "ogbono_soup": (2.0, 4.0, 6.0, "bowl"),
    "efo_riro": (2.0, 4.0, 6.0, "bowl"),
    "afang_soup": (2.5, 4.5, 6.5, "bowl"),
    "edikang_ikong": (2.5, 4.5, 6.5, "bowl"),
    "banga_soup": (2.0, 4.0, 6.0, "bowl"),
    "oha_soup": (2.0, 4.0, 6.0, "bowl"),
    "pepper_soup": (3.0, 5.0, 7.0, "bowl"),
    "bitterleaf_soup": (2.0, 4.0, 6.0, "bowl"),
    "white_soup": (2.0, 4.0, 6.0, "bowl"),
    "ewedu_soup": (1.5, 3.0, 4.5, "bowl"),
    "gbegiri_soup": (1.5, 3.0, 4.5, "bowl"),

    # STEWS (Thicker, bowl or layered)
    "chicken_stew": (2.0, 4.0, 6.0, "bowl"),
    "fish_stew": (2.0, 4.0, 6.0, "bowl"),
    "beef_stew": (2.0, 4.0, 6.0, "bowl"),
    "ayamase": (2.0, 4.0, 6.0, "bowl"),

    # BEANS DISHES
    "beans_porridge": (3.0, 5.0, 7.0, "bowl"),
    "moi_moi": (4.0, 6.0, 8.0, "mound"),
    "akara": (1.5, 2.5, 3.5, "flat_stack"),
    "ewa_agoyin": (2.0, 4.0, 6.0, "bowl"),

    # YAM DISHES
    "yam_porridge": (3.0, 5.0, 7.0, "bowl"),
    "boiled_yam": (2.0, 4.0, 6.0, "pieces"),
    "fried_yam": (2.0, 4.0, 6.0, "flat_stack"),
    "dundun": (2.0, 4.0, 6.0, "flat_stack"),

    # PLANTAIN DISHES (Flat pieces)
    "fried_plantain": (1.0, 2.0, 3.0, "flat_stack"),
    "dodo": (1.0, 2.0, 3.0, "flat_stack"),
    "boli": (3.0, 5.0, 7.0, "pieces"),

    # PROTEINS (Individual pieces)
    "grilled_chicken": (3.0, 5.0, 8.0, "pieces"),
    "fried_chicken": (3.0, 5.0, 8.0, "pieces"),
    "grilled_fish": (2.0, 4.0, 6.0, "flat_piece"),
    "fried_fish": (2.0, 4.0, 6.0, "flat_piece"),
    "suya": (1.0, 2.0, 3.0, "flat_stack"),
    "ponmo": (0.5, 1.0, 2.0, "flat_stack"),
    "nkwobi": (2.0, 4.0, 6.0, "bowl"),

    # SNACKS
    "puff_puff": (3.0, 4.0, 5.0, "pieces"),
    "chin_chin": (2.0, 3.0, 4.0, "heap"),
    "meat_pie": (3.0, 4.0, 5.0, "pieces"),
    "sausage_roll": (2.0, 3.0, 4.0, "pieces"),

    # OTHERS
    "gizdodo": (2.0, 4.0, 6.0, "mixed"),
    "abacha": (2.0, 3.0, 4.0, "flat"),
    "okpa": (2.0, 3.0, 4.0, "pieces"),
    "masa": (1.0, 2.0, 3.0, "flat_stack"),

    # PASTA
    "spaghetti": (2.0, 4.0, 6.0, "mound"),
    "indomie": (2.0, 4.0, 6.0, "mound"),
}


# Shape-based default heights (fallback)
SHAPE_DEFAULT_HEIGHTS: Dict[str, Tuple[float, float, float]] = {
    "mound": (4.0, 7.0, 10.0),        # Swallows, rice dishes
    "bowl": (2.0, 4.0, 6.0),          # Soups, stews
    "flat_stack": (1.0, 2.5, 4.0),    # Fried plantain, yam
    "pieces": (2.0, 4.0, 6.0),        # Chicken, fish pieces
    "flat_piece": (1.0, 2.0, 3.0),    # Single flat item
    "layered": (3.0, 5.0, 7.0),       # Rice with stew
    "heap": (2.0, 3.0, 4.0),          # Small snacks piled
    "mixed": (2.0, 4.0, 6.0),         # Mixed items
    "flat": (1.0, 2.0, 3.0),          # Flat dishes
}


def get_food_height(
    food_name: str,
    portion_size: str = "typical"
) -> Tuple[float, str]:
    """
    Get estimated height for a Nigerian food

    Args:
        food_name: Name of food (normalized to lowercase)
        portion_size: "small", "typical", or "large"

    Returns:
        Tuple of (height_cm, shape_type)
    """
    food_key = food_name.lower().replace(" ", "_")

    # Try exact match first
    if food_key in NIGERIAN_FOOD_HEIGHTS:
        min_h, typical_h, max_h, shape = NIGERIAN_FOOD_HEIGHTS[food_key]

        if portion_size == "small":
            height = min_h
        elif portion_size == "large":
            height = max_h
        else:  # typical
            height = typical_h

        logger.info(f"Food height for '{food_name}': {height}cm ({shape})")
        return (height, shape)

    # Try partial match (e.g., "jollof" in food_name)
    for key, (min_h, typical_h, max_h, shape) in NIGERIAN_FOOD_HEIGHTS.items():
        if key in food_key or food_key in key:
            if portion_size == "small":
                height = min_h
            elif portion_size == "large":
                height = max_h
            else:
                height = typical_h

            logger.info(f"Food height for '{food_name}' (partial match '{key}'): {height}cm ({shape})")
            return (height, shape)

    # Fallback: Use generic mound shape (most common)
    logger.warning(f"Unknown food '{food_name}', using default mound height")
    min_h, typical_h, max_h = SHAPE_DEFAULT_HEIGHTS["mound"]

    if portion_size == "small":
        height = min_h
    elif portion_size == "large":
        height = max_h
    else:
        height = typical_h

    return (height, "mound")


def get_height_for_shape(shape_type: str) -> float:
    """
    Get typical height for a shape type

    Args:
        shape_type: Shape type (e.g., "mound", "bowl", "flat_stack")

    Returns:
        Typical height in cm
    """
    if shape_type in SHAPE_DEFAULT_HEIGHTS:
        _, typical, _ = SHAPE_DEFAULT_HEIGHTS[shape_type]
        return typical

    # Default fallback
    return 5.0


def estimate_portion_size_category(
    volume_ml: float,
    food_name: str
) -> str:
    """
    Estimate if portion is small/typical/large based on volume

    Args:
        volume_ml: Estimated volume in ml
        food_name: Name of the food

    Returns:
        "small", "typical", or "large"
    """
    food_key = food_name.lower().replace(" ", "_")

    # Get food's typical height to infer volume category
    _, typical_height, _, shape = NIGERIAN_FOOD_HEIGHTS.get(
        food_key,
        (4.0, 7.0, 10.0, "mound")
    )

    # Rough volume thresholds (very approximate)
    # Based on typical Nigerian portions
    if shape == "mound" or shape == "bowl":
        # Swallows and soups
        if volume_ml < 200:
            return "small"
        elif volume_ml > 400:
            return "large"
        else:
            return "typical"
    else:
        # Other foods
        if volume_ml < 150:
            return "small"
        elif volume_ml > 350:
            return "large"
        else:
            return "typical"


# Convenience function
def get_food_height_cm(food_name: str, portion_size: str = "typical") -> float:
    """
    Get food height in cm (returns only the height value)

    Args:
        food_name: Name of the food
        portion_size: "small", "typical", or "large"

    Returns:
        Height in centimeters
    """
    height, _ = get_food_height(food_name, portion_size)
    return height
