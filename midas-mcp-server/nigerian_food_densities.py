"""
Nigerian Food Density Database
Density values (g/ml) for common Nigerian foods to convert volume to weight
"""

NIGERIAN_FOOD_DENSITIES = {
    # Rice dishes
    "jollof-rice": 0.85,
    "fried-rice": 0.80,
    "white-rice": 0.90,
    "ofada-rice": 0.88,
    
    # Starchy sides
    "pounded-yam": 1.10,
    "fufu": 1.05,
    "eba": 1.08,
    "amala": 1.06,
    
    # Soups (liquid-based)
    "egusi-soup": 0.95,
    "efo-riro": 0.92,
    "okra-soup": 0.90,
    "ogbono-soup": 0.93,
    "afang-soup": 0.94,
    "edikang-ikong": 0.95,
    "banga-soup": 0.96,
    "oha-soup": 0.93,
    "pepper-soup": 0.88,
    
    # Beans dishes
    "moi-moi": 1.00,
    "akara": 0.75,
    "ewa-agoyin": 0.95,
    "beans-porridge": 0.92,
    
    # Meat/Protein
    "suya": 0.85,
    "chicken-stew": 0.90,
    "fish-stew": 0.88,
    "nkwobi": 0.95,
    
    # Snacks
    "plantain-fried": 0.70,
    "boli": 0.65,
    "puff-puff": 0.60,
    "chin-chin": 0.55,
    
    # Porridges
    "yam-porridge": 0.88,
    "pap": 0.85,
    "oats": 0.80,
    
    # Default fallback
    "default": 0.90
}


def get_density(food_name: str) -> float:
    """
    Get density value for a Nigerian food item
    
    Args:
        food_name: Name or ID of the food
    
    Returns:
        Density in g/ml
    """
    # Normalize food name
    food_key = food_name.lower().replace(" ", "-").replace("_", "-")
    
    # Try exact match
    if food_key in NIGERIAN_FOOD_DENSITIES:
        return NIGERIAN_FOOD_DENSITIES[food_key]
    
    # Try partial match
    for key, density in NIGERIAN_FOOD_DENSITIES.items():
        if key in food_key or food_key in key:
            return density
    
    # Return default
    return NIGERIAN_FOOD_DENSITIES["default"]


def estimate_weight_from_volume(volume_ml: float, food_name: str) -> float:
    """
    Estimate weight from volume using food-specific density
    
    Args:
        volume_ml: Volume in milliliters
        food_name: Name of the food
    
    Returns:
        Estimated weight in grams
    """
    density = get_density(food_name)
    return volume_ml * density
