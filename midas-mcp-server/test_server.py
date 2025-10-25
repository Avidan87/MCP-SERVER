"""
Test script for MiDaS MCP Server
"""

import requests
import json
from pathlib import Path

# Server URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_depth_estimation(image_path: str):
    """Test depth estimation endpoint"""
    print(f"Testing depth estimation with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found at {image_path}")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/estimate_depth", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


def test_portion_estimation(image_path: str, food_type: str = "jollof-rice"):
    """Test portion estimation endpoint"""
    print(f"Testing portion estimation with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found at {image_path}")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"food_type": food_type, "reference_object": "plate"}
        response = requests.post(f"{BASE_URL}/estimate_portion", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    print("=== MiDaS MCP Server Test Suite ===\n")
    
    # Test health check
    test_health_check()
    
    # Test with sample image (you'll need to provide an actual image path)
    sample_image = "sample_food.jpg"  # Replace with actual image path
    
    print(f"Note: To test depth and portion estimation, provide a valid image path")
    print(f"Current test image: {sample_image}\n")
    
    # Uncomment these when you have a test image
    # test_depth_estimation(sample_image)
    # test_portion_estimation(sample_image, "jollof-rice")
