"""
MiDaS MCP Server for KAI Portion Agent
Provides depth estimation and portion size calculation endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging
from portion_calculator import estimate_portion_from_depth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MiDaS MCP Server",
    description="Monocular depth estimation for portion size calculation",
    version="1.0.0"
)

# CORS middleware for Agent Builder integration and Railway healthchecks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "healthcheck.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to handle Railway healthcheck requests
@app.middleware("http")
async def railway_healthcheck_middleware(request: Request, call_next):
    """Handle Railway healthcheck requests"""
    # Log all requests for debugging
    logger.info(f"Request: {request.method} {request.url} from {request.client.host}")
    
    # Handle Railway healthcheck hostname
    if request.headers.get("host") == "healthcheck.railway.app":
        logger.info("Railway healthcheck request detected")
    
    response = await call_next(request)
    return response

# Global model instance
midas_model = None
midas_transform = None
device = None


class DepthResponse(BaseModel):
    """Response model for depth estimation"""
    depth_map_shape: tuple
    min_depth: float
    max_depth: float
    mean_depth: float
    success: bool
    message: str


class PortionEstimate(BaseModel):
    """Response model for portion estimation"""
    estimated_weight_grams: float
    estimated_volume_ml: float
    confidence: float
    reference_object_detected: bool
    success: bool
    message: str


def load_midas_model():
    """Load MiDaS model on startup"""
    global midas_model, midas_transform, device
    
    try:
        logger.info("Loading MiDaS model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Use a smaller, faster model for Railway deployment
        # DPT_Hybrid is smaller and faster than DPT_Large
        logger.info("Loading DPT_Hybrid model (optimized for deployment)...")
        midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        midas_model.to(device)
        midas_model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = midas_transforms.dpt_transform
        
        logger.info("MiDaS model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {str(e)}")
        logger.error("Server will start but model-dependent endpoints will be unavailable")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup"""
    logger.info("Starting model loading process...")
    success = load_midas_model()
    if not success:
        logger.warning("Server started but MiDaS model failed to load")
        logger.warning("Model-dependent endpoints will return 503 errors")
    else:
        logger.info("Model loading completed successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return JSONResponse(
        status_code=200,
        content={
            "service": "MiDaS MCP Server",
            "status": "running",
            "model_loaded": midas_model is not None,
            "endpoints": ["/health", "/ready", "/estimate_depth", "/estimate_portion"]
        }
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check - only returns healthy when model is loaded"""
    if midas_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    return {
        "status": "ready",
        "model_loaded": True,
        "device": str(device) if device else "not initialized"
    }


@app.post("/estimate_depth", response_model=DepthResponse)
async def estimate_depth(file: UploadFile = File(...)):
    """
    Estimate depth map from uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        DepthResponse with depth map statistics
    """
    if midas_model is None:
        raise HTTPException(status_code=503, detail="MiDaS model not loaded")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Prepare input for MiDaS
        input_batch = midas_transform(img_bgr).to(device)
        
        # Perform depth estimation
        with torch.no_grad():
            prediction = midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        return DepthResponse(
            depth_map_shape=depth_map.shape,
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            mean_depth=float(depth_map.mean()),
            success=True,
            message="Depth estimation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Depth estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")


@app.post("/estimate_portion", response_model=PortionEstimate)
async def estimate_portion(
    file: UploadFile = File(...),
    food_type: Optional[str] = None,
    reference_object: Optional[str] = None
):
    """
    Estimate portion size (weight/volume) from image using depth estimation
    
    Args:
        file: Image file (JPEG, PNG)
        food_type: Optional Nigerian food type for density lookup
        reference_object: Optional reference object in image (e.g., "hand", "plate")
    
    Returns:
        PortionEstimate with weight and volume estimates
    """
    if midas_model is None:
        raise HTTPException(status_code=503, detail="MiDaS model not loaded")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Prepare input for MiDaS
        input_batch = midas_transform(img_bgr).to(device)
        
        # Perform depth estimation
        with torch.no_grad():
            prediction = midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Use the portion calculator to estimate volume and weight
        portion_results = estimate_portion_from_depth(
            image=img_array,
            depth_map=depth_map,
            food_type=food_type,
            reference_object=reference_object
        )
        
        return PortionEstimate(
            estimated_weight_grams=portion_results["weight_grams"],
            estimated_volume_ml=portion_results["volume_ml"],
            confidence=portion_results["confidence"],
            reference_object_detected=portion_results["reference_detected"],
            success=True,
            message=f"Portion estimation completed successfully. Food pixels: {portion_results['food_pixels']}, Scale: {portion_results['pixel_to_cm_ratio']:.4f} cm/pixel"
        )
        
    except Exception as e:
        logger.error(f"Portion estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portion estimation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment"""
    # Always return HTTP 200 for Railway healthcheck
    # Railway just needs to know the service is responding
    # The model can load in the background
    logger.info("Health check requested")
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "MiDaS MCP Server",
            "model_loaded": midas_model is not None,
            "device": str(device) if device else "not initialized",
            "message": "Service is running" + (" (model ready)" if midas_model else " (model loading...)")
        }
    )


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
