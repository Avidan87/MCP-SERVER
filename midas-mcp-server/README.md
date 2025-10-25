# MiDaS MCP Server

Monocular depth estimation server for KAI Portion Agent using Intel's MiDaS model.

## Features

- **Depth Estimation**: Estimate depth maps from food images
- **Portion Calculation**: Calculate portion sizes (weight/volume) using depth information
- **Nigerian Food Database**: Integrated density values for Nigerian foods
- **FastAPI**: High-performance REST API
- **Railway Ready**: Optimized for Railway deployment

## API Endpoints

### Health Check
```
GET /health
```
Returns server health status and model loading state.

### Root
```
GET /
```
Returns service information and available endpoints.

### Estimate Depth
```
POST /estimate_depth
```
Upload an image to get depth map statistics.

**Request:**
- `file`: Image file (JPEG, PNG)

**Response:**
```json
{
  "depth_map_shape": [height, width],
  "min_depth": 0.0,
  "max_depth": 1.0,
  "mean_depth": 0.5,
  "success": true,
  "message": "Depth estimation completed successfully"
}
```

### Estimate Portion
```
POST /estimate_portion
```
Upload an image to estimate portion size (weight and volume).

**Request:**
- `file`: Image file (JPEG, PNG)
- `food_type` (optional): Nigerian food type (e.g., "jollof_rice", "egusi_soup")
- `reference_object` (optional): Reference object in image (e.g., "hand", "plate")

**Response:**
```json
{
  "estimated_weight_grams": 250.0,
  "estimated_volume_ml": 280.0,
  "confidence": 0.75,
  "reference_object_detected": true,
  "success": true,
  "message": "Portion estimation completed successfully..."
}
```

## Deployment on Railway

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to https://railway.app/
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository
   - Railway will auto-detect the Dockerfile and deploy

3. **Get Your URL**
   - After deployment, Railway provides a public URL
   - Test with: `https://<your-app>.railway.app/health`

## Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements-midas.txt
   ```

2. **Run the server**
   ```bash
   python start_server.py
   ```

3. **Test the API**
   ```bash
   curl http://localhost:8000/health
   ```

## Environment Variables

- `PORT`: Server port (default: 8000, Railway sets this automatically)

## Model Information

- **Model**: Intel MiDaS DPT_Hybrid
- **Purpose**: Monocular depth estimation
- **Size**: ~400MB
- **Performance**: Optimized for deployment with single worker

## Tech Stack

- **FastAPI**: Web framework
- **PyTorch**: Deep learning framework
- **MiDaS**: Depth estimation model
- **OpenCV**: Image processing
- **Uvicorn**: ASGI server

## License

MIT
