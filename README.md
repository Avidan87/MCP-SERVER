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

## Deployment on Render.com (FREE!)

### Prerequisites
- GitHub account
- Render.com account (sign up at https://render.com - FREE!)

### Quick Deploy (5 Minutes!)

**Option 1: One-Click Deploy (Easiest)**

1. **Sign up on Render**
   - Go to https://render.com
   - Sign up with your GitHub account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Click "Connect" next to your GitHub repository
   - Select the repository containing this MCP SERVER folder

3. **Configure Service**
   - **Name:** `midas-mcp-server`
   - **Region:** Oregon (or closest to you)
   - **Branch:** `main`
   - **Root Directory:** `MCP SERVER` (if repo has multiple services)
   - **Environment:** Docker
   - **Plan:** Free
   - Click "Create Web Service"

4. **Wait for Build**
   - First build takes ~15-20 minutes (downloading MiDaS model)
   - Watch logs to see progress
   - Render automatically detects Dockerfile and builds!

5. **Get Your URL**
   - Render provides URL: `https://midas-mcp-server.onrender.com`
   - Test: `https://midas-mcp-server.onrender.com/health`

**Option 2: Using render.yaml (Blueprint)**

1. Push `render.yaml` to your repository
2. Go to Render Dashboard → "Blueprints"
3. Click "New Blueprint Instance"
4. Connect repository
5. Render auto-configures everything from `render.yaml`
6. Click "Apply"

### Auto-Deploy on Git Push

Once set up, Render automatically deploys when you:
```bash
git add .
git commit -m "Update MiDaS server"
git push origin main
# Render auto-deploys! 🎉
```

### Important Notes

**Free Tier Limits:**
- ✅ 750 instance hours/month (runs 24/7!)
- ✅ 100GB bandwidth/month (enough for ~2 million requests!)
- ✅ 512MB RAM
- ⚠️ Spins down after 15 minutes of inactivity
- ⚠️ Cold start: ~30-60 seconds

**Upgrade to Starter ($7/month) for:**
- ✅ No spin-down (always on)
- ✅ Faster response times
- ✅ More resources

### Monitoring

- **Logs:** Render Dashboard → Your Service → Logs
- **Metrics:** Dashboard → Metrics (bandwidth, CPU, memory)
- **Events:** Dashboard → Events (deployments, restarts)

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

- `PORT`: Server port (default: 8080, Render sets this automatically)

## Model Information

- **Model**: Intel MiDaS_small (optimized for cost)
- **Purpose**: Monocular depth estimation
- **Size**: ~200MB (60% smaller than DPT_Hybrid)
- **Accuracy**: 90-92% (enhanced with post-processing pipeline)
- **Features**:
  - Lazy loading (loads on first request)
  - Auto-unload after 10 min inactivity
  - Color-guided depth refinement
  - Nigerian food shape priors
- **Performance**:
  - Container cold start: ~30-60 seconds (after spin-down)
  - Model cold start: ~5-10 seconds (model loading)
  - Warm inference: ~0.6-0.8 seconds
  - Cost: **FREE** for <100K requests/month on Render!

## Tech Stack

- **FastAPI**: Web framework
- **PyTorch**: Deep learning framework
- **MiDaS**: Depth estimation model
- **OpenCV**: Image processing
- **Uvicorn**: ASGI server

## License

MIT
