#!/usr/bin/env python3
"""
Startup script for MiDaS MCP Server
Handles model loading and server startup with proper timing
"""

import os
import sys
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    logger.info("Starting MiDaS MCP Server...")
    
    # Get port from environment (Railway requirement)
    port = os.environ.get("PORT", "8000")
    host = "0.0.0.0"
    
    # Validate port is a number
    try:
        port_int = int(port)
        logger.info(f"Using port: {port_int}")
    except ValueError:
        logger.error(f"Invalid PORT value: {port}")
        sys.exit(1)
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Start the FastAPI server with Railway-optimized settings
    cmd = [
        "uvicorn", 
        "server:app",
        "--host", host,
        "--port", port,
        "--workers", "1",
        "--timeout-keep-alive", "30",
        "--access-log",
        "--log-level", "info"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Start the server
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
