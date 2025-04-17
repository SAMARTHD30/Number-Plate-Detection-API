from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
# import redis.asyncio as redis
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
from dotenv import load_dotenv
import os
import logging
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.model import get_model
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Use settings.MAX_FILE_SIZE from config for consistency
# MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB in bytes

class LimitUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            if "content-length" in request.headers:
                content_length = int(request.headers["content-length"])
                if content_length > settings.MAX_FILE_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"File too large. Maximum size allowed is {settings.MAX_FILE_SIZE/1024/1024}MB"
                        },
                    )
        return await call_next(request)

app = FastAPI(
    title="License Plate Detection API",
    description="API for detecting and processing license plates in images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add file size limit middleware
app.add_middleware(LimitUploadSize)

@app.on_event("startup")
async def startup():
    """Load model on startup to avoid cold start latency"""
    logger.info("Initializing license plate detection model...")
    model = get_model()
    if model is not None:
        logger.info("✅ Model loaded successfully")
    else:
        logger.error("❌ Failed to load model - check logs for details")

# Initialize rate limiter
# @app.on_event("startup")
# async def startup():
#     redis_client = redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
#     await FastAPILimiter.init(redis_client)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "License Plate Detection API is running"}