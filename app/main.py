from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from dotenv import load_dotenv
import os
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables
load_dotenv()

# Configure maximum upload size (100MB)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB in bytes

class LimitUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            if "content-length" in request.headers:
                content_length = int(request.headers["content-length"])
                if content_length > MAX_UPLOAD_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"File too large. Maximum size allowed is {MAX_UPLOAD_SIZE/1024/1024}MB"
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

# Initialize rate limiter
@app.on_event("startup")
async def startup():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        password=os.getenv("REDIS_PASSWORD", None)
    )
    await FastAPILimiter.init(redis_client)

# Include API routes
app.include_router(
    api_router,
    dependencies=[Depends(RateLimiter(times=10, minutes=1))]  # 10 requests per minute
)

@app.get("/")
async def root():
    return {"message": "Welcome to License Plate Detection API"}