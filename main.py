import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.database.session import engine
from src.database.base import Base
from src.routes.index import api_router
from src.utils.logger import setup_application_logging, get_logger

# Setup centralized logging (console only)
setup_application_logging(log_level="INFO")
logger = get_logger(__name__)

# Create Database Tables
logger.info("🗄️ Creating database tables...")
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Vidhi AI - Production API",
    description="Nepali Legal AI Assistant with RAG-based question answering",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 1. Middleware for Localhost and Production
# Allows your frontend (running on different ports/origins) to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "http://localhost:3000",  # React/Next.js default
        # Add your production domain here later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS middleware configured")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Vidhi AI API is starting up...")
    logger.info("📚 API Documentation available at: http://localhost:8000/api/docs")
    
    # Initialize LLM service (lazy loading on first request)
    try:
        from src.llm.dependencies import get_llm_service
        logger.info("🔧 LLM Service will initialize on first request...")
    except Exception as e:
        logger.warning(f"⚠️ LLM Service may not be available: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Vidhi AI API is shutting down...")

# Health check endpoint
@app.get("/health")
def health_check():
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "service": "Vidhi AI",
        "version": "1.0.0"
    }

# 2. Subscribe to the Master Router (contains /users and /chat)
app.include_router(api_router, prefix="/api")
logger.info("✅ API routes registered")

# 3. Serve Frontend Files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
logger.info("✅ Frontend files mounted")

if __name__ == "__main__":
    logger.info("🌐 Starting Uvicorn server on http://localhost:8000")
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)