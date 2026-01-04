from fastapi import APIRouter
from src.routes import chat, user

api_router = APIRouter()

# Combine all routers here
api_router.include_router(user.router, prefix="/users", tags=["Users"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])