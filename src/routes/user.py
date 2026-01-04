from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.session import get_db
from src.crud import user_ops
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/create")
def create_user(user_id: str, name: str, db: Session = Depends(get_db)):
    logger.info(f"📝 User creation request: {user_id} - {name}")
    existing = user_ops.get_user(db, user_id)
    if existing:
        logger.warning(f"⚠️ User already exists: {user_id}")
        raise HTTPException(status_code=400, detail="User already exists")
    
    user = user_ops.create_user(db, user_id, name)
    logger.info(f"✅ User created via API: {user_id}")
    return user

@router.get("/get_all_users")
def get_all_users(db: Session = Depends(get_db)):
    logger.info("📋 Fetching all users via API")
    users = user_ops.get_all_users(db)
    return {"users": users}