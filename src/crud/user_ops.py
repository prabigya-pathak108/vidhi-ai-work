from sqlalchemy.orm import Session
from src.database import models
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_user(db: Session, user_id: str, name: str):
    logger.info(f"Creating user: {user_id} - {name}")
    db_user = models.User(id=user_id, name=name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"✅ User created successfully: {user_id}")
    return db_user

def get_user(db: Session, user_id: str):
    logger.debug(f"Fetching user: {user_id}")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        logger.debug(f"✅ User found: {user_id}")
    else:
        logger.debug(f"❌ User not found: {user_id}")
    return user

def get_all_users(db: Session):
    logger.debug("Fetching all users")
    users = db.query(models.User).all()
    logger.info(f"✅ Retrieved {len(users)} users")
    return users