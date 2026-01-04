import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

logger.info(f"🔌 Connecting to database...")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("✅ Database engine initialized")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()