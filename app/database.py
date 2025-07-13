# /app/database.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from app.config import settings

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

db_manager = MongoDB()

async def connect_to_mongo():
    print("Connecting to MongoDB...")
    db_manager.client = AsyncIOMotorClient(settings.MONGODB_URI)
    db_manager.db = db_manager.client[settings.MONGO_DATABASE_NAME]
    print("Successfully connected to MongoDB.")

def close_mongo_connection():
    print("Closing MongoDB connection...")
    if db_manager.client:
        db_manager.client.close()
    print("MongoDB connection closed.")

def get_database() -> AsyncIOMotorDatabase:
    """
    Returns the database instance.
    """
    if db_manager.db is None:
        # This is a fallback for cases where the dependency injection isn't used
        # or for scripts that might run outside the FastAPI app context.
        # Note: This does not establish a new connection pool on its own.
        # The main connection should be managed by the app's lifespan events.
        raise RuntimeError("Database connection not established. Ensure connect_to_mongo() has been awaited.")
    return db_manager.db 