# /app/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

# --- NEW: Disable Telemetry ---
# This prevents the 'resource' module from being imported on Windows,
# which is not available and causes a crash. This should be the first
# thing that runs in your application.
os.environ["DISABLE_TELEMETRY"] = "1"


# Build an absolute path to the .env file in the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_PROJECT_ROOT, '.env')

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    # Renamed to match the variable in the .env file
    MONGODB_URI: str
    MONGO_DATABASE_NAME: str
    # Renamed to match the variable in the .env file
    GOOGLE_CALENDAR_CREDENTIALS_PATH: str
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./user_chroma_db"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None
    CHROMA_COLLECTION_PREFIX: str = "okada_user_"

    class Config:
        env_file = _ENV_FILE
        extra = 'ignore' # This will ignore extra variables like 'openai_api_key'

settings = Settings()  # type: ignore
