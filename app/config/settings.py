from pydantic import SecretStr
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Config for environmental variables and system settings"""
    GEMINI_API_KEY: SecretStr
    GEMINI_MODEL_ID: Optional[str] = None
    
    
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"