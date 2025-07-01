import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Weather API
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeatherMap
    
    # Search API
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  # SerpAPI or similar
    
    # Database
    DB_PATH = "conversations.db"
    
    # Default settings
    DEFAULT_LLM_PROVIDER = "groq"
    DEFAULT_MODEL = "qwen-qwq-32b"

settings = Settings()