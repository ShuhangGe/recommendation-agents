"""
Configuration settings for the Sekai Recommendation Agent system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Agent Settings
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.8"))
IMPROVEMENT_THRESHOLD = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.01"))  # 1% improvement threshold

# Model Types
MODEL_TYPE_GOOGLE = "google"
MODEL_TYPE_OPENAI = "openai"

# Available Models
AVAILABLE_MODELS = {
    # Google models
    "gemini-1.5-flash": {"type": MODEL_TYPE_GOOGLE, "description": "Fast, efficient model for recommendations"},
    "gemini-1.5-pro": {"type": MODEL_TYPE_GOOGLE, "description": "Powerful model for evaluation and optimization"},
    
    # OpenAI models
    "gpt-3.5-turbo": {"type": MODEL_TYPE_OPENAI, "description": "Balanced performance and speed"},
    "gpt-4": {"type": MODEL_TYPE_OPENAI, "description": "Most powerful model, best for complex tasks"},
    "gpt-4-turbo": {"type": MODEL_TYPE_OPENAI, "description": "Faster version of GPT-4 with good performance"}
}

# Default Model Settings
DEFAULT_RECOMMENDATION_MODEL = "gpt-3.5-turbo"
DEFAULT_EVALUATION_MODEL = "gpt-3.5-turbo"
DEFAULT_OPTIMIZER_MODEL = "gpt-3.5-turbo"
DEFAULT_STORY_GENERATION_MODEL = "gpt-3.5-turbo"
DEFAULT_USER_GENERATION_MODEL = "gpt-3.5-turbo"

# Model Settings (can be overridden by environment variables)
RECOMMENDATION_MODEL = os.getenv("RECOMMENDATION_MODEL", DEFAULT_RECOMMENDATION_MODEL)
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", DEFAULT_EVALUATION_MODEL)
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", DEFAULT_OPTIMIZER_MODEL)
STORY_GENERATION_MODEL = os.getenv("STORY_GENERATION_MODEL", DEFAULT_STORY_GENERATION_MODEL)
USER_GENERATION_MODEL = os.getenv("USER_GENERATION_MODEL", DEFAULT_USER_GENERATION_MODEL)

# File Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
USER_PROFILES_FILE = "user_profiles.json"
STORY_DATA_FILE = "story_data.json"
RESULTS_FILE = "optimization_results.csv"

# Cache Settings
ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "True").lower() in ("true", "1", "yes")
ENABLE_PROMPT_CACHE = os.getenv("ENABLE_PROMPT_CACHE", "True").lower() in ("true", "1", "yes")
CACHE_DIR = ".cache"

# Evaluation Metric
METRIC = os.getenv("METRIC", "precision@10")  # Options: precision@10, recall, semantic_overlap

def get_model_type(model_name):
    """Get the type of a model by name."""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]["type"]
    # Default to OpenAI for unknown models
    return MODEL_TYPE_OPENAI

def list_available_models():
    """Return a formatted string of available models."""
    result = "Available models:\n"
    
    result += "\nGoogle Models:\n"
    for name, info in AVAILABLE_MODELS.items():
        if info["type"] == MODEL_TYPE_GOOGLE:
            result += f"  - {name}: {info['description']}\n"
    
    result += "\nOpenAI Models:\n"
    for name, info in AVAILABLE_MODELS.items():
        if info["type"] == MODEL_TYPE_OPENAI:
            result += f"  - {name}: {info['description']}\n"
            
    return result 