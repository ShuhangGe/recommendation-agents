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
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.8"))
IMPROVEMENT_THRESHOLD = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.005"))  # 0.5% improvement threshold

# Model Types
MODEL_TYPE_GOOGLE = "google"
MODEL_TYPE_OPENAI = "openai"
MODEL_TYPE_EMBEDDING = "embedding"  # For embedding models
MODEL_TYPE_MODERATION = "moderation"  # For content moderation
MODEL_TYPE_IMAGE = "image"  # For image generation models
MODEL_TYPE_AUDIO = "audio"  # For audio transcription/generation models

# Available Models
AVAILABLE_MODELS = {
    # Google models
    "gemini-1.5-flash": {"type": MODEL_TYPE_GOOGLE, "description": "Fast, efficient model for recommendations"},
    "gemini-1.5-pro": {"type": MODEL_TYPE_GOOGLE, "description": "Powerful model for evaluation and optimization"},
    "gemini-1.5-pro-latest": {"type": MODEL_TYPE_GOOGLE, "description": "Latest version of Gemini 1.5 Pro with improved capabilities"},
    "gemini-1.0-pro": {"type": MODEL_TYPE_GOOGLE, "description": "Original Gemini Pro model with good balance of speed and quality"},
    "gemini-1.0-pro-vision": {"type": MODEL_TYPE_GOOGLE, "description": "Vision-capable model for multimodal tasks"},
    
    # OpenAI Chat Models
    "gpt-3.5-turbo": {"type": MODEL_TYPE_OPENAI, "description": "Current standard GPT-3.5 Turbo model"},
    "gpt-3.5-turbo-16k": {"type": MODEL_TYPE_OPENAI, "description": "Extended context version of GPT-3.5 Turbo"},
    "gpt-3.5-turbo-instruct": {"type": MODEL_TYPE_OPENAI, "description": "Instruction-optimized version of GPT-3.5"},
    "gpt-3.5-turbo-0125": {"type": MODEL_TYPE_OPENAI, "description": "Specific version of GPT-3.5 Turbo from January 2024"},
    
    "gpt-4": {"type": MODEL_TYPE_OPENAI, "description": "Standard GPT-4 model with strong reasoning capabilities"},
    "gpt-4-turbo": {"type": MODEL_TYPE_OPENAI, "description": "Faster version of GPT-4 with updated knowledge"},
    "gpt-4o": {"type": MODEL_TYPE_OPENAI, "description": "Latest optimized GPT-4 model with improved speed and capabilities"},
    "gpt-4o-mini": {"type": MODEL_TYPE_OPENAI, "description": "Smaller, more efficient version of GPT-4o"},
    "gpt-4-vision-preview": {"type": MODEL_TYPE_OPENAI, "description": "Vision-capable model for image understanding"},
    "gpt-4-1106-preview": {"type": MODEL_TYPE_OPENAI, "description": "November 2023 preview version of GPT-4"},
    "gpt-4-0125-preview": {"type": MODEL_TYPE_OPENAI, "description": "January 2024 preview version of GPT-4"},
    "gpt-4-32k": {"type": MODEL_TYPE_OPENAI, "description": "Extended context version of GPT-4 (32k tokens)"},
    
    # Anthropic Claude Models (via OpenAI format)
    "claude-3-opus": {"type": MODEL_TYPE_OPENAI, "description": "Most powerful Anthropic Claude model"},
    "claude-3-sonnet": {"type": MODEL_TYPE_OPENAI, "description": "Balanced Claude model with good performance"},
    "claude-3-haiku": {"type": MODEL_TYPE_OPENAI, "description": "Fast Claude model optimized for speed"},

}

# Default Model Settings
DEFAULT_RECOMMENDATION_MODEL = "gpt-4o-mini"
DEFAULT_EVALUATION_MODEL = "gpt-4o-mini"
DEFAULT_OPTIMIZER_MODEL = "gpt-4o-mini"
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
    
    # Try to guess the model type from the name pattern
    if model_name.startswith("text-embedding"):
        return MODEL_TYPE_EMBEDDING
    elif model_name.startswith("text-moderation"):
        return MODEL_TYPE_MODERATION
    elif model_name.startswith(("dall-e", "image")):
        return MODEL_TYPE_IMAGE
    elif model_name.startswith(("whisper", "tts")):
        return MODEL_TYPE_AUDIO
    elif model_name.startswith(("gpt", "claude")):
        return MODEL_TYPE_OPENAI
    elif model_name.startswith("gemini"):
        return MODEL_TYPE_GOOGLE
        
    # Default to OpenAI for unknown models
    return MODEL_TYPE_OPENAI

def list_available_models():
    """Return a formatted string of available models."""
    result = "Available models:\n"
    
    # Group models by type
    model_types = {
        MODEL_TYPE_GOOGLE: "Google Models",
        MODEL_TYPE_OPENAI: "OpenAI Chat Models",
        MODEL_TYPE_EMBEDDING: "Embedding Models",
        MODEL_TYPE_MODERATION: "Moderation Models",
        MODEL_TYPE_IMAGE: "Image Generation Models",
        MODEL_TYPE_AUDIO: "Audio Models"
    }
    
    # Print models grouped by type
    for model_type, type_name in model_types.items():
        models_of_type = {name: info for name, info in AVAILABLE_MODELS.items() 
                         if info["type"] == model_type}
        
        if models_of_type:
            result += f"\n{type_name}:\n"
            for name, info in models_of_type.items():
                result += f"  - {name}: {info['description']}\n"
    
    return result 