"""
Utility functions for caching embeddings and prompts.
"""
import os
import json
import hashlib
from typing import Dict, List, Any, Optional

import config

# Ensure cache directory exists
os.makedirs(config.CACHE_DIR, exist_ok=True)

class EmbeddingCache:
    """Cache for story embeddings to avoid redundant computation."""
    
    def __init__(self, cache_file: str = "story_embeddings.json"):
        """
        Initialize the embedding cache.
        
        Args:
            cache_file: Name of the cache file
        """
        self.cache_path = os.path.join(config.CACHE_DIR, cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load the cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f)
        except IOError as e:
            print(f"Error saving embedding cache: {str(e)}")
    
    def get(self, story_id: str) -> Optional[List[float]]:
        """
        Get embedding for a story from the cache.
        
        Args:
            story_id: ID of the story
            
        Returns:
            Embedding vector if cached, None otherwise
        """
        return self.cache.get(story_id)
    
    def put(self, story_id: str, embedding: List[float]):
        """
        Store embedding for a story in the cache.
        
        Args:
            story_id: ID of the story
            embedding: Embedding vector
        """
        self.cache[story_id] = embedding
        self._save_cache()
    
    def contains(self, story_id: str) -> bool:
        """
        Check if embedding for a story is in the cache.
        
        Args:
            story_id: ID of the story
            
        Returns:
            True if cached, False otherwise
        """
        return story_id in self.cache


class PromptCache:
    """Cache for successful prompts and their performance metrics."""
    
    def __init__(self, cache_file: str = "prompt_cache.json"):
        """
        Initialize the prompt cache.
        
        Args:
            cache_file: Name of the cache file
        """
        self.cache_path = os.path.join(config.CACHE_DIR, cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load the cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f)
        except IOError as e:
            print(f"Error saving prompt cache: {str(e)}")
    
    def _compute_prompt_hash(self, prompt: str) -> str:
        """
        Compute a hash for a prompt for efficient lookup.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Hash string
        """
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def get_best_prompt(self) -> Optional[Dict[str, Any]]:
        """
        Get the best performing prompt from the cache.
        
        Returns:
            Dict with prompt and score, or None if cache is empty
        """
        if not self.cache:
            return None
        
        # Sort prompts by score in descending order
        sorted_prompts = sorted(self.cache.values(), key=lambda x: x['score'], reverse=True)
        return sorted_prompts[0] if sorted_prompts else None
    
    def add_prompt(self, prompt: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a prompt and its performance metrics to the cache.
        
        Args:
            prompt: The prompt text
            score: Evaluation score
            metadata: Additional metadata (e.g., iteration number)
        """
        prompt_hash = self._compute_prompt_hash(prompt)
        self.cache[prompt_hash] = {
            'prompt': prompt,
            'score': score,
            'metadata': metadata or {}
        }
        self._save_cache()
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """
        Get all prompts from the cache.
        
        Returns:
            List of prompt dictionaries
        """
        return list(self.cache.values()) 