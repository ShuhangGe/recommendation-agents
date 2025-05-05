"""
Recommendation Agent that returns story recommendations for a user.
Uses a fast model to quickly generate recommendations based on user preferences.
"""
import json
from typing import List, Dict, Any

import google.generativeai as genai

import config
from utils.cache_utils import PromptCache
from utils.data_utils import create_openai_client

class RecommendationAgent:
    """Agent that recommends stories to users based on their preferences."""
    
    def __init__(self, stories: List[Dict[str, Any]], model_name: str = None):
        """
        Initialize the recommendation agent.
        
        Args:
            stories: List of all available stories
            model_name: Name of the model to use (defaults to config setting)
        """
        self.stories = stories
        self.model_name = model_name or config.RECOMMENDATION_MODEL
        self.prompt_cache = PromptCache()
        self.current_prompt = self._get_initial_prompt()
    
    def _get_initial_prompt(self) -> str:
        """
        Get the initial recommendation prompt.
        
        Returns:
            Initial prompt string
        """
        # Check if there's a best prompt in the cache
        cached_prompt = self.prompt_cache.get_best_prompt()
        if cached_prompt:
            return cached_prompt['prompt']
        
        # Default initial prompt
        return f"""
        You are a smart recommendation agent for Sekai, a platform for interactive stories. 
        Your task is to recommend the most relevant stories for a user based on their preferences.
        
        Here are the available stories:
        {json.dumps(self.stories, indent=2)}
        
        When a user provides their preferences, recommend exactly 10 stories that would be most relevant to them.
        Return ONLY the IDs of the recommended stories as a valid JSON list, like this: ["123456", "234567", ...]
        Do not include any explanation or additional text in your response.
        """
    
    def update_prompt(self, new_prompt: str):
        """
        Update the recommendation prompt.
        
        Args:
            new_prompt: New prompt to use
        """
        self.current_prompt = new_prompt
    
    def recommend(self, user_preferences: List[str], model_name: str = None, max_retries: int = 3) -> List[str]:
        """
        Generate story recommendations for a user.
        
        Args:
            user_preferences: List of user preference tags
            model_name: Optional model name to use for this specific recommendation (overrides agent's model)
            max_retries: Maximum number of retries if the response is invalid
            
        Returns:
            List of recommended story IDs
        """
        # Prepare the complete prompt with user preferences
        user_prefs_str = ', '.join(user_preferences)
        complete_prompt = f"{self.current_prompt}\n\nUser preferences: {user_prefs_str}\n\nRecommended story IDs:"
        
        # Use the provided model name or fall back to the agent's model
        model_to_use = model_name or self.model_name
        
        # Check model type and use appropriate API
        model_type = config.get_model_type(model_to_use)
        
        for attempt in range(max_retries):
            try:
                if model_type == config.MODEL_TYPE_GOOGLE:
                    model = genai.GenerativeModel(model_to_use)
                    response = model.generate_content(complete_prompt)
                    raw_response = response.text.strip()
                elif model_type == config.MODEL_TYPE_OPENAI:
                    import openai
                    messages = [{"role": "user", "content": complete_prompt}]
                    try:
                        # Try to use the client attribute first (set in load_api_keys)
                        if hasattr(openai, 'client') and openai.client:
                            client = openai.client
                        else:
                            # Create a new client
                            client = create_openai_client()
                            
                        response = client.chat.completions.create(
                            model=model_to_use,
                            messages=messages,
                            temperature=0.7
                        )
                        raw_response = response.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"OpenAI API error: {str(e)}")
                        raise
                else:
                    raise ValueError(f"Unknown model type for model: {model_to_use}")
                
                # Extract JSON list from response
                if "[" in raw_response and "]" in raw_response:
                    json_str = raw_response[raw_response.find("["):raw_response.rfind("]")+1]
                    recommended_ids = json.loads(json_str)
                    
                    # Ensure we have exactly 10 recommendations
                    if len(recommended_ids) > 10:
                        recommended_ids = recommended_ids[:10]
                    elif len(recommended_ids) < 10:
                        # If we have fewer than 10, fill with random stories
                        available_ids = [s["id"] for s in self.stories if s["id"] not in recommended_ids]
                        needed = 10 - len(recommended_ids)
                        additional_ids = available_ids[:needed] if needed <= len(available_ids) else available_ids
                        recommended_ids.extend(additional_ids)
                    
                    return recommended_ids
                
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                print(f"Error parsing recommendation response (attempt {attempt+1}): {str(e)}")
        
        # If all retries fail, return 10 random story IDs
        print("Failed to generate valid recommendations, returning random stories")
        return [s["id"] for s in self.stories[:10]]
    
    def get_stories_by_ids(self, story_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get story objects by their IDs.
        
        Args:
            story_ids: List of story IDs
            
        Returns:
            List of story objects
        """
        id_to_story = {s["id"]: s for s in self.stories}
        return [id_to_story[sid] for sid in story_ids if sid in id_to_story] 