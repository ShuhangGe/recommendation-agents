"""
Evaluation Agent that evaluates recommendation quality by simulating
user preferences and comparing with ground truth recommendations.
"""
import json
from typing import List, Dict, Any, Tuple

import google.generativeai as genai

import config
from utils.metrics import calculate_metric
from utils.data_utils import create_openai_client

class EvaluationAgent:
    """Agent that evaluates recommendation quality."""
    
    def __init__(self, stories: List[Dict[str, Any]], model_name: str = None):
        """
        Initialize the evaluation agent.
        
        Args:
            stories: List of all available stories
            model_name: Name of the model to use (defaults to config setting)
        """
        self.stories = stories
        self.model_name = model_name or config.EVALUATION_MODEL
        self.story_dict = {s["id"]: s for s in stories}
    
    def extract_tags(self, user_profile: str, model_name: str = None, max_retries: int = 3) -> List[str]:
        """
        Extract tags that a user would select based on their profile.
        
        Args:
            user_profile: Full user profile text
            model_name: Optional model name to use for this specific task (overrides agent's model)
            max_retries: Maximum number of retries if the response is invalid
            
        Returns:
            List of tags
        """
        prompt = f"""
        You are an evaluation agent for Sekai, a platform for interactive stories.
        
        Given a user profile, extract the tags that this user would likely select on Sekai's first screen.
        Focus on extracting key preferences, interests, and themes from the profile.
        
        User profile:
        {user_profile}
        
        Return ONLY a valid JSON list of tags/keywords that represent the user's preferences, like: 
        ["action", "romance", "underdog", "rivalry", "naruto"]
        
        Extract 10-15 most important tags. Do not include any explanation or additional text.
        """
        
        # Use the provided model name or fall back to the agent's model
        model_to_use = model_name or self.model_name
        
        # Check model type and use appropriate API
        model_type = config.get_model_type(model_to_use)
        
        for attempt in range(max_retries):
            try:
                if model_type == config.MODEL_TYPE_GOOGLE:
                    model = genai.GenerativeModel(model_to_use)
                    response = model.generate_content(prompt)
                    raw_response = response.text.strip()
                elif model_type == config.MODEL_TYPE_OPENAI:
                    import openai
                    messages = [{"role": "user", "content": prompt}]
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
                    tags = json.loads(json_str)
                    return tags
                
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                print(f"Error parsing tag extraction response (attempt {attempt+1}): {str(e)}")
        
        # If all retries fail, extract words from the profile as tags
        print("Failed to extract tags properly, using fallback method")
        words = user_profile.lower().split()
        # Simple filtering for meaningful words
        tags = [w for w in words if len(w) > 4 and w not in 
                ["would", "could", "should", "their", "there", "these", "those", "about", "after", "again", "among"]]
        return list(set(tags))[:15]  # Deduplicate and limit to 15 tags
    
    def generate_ground_truth(self, user_profile: str, model_name: str = None, max_retries: int = 3) -> List[str]:
        """
        Generate ground truth recommendations based on the full user profile.
        
        Args:
            user_profile: Full user profile text
            model_name: Optional model name to use for this specific task (overrides agent's model)
            max_retries: Maximum number of retries if the response is invalid
            
        Returns:
            List of recommended story IDs
        """
        prompt = f"""
        You are an evaluation agent for Sekai, a platform for interactive stories.
        
        Given a user profile and a list of available stories, recommend the 10 most relevant stories for this user.
        
        User profile:
        {user_profile}
        
        Available stories:
        {json.dumps(self.stories, indent=2)}
        
        Return ONLY a valid JSON list of the 10 most relevant story IDs, like:
        ["123456", "234567", "345678", ...]
        
        Do not include any explanation or additional text in your response.
        """
        
        # Use the provided model name or fall back to the agent's model
        model_to_use = model_name or self.model_name
        
        # Check model type and use appropriate API
        model_type = config.get_model_type(model_to_use)
        
        for attempt in range(max_retries):
            try:
                if model_type == config.MODEL_TYPE_GOOGLE:
                    model = genai.GenerativeModel(model_to_use)
                    response = model.generate_content(prompt)
                    raw_response = response.text.strip()
                elif model_type == config.MODEL_TYPE_OPENAI:
                    import openai
                    messages = [{"role": "user", "content": prompt}]
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
                    story_ids = json.loads(json_str)
                    
                    # Ensure we have exactly 10 recommendations
                    if len(story_ids) > 10:
                        story_ids = story_ids[:10]
                    elif len(story_ids) < 10:
                        # If we have fewer than 10, add more
                        all_ids = [s["id"] for s in self.stories]
                        remaining_ids = [sid for sid in all_ids if sid not in story_ids]
                        story_ids.extend(remaining_ids[:10-len(story_ids)])
                    
                    return story_ids
                
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                print(f"Error parsing ground truth response (attempt {attempt+1}): {str(e)}")
        
        # If all retries fail, return 10 random story IDs
        print("Failed to generate valid ground truth, returning random stories")
        return [s["id"] for s in self.stories[:10]]
    
    def evaluate(self, recommendation_agent, user_profile: str, tag_model: str = None, ground_truth_model: str = None, recommendation_model: str = None, metric: str = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate recommendation quality for a user.
        
        Args:
            recommendation_agent: Instance of the RecommendationAgent
            user_profile: Full user profile text
            tag_model: Optional model name to use for tag extraction
            ground_truth_model: Optional model name to use for ground truth generation
            recommendation_model: Optional model name to use for recommendations
            metric: Evaluation metric to use (defaults to config.METRIC)
            
        Returns:
            Tuple of (score, evaluation_details)
        """
        # Step 1: Extract tags from user profile
        tags = self.extract_tags(user_profile, model_name=tag_model)
        print(f"Extracted tags: {tags}")
        
        # Step 2: Generate ground truth recommendations
        ground_truth_ids = self.generate_ground_truth(user_profile, model_name=ground_truth_model)
        ground_truth_stories = [self.story_dict[sid] for sid in ground_truth_ids if sid in self.story_dict]
        print(f"Ground truth stories: {[s['title'] for s in ground_truth_stories]}")
        
        # Step 3: Get recommendations from the recommendation agent
        recommended_ids = recommendation_agent.recommend(tags, model_name=recommendation_model)
        recommended_stories = [self.story_dict[sid] for sid in recommended_ids if sid in self.story_dict]
        print(f"Recommended stories: {[s['title'] for s in recommended_stories]}")
        
        # Step 4: Calculate evaluation metric
        metric_to_use = metric or config.METRIC
        score = calculate_metric(recommended_stories, ground_truth_stories, metric_to_use)
        
        # Prepare evaluation details
        evaluation_details = {
            "extracted_tags": tags,
            "ground_truth_ids": ground_truth_ids,
            "recommended_ids": recommended_ids,
            "metric": metric_to_use,
            "score": score
        }
        
        return score, evaluation_details 