"""
Prompt-Optimizer Agent that proposes prompt tweaks based on prior evaluations.
"""
import json
from typing import List, Dict, Any

import google.generativeai as genai

import config
from utils.cache_utils import PromptCache
from utils.data_utils import create_openai_client

class PromptOptimizerAgent:
    """Agent that optimizes prompts for recommendations."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the prompt optimizer agent.
        
        Args:
            model_name: Name of the model to use (defaults to config setting)
        """
        self.model_name = model_name or config.OPTIMIZER_MODEL
        self.prompt_cache = PromptCache()
        self.iteration = 0
    
    def optimize(self, current_prompt: str, evaluation_results: List[Dict[str, Any]], 
                stories: List[Dict[str, Any]], model_name: str = None, max_retries: int = 3) -> str:
        """
        Optimize the recommendation prompt based on evaluation results.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results (score and details)
            stories: List of all available stories
            model_name: Optional model name to use for this specific task (overrides agent's model)
            max_retries: Maximum number of retries if the response is invalid
            
        Returns:
            Optimized prompt
        """
        self.iteration += 1
        
        # If this is the first iteration and we have a cached best prompt, use it
        if self.iteration == 1:
            best_cached = self.prompt_cache.get_best_prompt()
            if best_cached:
                print(f"Using cached best prompt with score {best_cached['score']}")
                return best_cached['prompt']
        
        # Prepare the optimization prompt
        prompt = self._create_optimization_prompt(current_prompt, evaluation_results, stories)
        
        # Use the provided model name or fall back to the agent's model
        model_to_use = model_name or self.model_name
        
        # Check model type and use appropriate API
        model_type = config.get_model_type(model_to_use)
        
        for attempt in range(max_retries):
            try:
                if model_type == config.MODEL_TYPE_GOOGLE:
                    model = genai.GenerativeModel(model_to_use)
                    response = model.generate_content(prompt)
                    optimized_prompt = response.text.strip()
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
                            temperature=0.7,
                            max_tokens=1000
                        )
                        optimized_prompt = response.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"OpenAI API error: {str(e)}")
                        raise
                else:
                    raise ValueError(f"Unknown model type for model: {model_to_use}")
                
                # Basic validation - ensure the optimized prompt contains key components
                if len(optimized_prompt) > 100 and "stories" in optimized_prompt and "recommend" in optimized_prompt:
                    # Add the current score to prompt cache
                    avg_score = sum(r["score"] for r in evaluation_results) / len(evaluation_results)
                    self.prompt_cache.add_prompt(
                        optimized_prompt, 
                        avg_score, 
                        {"iteration": self.iteration}
                    )
                    return optimized_prompt
                
            except Exception as e:
                print(f"Error optimizing prompt (attempt {attempt+1}): {str(e)}")
        
        # If all retries fail, return the current prompt with minor modifications
        print("Failed to generate valid optimized prompt, returning modified current prompt")
        return self._fallback_optimization(current_prompt, evaluation_results)
    
    def _create_optimization_prompt(self, current_prompt: str, evaluation_results: List[Dict[str, Any]], 
                                   stories: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for optimizing the recommendation prompt.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            stories: List of all available stories
            
        Returns:
            Optimization prompt
        """
        # Calculate average score
        avg_score = sum(r["score"] for r in evaluation_results) / len(evaluation_results)
        
        # Extract key information from evaluation results
        successful_cases = []
        unsuccessful_cases = []
        
        for result in evaluation_results:
            if result["score"] >= 0.5:  # Consider cases with score >= 0.5 as successful
                successful_cases.append({
                    "tags": result["extracted_tags"],
                    "ground_truth": result["ground_truth_ids"],
                    "recommendations": result["recommended_ids"],
                    "score": result["score"]
                })
            else:
                unsuccessful_cases.append({
                    "tags": result["extracted_tags"],
                    "ground_truth": result["ground_truth_ids"],
                    "recommendations": result["recommended_ids"],
                    "score": result["score"]
                })
        
        # Create the optimization prompt
        return f"""
        You are a prompt optimization expert. Your task is to improve a prompt for a story recommendation system.
        
        Current prompt:
        ```
        {current_prompt}
        ```
        
        Current average evaluation score: {avg_score:.4f} using {config.METRIC} metric.
        
        Successful recommendation cases:
        {json.dumps(successful_cases, indent=2)}
        
        Unsuccessful recommendation cases:
        {json.dumps(unsuccessful_cases, indent=2)}
        
        Example stories:
        {json.dumps(stories[:3], indent=2)}
        
        Please create an improved version of the recommendation prompt that will:
        1. Better match user preferences to relevant story attributes
        2. Improve the precision of recommendations
        3. Handle diverse user preferences more effectively
        4. Be concise yet comprehensive
        
        Return ONLY the improved prompt without any explanation or additional text.
        """
    
    def _fallback_optimization(self, current_prompt: str, evaluation_results: List[Dict[str, Any]]) -> str:
        """
        Fallback optimization method when LLM optimization fails.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            
        Returns:
            Modified prompt
        """
        # Extract frequently occurring tags in ground truth recommendations
        all_ground_truth_ids = []
        all_tags = set()
        
        for result in evaluation_results:
            all_ground_truth_ids.extend(result["ground_truth_ids"])
            all_tags.update(result["extracted_tags"])
        
        # Add emphasis on common tags
        emphasis = ", ".join(list(all_tags)[:10])
        
        # Modify the current prompt
        modified_prompt = current_prompt + f"\n\nPay special attention to these important themes and preferences: {emphasis}."
        
        return modified_prompt 