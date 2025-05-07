"""
Adaptive Optimizer Agent that dynamically improves recommendation prompts.
This agent analyzes evaluation results and improves prompts for better recommendations.
"""
import json
import random
import time
from typing import List, Dict, Any, Optional

import config
from utils.data_utils import generate_with_model
from utils.cache_utils import PromptCache
from agents.agent_core import Agent, Memory

class AdaptiveOptimizerAgent(Agent):
    """
    Adaptive agent that optimizes recommendation prompts based on evaluation results.
    This agent can:
    1. Analyze evaluation results to identify strengths and weaknesses
    2. Generate improved prompts using various strategies
    3. Track prompt performance over time
    4. Adapt optimization strategies based on context
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the adaptive optimizer agent.
        
        Args:
            model_name: Name of the model to use (defaults to config setting)
        """
        super().__init__(name="AdaptiveOptimizer", model_name=model_name or config.OPTIMIZER_MODEL)
        self.prompt_cache = PromptCache()
        self.iteration = 0
        
        # Register available tools
        self.register_tool("analyze_results", self._analyze_results)
        self.register_tool("optimize_prompt", self._optimize_prompt)
        self.register_tool("fetch_best_prompt", self._fetch_best_prompt)
        
        # Set initial goals
        self.set_goal("Optimize recommendation prompts for better performance")
        self.set_goal("Identify patterns in successful and unsuccessful recommendations")
        self.set_goal("Adapt optimization strategy based on progress patterns")
    
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze evaluation results to understand improvement opportunities.
        
        Args:
            input_data: Dictionary containing current prompt, evaluation results, etc.
            
        Returns:
            Dictionary of perceptions
        """
        self.logger.info("Analyzing evaluation results")
        self.iteration += 1
        
        # Extract input data
        current_prompt = input_data.get("current_prompt", "")
        evaluation_results = input_data.get("evaluation_results", [])
        current_score = input_data.get("current_score", 0.0)
        previous_scores = input_data.get("previous_scores", [])
        available_stories = input_data.get("available_stories", [])
        
        # Analyze the evaluation results
        analysis = self._analyze_results(evaluation_results, current_score, previous_scores)
        
        # Construct perceptions
        perceptions = {
            "current_prompt": current_prompt,
            "current_score": current_score,
            "score_history": previous_scores + [current_score],
            "iteration": self.iteration,
            "analysis": analysis,
            "prompt_length": len(current_prompt),
            "story_count": len(available_stories),
            "score_trends": self._analyze_score_trends(previous_scores + [current_score]),
            "improvement_needed": current_score < config.SCORE_THRESHOLD
        }
        
        # Check for cached best prompt if this is the first iteration
        if self.iteration == 1:
            best_cached = self.prompt_cache.get_best_prompt()
            if best_cached:
                perceptions["best_cached_prompt"] = best_cached.get("prompt")
                perceptions["best_cached_score"] = best_cached.get("score", 0.0)
        
        # Store perceptions in memory
        self.memory.add_to_short_term("latest_perceptions", perceptions)
        
        self.logger.info(f"Perceived score: {current_score:.4f} at iteration {self.iteration}")
        return perceptions
    
    def _analyze_results(self, evaluation_results: List[Dict[str, Any]], 
                       current_score: float, previous_scores: List[float]) -> Dict[str, Any]:
        """
        Analyze evaluation results for insights.
        
        Args:
            evaluation_results: List of evaluation results
            current_score: Current average score
            previous_scores: List of previous scores
            
        Returns:
            Analysis dictionary with insights
        """
        if not evaluation_results:
            return {"status": "no_data"}
            
        # Calculate successful vs unsuccessful cases
        successful_cases = []
        unsuccessful_cases = []
        
        for result in evaluation_results:
            score = result.get("score", 0.0)
            if score >= 0.5:  # Consider cases with score >= 0.5 as successful
                successful_cases.append(result)
            else:
                unsuccessful_cases.append(result)
        
        # Calculate success rate
        success_rate = len(successful_cases) / len(evaluation_results) if evaluation_results else 0
        
        # Check for score plateau
        is_plateauing = False
        if len(previous_scores) >= 3:
            recent_improvements = [previous_scores[i] - previous_scores[i-1] for i in range(1, len(previous_scores))]
            is_plateauing = all(imp < config.IMPROVEMENT_THRESHOLD for imp in recent_improvements[-2:])
        
        # Extract commonly occurring tags in successful and unsuccessful cases
        successful_tags = self._extract_common_tags(successful_cases)
        unsuccessful_tags = self._extract_common_tags(unsuccessful_cases)
        
        # Prepare analysis
        analysis = {
            "successful_count": len(successful_cases),
            "unsuccessful_count": len(unsuccessful_cases),
            "success_rate": success_rate,
            "is_plateauing": is_plateauing,
            "successful_tags": successful_tags,
            "unsuccessful_tags": unsuccessful_tags
        }
        
        return analysis
    
    def _extract_common_tags(self, cases: List[Dict[str, Any]]) -> List[str]:
        """
        Extract commonly occurring tags from cases.
        
        Args:
            cases: List of evaluation result cases
            
        Returns:
            List of common tags
        """
        if not cases:
            return []
            
        # Collect all tags
        all_tags = []
        for case in cases:
            all_tags.extend(case.get("extracted_tags", []))
            
        # Count tag frequencies
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        # Return the most common tags (up to 10)
        return [tag for tag, _ in tag_counts.most_common(10)]
    
    def _analyze_score_trends(self, scores: List[float]) -> Dict[str, Any]:
        """
        Analyze score trends to determine optimization strategy.
        
        Args:
            scores: List of historical scores
            
        Returns:
            Dictionary with trend analysis
        """
        if len(scores) < 2:
            return {"trend": "insufficient_data"}
            
        # Calculate improvements
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        # Calculate average improvement
        avg_improvement = sum(improvements) / len(improvements)
        
        # Determine if improvements are diminishing
        diminishing = False
        if len(improvements) >= 3:
            diminishing = improvements[-1] < improvements[-2] < improvements[-3]
        
        # Determine current trend
        if avg_improvement < 0:
            trend = "negative"
        elif avg_improvement < config.IMPROVEMENT_THRESHOLD / 2:
            trend = "stagnating"
        elif diminishing:
            trend = "diminishing"
        else:
            trend = "improving"
            
        return {
            "trend": trend,
            "avg_improvement": avg_improvement,
            "improvements": improvements,
            "diminishing": diminishing
        }
    
    def reason(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best optimization strategy based on perceptions.
        
        Args:
            perceptions: Dictionary of perceptions about evaluation results
            
        Returns:
            Plan of action for optimizing the prompt
        """
        self.logger.info("Reasoning about optimization strategy")
        
        # Extract key perceptions
        current_prompt = perceptions.get("current_prompt", "")
        current_score = perceptions.get("current_score", 0.0)
        score_history = perceptions.get("score_history", [])
        iteration = perceptions.get("iteration", 0)
        analysis = perceptions.get("analysis", {})
        score_trends = perceptions.get("score_trends", {})
        best_cached_prompt = perceptions.get("best_cached_prompt")
        best_cached_score = perceptions.get("best_cached_score", 0.0)
        
        # Determine whether to use cached prompt (only on first iteration)
        use_cached_prompt = (iteration == 1 and 
                            best_cached_prompt and 
                            best_cached_score > current_score)
        
        # Decide on optimization approach based on trends
        trend = score_trends.get("trend", "")
        
        if use_cached_prompt:
            strategy = "use_cached_prompt"
            temperature = 0.7  # Standard temperature
        elif trend == "negative":
            strategy = "radical_revision"
            temperature = 1.2  # High temperature for creative exploration
        elif trend == "stagnating":
            strategy = "structural_change"
            temperature = 1.0  # Higher temperature to escape plateau
        elif trend == "diminishing":
            strategy = "focused_enhancement"
            temperature = 0.8  # Moderate temperature
        else:  # "improving"
            strategy = "iterative_refinement"
            temperature = 0.7  # Standard temperature
            
        # Add randomness to temperature to avoid local optima
        # Higher iterations get more temperature variation
        temperature_variation = min(0.3, iteration * 0.02)
        temperature = max(0.5, min(1.3, temperature + random.uniform(-temperature_variation, temperature_variation)))
        
        # Decide on focus areas based on analysis
        focus_areas = []
        
        if analysis.get("unsuccessful_count", 0) > 0:
            focus_areas.append("unsuccessful_cases")
        
        if analysis.get("is_plateauing", False):
            focus_areas.append("structural_prompting")
            
        # If we have a good success rate, focus on making good cases even better
        if analysis.get("success_rate", 0) > 0.7:
            focus_areas.append("amplify_strengths")
            
        # Default focus if none selected
        if not focus_areas:
            focus_areas.append("general_improvement")
        
        # Create optimization plan
        plan = {
            "strategy": strategy,
            "temperature": temperature,
            "focus_areas": focus_areas,
            "use_cached_prompt": use_cached_prompt,
            "best_cached_prompt": best_cached_prompt if use_cached_prompt else None,
            "explanation_detail": "high" if current_score > 0.6 else "moderate"
        }
        
        self.logger.info(f"Selected strategy: {strategy} with temperature {temperature:.2f}")
        return plan
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an optimized prompt based on the plan.
        
        Args:
            plan: Plan of action for optimization
            
        Returns:
            Dictionary with optimized prompt and metadata
        """
        self.logger.info(f"Executing plan with strategy: {plan['strategy']}")
        
        # Get perceptions from memory
        perceptions = self.memory.get_from_short_term("latest_perceptions", {})
        if not perceptions:
            self.logger.error("No perceptions found in memory")
            return {"error": "No perceptions available for optimization"}
            
        # Extract needed data
        current_prompt = perceptions.get("current_prompt", "")
        evaluation_results = perceptions.get("evaluation_results", [])
        available_stories = perceptions.get("available_stories", [])
        current_score = perceptions.get("current_score", 0.0)
        
        # If using cached prompt, return it
        if plan.get("use_cached_prompt", False) and plan.get("best_cached_prompt"):
            optimized_prompt = plan["best_cached_prompt"]
            self.logger.info("Using cached best prompt")
        else:
            # Generate optimized prompt
            optimized_prompt = self._optimize_prompt(
                current_prompt, 
                evaluation_results, 
                available_stories,
                plan.get("strategy", "iterative_refinement"),
                plan.get("temperature", 0.7),
                plan.get("focus_areas", ["general_improvement"])
            )
        
        # Store the optimized prompt in cache
        self.prompt_cache.add_prompt(
            optimized_prompt, 
            current_score, 
            {
                "iteration": self.iteration, 
                "strategy": plan.get("strategy"),
                "temperature": plan.get("temperature")
            }
        )
        
        # Prepare result
        result = {
            "optimized_prompt": optimized_prompt,
            "strategy": plan.get("strategy"),
            "temperature": plan.get("temperature"),
            "focus_areas": plan.get("focus_areas", []),
            "iteration": self.iteration,
            "prompt_difference": len(optimized_prompt) - len(current_prompt)
        }
        
        # Store the result in memory
        self.memory.add_to_short_term("latest_optimization", result)
        
        self.logger.info(f"Generated optimized prompt (length: {len(optimized_prompt)} chars)")
        return result
    
    def _optimize_prompt(self, current_prompt: str, evaluation_results: List[Dict[str, Any]], 
                      stories: List[Dict[str, Any]], strategy: str, temperature: float,
                      focus_areas: List[str]) -> str:
        """
        Generate an optimized prompt using LLM.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            stories: List of available stories
            strategy: Optimization strategy to use
            temperature: Temperature for generation
            focus_areas: Areas to focus on for improvement
            
        Returns:
            Optimized prompt
        """
        # Prepare strategy-specific guidance
        strategy_guidance = {
            "iterative_refinement": """
                Refine the current prompt with targeted improvements while preserving its structure.
                Focus on improving how user preferences are matched to story attributes.
            """,
            "focused_enhancement": """
                Enhance specific sections of the prompt that handle preference matching.
                Pay special attention to how diverse preferences are balanced.
            """,
            "structural_change": """
                IMPORTANT: Make significant structural changes to the prompt to overcome plateauing.
                Consider completely different approaches to organizing information and reasoning.
                Try a different prompt structure, different reasoning flows, or different emphasis points.
            """,
            "radical_revision": """
                CRITICAL: The current approach is not working well. Start from first principles and 
                create a substantially different prompt with a fresh approach.
                Discard most of the current structure and reasoning, keeping only what's essential.
            """
        }
        
        # Prepare focus area guidance
        focus_guidance = {
            "unsuccessful_cases": f"""
                Focus on improving performance for unsuccessful cases.
                These tags were common in unsuccessful recommendations:
                {json.dumps(self._extract_common_tags([r for r in evaluation_results if r.get('score', 0) < 0.5]), indent=2)}
            """,
            "amplify_strengths": f"""
                Build upon what's already working well.
                These tags were common in successful recommendations:
                {json.dumps(self._extract_common_tags([r for r in evaluation_results if r.get('score', 0) >= 0.5]), indent=2)}
            """,
            "structural_prompting": """
                The optimization seems to have plateaued. Try a completely different prompt structure.
                Consider:
                1. Different ordering of information
                2. Different reasoning steps
                3. Different ways to match preferences to stories
                4. More explicit reasoning about matching logic
            """,
            "general_improvement": """
                Improve the overall quality of the prompt for better recommendations.
                Focus on clarity, precision, and comprehensive coverage of user preferences.
            """
        }
        
        # Calculate average score
        avg_score = sum(r.get("score", 0) for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
        
        # Create the optimization prompt
        optimization_prompt = f"""
        You are a prompt optimization expert. Your task is to improve a prompt for a story recommendation system.
        
        Current prompt:
        ```
        {current_prompt}
        ```
        
        Current average evaluation score: {avg_score:.4f}
        Iteration: {self.iteration}
        
        {strategy_guidance.get(strategy, '')}
        
        {' '.join(focus_guidance.get(focus, '') for focus in focus_areas)}
        
        Evaluation results:
        {json.dumps(evaluation_results[:3], indent=2)}
        
        Example stories:
        {json.dumps(stories[:3], indent=2)}
        
        Please create an improved version of the recommendation prompt that will:
        1. Better match user preferences to relevant story attributes
        2. Improve the precision of recommendations
        3. Handle diverse user preferences more effectively
        4. Be concise yet comprehensive
        
        Return ONLY the improved prompt without any explanation or additional text.
        """
        
        try:
            optimized_prompt = generate_with_model(
                prompt=optimization_prompt,
                model_name=self.model_name,
                system_message="You are an expert prompt engineer specializing in recommendation systems.",
                temperature=temperature,
                max_tokens=1000
            )
            
            # Basic validation
            if len(optimized_prompt) > 100:
                return optimized_prompt.strip()
                
        except Exception as e:
            self.logger.error(f"Error optimizing prompt: {str(e)}")
        
        # Fallback: return slightly modified current prompt
        self.logger.warning("Using fallback optimization approach")
        return self._fallback_optimization(current_prompt, evaluation_results)
        
    def _fallback_optimization(self, current_prompt: str, evaluation_results: List[Dict[str, Any]]) -> str:
        """
        Fallback optimization method when LLM optimization fails.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            
        Returns:
            Modified prompt
        """
        # Extract tags from evaluation results
        all_tags = []
        for result in evaluation_results:
            all_tags.extend(result.get("extracted_tags", []))
            
        # Count tag frequencies
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        # Get most common tags
        common_tags = [tag for tag, _ in tag_counts.most_common(10)]
        emphasis = ", ".join(common_tags)
        
        # Add emphasis to current prompt
        return current_prompt + f"\n\nPay special attention to these important themes and preferences: {emphasis}."
    
    def _fetch_best_prompt(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the best performing prompt from cache.
        
        Returns:
            Best prompt entry or None if cache is empty
        """
        return self.prompt_cache.get_best_prompt()
    
    def optimize(self, current_prompt: str, evaluation_results: List[Dict[str, Any]], 
               stories: List[Dict[str, Any]], previous_scores: List[float] = None,
               model_name: str = None) -> str:
        """
        Optimize a recommendation prompt based on evaluation results.
        This is a simplified interface for backwards compatibility.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            stories: List of all available stories
            previous_scores: Optional list of previous scores
            model_name: Optional model name to use
            
        Returns:
            Optimized prompt
        """
        # Update model if specified
        if model_name:
            self.model_name = model_name
            
        # Calculate current score
        current_score = sum(r.get("score", 0) for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
            
        # Prepare input data
        input_data = {
            "current_prompt": current_prompt,
            "evaluation_results": evaluation_results,
            "available_stories": stories,
            "current_score": current_score,
            "previous_scores": previous_scores or []
        }
        
        # Run the full perception-reasoning-action loop
        result = self.run_perception_reasoning_action_loop(input_data)
        
        # Return the optimized prompt
        return result.get("optimized_prompt", current_prompt)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'AdaptiveOptimizerAgent':
        """
        Create an AdaptiveOptimizerAgent instance from serialized data.
        
        Args:
            data: Serialized agent data
            
        Returns:
            AdaptiveOptimizerAgent instance
        """
        agent = cls(model_name=data.get("model_name"))
        agent.memory = Memory.deserialize(data.get("memory", {}))
        agent.goals = data.get("goals", [])
        agent.iteration = data.get("iteration", 0)
        
        return agent 