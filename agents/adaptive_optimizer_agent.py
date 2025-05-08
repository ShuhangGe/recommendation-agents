"""
Adaptive Optimizer Agent that analyzes evaluation results to optimize recommendation prompts.
This agent is inspired by AutoPrompt's Intent-based Prompt Calibration approach to 
systematically improve prompts by identifying and addressing edge cases.
"""
import json
import random
import time
from typing import List, Dict, Any, Optional
import numpy as np

import config
from utils.data_utils import generate_with_model, create_openai_client
from utils.cache_utils import PromptCache
from agents.agent_core import Agent, Memory
# Import prompt templates
from utils.prompt_templates import (
    INTENT_BASED_CALIBRATION_SYSTEM_PROMPT,
    OPTIMIZATION_STANDARD_TEMPLATE,
    OPTIMIZATION_EXPLORATION_TEMPLATE,
    OPTIMIZATION_REFINEMENT_TEMPLATE,
    FALLBACK_OPTIMIZATION_TEMPLATE,
    EDGE_CASE_TEST_TEMPLATE
)

class AdaptiveOptimizerAgent(Agent):
    """
    Adaptive agent that optimizes prompts based on evaluation results.
    
    Implements Intent-based Prompt Calibration from AutoPrompt to:
    1. Identify boundary/edge cases where recommendations fail
    2. Extract patterns from successful vs unsuccessful recommendations
    3. Generate improved prompts that handle these edge cases
    4. Maintain a memory of optimization history for detecting trends
    5. Balance exploration (trying new approaches) with exploitation (refining successful ones)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the adaptive optimizer agent.
        
        Args:
            model_name: Name of the model to use (defaults to config setting)
        """
        super().__init__(name="AdaptiveOptimizer", model_name=model_name or config.OPTIMIZER_MODEL)
        self.prompt_cache = PromptCache()
        
        # Tracking optimization progress
        self.optimization_history = []
        self.best_score = 0.0
        self.best_prompt = ""
        self.plateau_counter = 0
        self.min_improvement_threshold = 0.005  # 0.5% improvement threshold
        
        # Set initial goals
        self.set_goal("Analyze evaluation results to identify optimization opportunities")
        self.set_goal("Generate improved prompts that handle edge cases")
        self.set_goal("Balance prompt complexity with effectiveness")
        
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to understand the optimization context.
        
        Args:
            input_data: Dictionary containing current prompt, evaluation results, etc.
            
        Returns:
            Dictionary of perceptions
        """
        self.logger.info("Perceiving optimization context and evaluation results")
        
        # Extract input data
        current_prompt = input_data.get("current_prompt", "")
        evaluation_results = input_data.get("evaluation_results", [])
        current_score = input_data.get("current_score", 0.0)
        iteration = input_data.get("iteration", 1)
        edge_cases = input_data.get("edge_cases", [])
        
        # Group evaluation results by user
        user_evaluations = {}
        for result in evaluation_results:
            user_id = result.get("user_id", "unknown_user")
            if user_id not in user_evaluations:
                user_evaluations[user_id] = []
            user_evaluations[user_id].append(result)
        
        # Get previous scores and update
        previous_scores = self.memory.get_from_short_term("previous_scores", [])
        previous_scores.append(current_score)
        self.memory.add_to_short_term("previous_scores", previous_scores[-5:])
        
        # Analyze results by user
        user_analyses = {}
        for user_id, user_results in user_evaluations.items():
            user_analyses[user_id] = self._analyze_results(user_results, current_score, previous_scores)
        
        # Perform overall analysis across all users
        overall_analysis = self._analyze_results(evaluation_results, current_score, previous_scores)
        
        perceptions = {
            "current_prompt": current_prompt,
            "evaluation_results": evaluation_results,
            "user_evaluations": user_evaluations,
            "user_analyses": user_analyses,
            "overall_analysis": overall_analysis,
            "current_score": current_score,
            "iteration": iteration,
            "edge_cases": edge_cases,
            "previous_scores": previous_scores[-5:],
            "score_trends": self._analyze_score_trends(previous_scores)
        }
        
        # Store perceptions in memory
        self.memory.add_to_short_term("latest_perceptions", perceptions)
        
        self.logger.info(f"Current score: {current_score:.4f}, Best score: {self.best_score:.4f}")
        return perceptions
    
    def _analyze_results(self, evaluation_results: List[Dict[str, Any]], 
                       current_score: float, previous_scores: List[float]) -> Dict[str, Any]:
        """
        Analyze evaluation results to identify patterns and edge cases.
        
        Args:
            evaluation_results: List of evaluation results for each user
            current_score: Current average score
            previous_scores: List of previous scores
            
        Returns:
            Dictionary with analysis results
        """
        if not evaluation_results:
            return {"status": "insufficient_data", "edge_cases": []}
            
        # Identify edge cases (low-scoring results)
        edge_cases = []
        for result in evaluation_results:
            score = result.get("score", 0)
            if score < 0.5:  # Consider low scores as edge cases
                edge_cases.append({
                    "user_id": result.get("user_id", "unknown"),
                    "score": score,
                    "strategy": result.get("strategy", "unknown"),
                    "weaknesses": result.get("weaknesses", [])
                })
        
        # Extract common weakness patterns
        weakness_patterns = {}
        for case in edge_cases:
            for weakness in case.get("weaknesses", []):
                weakness_type = weakness.get("type", "unknown")
                weakness_patterns[weakness_type] = weakness_patterns.get(weakness_type, 0) + 1
        
        # Sort weaknesses by frequency
        sorted_weaknesses = sorted(weakness_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate improvement trend
        improvement = 0
        if len(previous_scores) > 0:
            last_score = previous_scores[-1] if previous_scores else 0
            improvement = current_score - last_score
        
        # Check for plateau
        is_plateau = len(previous_scores) >= 3 and all(
            abs(current_score - prev) < self.min_improvement_threshold 
            for prev in previous_scores[-3:]
        )
        
        # Identify most common strategy in high-scoring cases
        high_score_strategies = {}
        for result in evaluation_results:
            if result.get("score", 0) > 0.7:
                strategy = result.get("strategy", "unknown")
                high_score_strategies[strategy] = high_score_strategies.get(strategy, 0) + 1
        
        best_strategy = max(high_score_strategies.items(), key=lambda x: x[1])[0] if high_score_strategies else "unknown"
        
        return {
            "edge_cases": edge_cases,
            "edge_case_count": len(edge_cases),
            "weakness_patterns": sorted_weaknesses,
            "improvement": improvement,
            "is_plateau": is_plateau,
            "best_strategy": best_strategy
        }
    
    def _extract_common_tags(self, cases: List[Dict[str, Any]]) -> List[str]:
        """
        Extract common tags from a list of cases.
        
        Args:
            cases: List of evaluation cases
            
        Returns:
            List of common tags
        """
        if not cases:
            return []
            
        # Extract all tags mentioned in strengths or weaknesses
        all_tags = []
        for case in cases:
            strengths = case.get("strengths", [])
            weaknesses = case.get("weaknesses", [])
            
            for item in strengths + weaknesses:
                tags = item.get("tags", [])
                all_tags.extend(tags)
                
            # Add any explicitly mentioned tags
            if "extracted_tags" in case:
                all_tags.extend(case.get("extracted_tags", []))
        
        # Count occurrences of each tag
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
        # Sort by frequency and return top tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:10]]
    
    def _analyze_score_trends(self, scores: List[float]) -> Dict[str, Any]:
        """
        Analyze trends in optimization scores.
        
        Args:
            scores: List of scores from optimization history
            
        Returns:
            Dictionary with trend analysis
        """
        if len(scores) < 2:
            return {"trend": "insufficient_data"}
            
        # Calculate differences between consecutive scores
        differences = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        # Calculate moving average of differences (last 3)
        window_size = min(3, len(differences))
        recent_diffs = differences[-window_size:]
        avg_recent_diff = sum(recent_diffs) / window_size
        
        # Determine overall trend
        if avg_recent_diff > self.min_improvement_threshold:
            trend = "improving"
        elif avg_recent_diff < -self.min_improvement_threshold:
            trend = "declining"
        else:
            trend = "plateau"
            
        # Check for convergence (diminishing improvements)
        is_converging = False
        if len(differences) >= 3:
            # Check if improvements are getting smaller
            if all(differences[i] < differences[i-1] for i in range(len(differences)-1, len(differences)-3, -1)) and differences[-1] > 0:
                is_converging = True
                
        # Calculate volatility (standard deviation of recent differences)
        volatility = np.std(recent_diffs) if len(recent_diffs) > 1 else 0
        
        return {
            "trend": trend,
            "avg_recent_diff": avg_recent_diff,
            "is_converging": is_converging,
            "volatility": float(volatility),
            "latest_score": scores[-1] if scores else 0
        }
    
    def reason(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best optimization strategy based on perceptions.
        
        Args:
            perceptions: Dictionary of perceptions about evaluation results
            
        Returns:
            Plan of action for optimization
        """
        self.logger.info("Reasoning about optimization strategy")
        
        # Extract key perceptions
        current_prompt = perceptions.get("current_prompt", "")
        current_score = perceptions.get("current_score", 0.0)
        iteration = perceptions.get("iteration", 0)
        result_analysis = perceptions.get("result_analysis", {})
        successful_tags = perceptions.get("successful_tags", [])
        unsuccessful_tags = perceptions.get("unsuccessful_tags", [])
        score_trend = perceptions.get("score_trend", {})
        plateau_counter = perceptions.get("plateau_counter", 0)
        
        # Determine if we should try something radically different
        should_explore = (
            plateau_counter >= 3 or  # Stuck in plateau for too long
            score_trend.get("trend") == "declining" or  # Scores are declining
            (current_score < 0.3 and iteration > 2)  # Really poor performance
        )
        
        # Should we focus on refining what works?
        should_exploit = (
            score_trend.get("trend") == "improving" and  # Scores are improving
            not score_trend.get("is_converging", False) and  # Not converging yet
            current_score > 0.6  # Already decent performance
        )
        
        # Determine areas to focus on based on weaknesses
        focus_areas = []
        if result_analysis and "weakness_patterns" in result_analysis:
            for weakness_type, count in result_analysis.get("weakness_patterns", []):
                if count >= 2:  # Weakness appears multiple times
                    focus_areas.append(weakness_type)
                    
        # If no clear weaknesses, focus on general improvements
        if not focus_areas:
            focus_areas = ["relevance", "diversity", "specificity"]
            
        # Determine optimization strategy
        if should_explore:
            strategy = "explore_new_approach"
            temperature = 0.8  # Higher temperature for more creativity
        elif should_exploit:
            strategy = "refine_current_approach"
            temperature = 0.4  # Lower temperature for more focused refinement
        else:
            strategy = "balanced_optimization"
            temperature = 0.6  # Balanced temperature
            
        # Check if we should revert to the best prompt and try again
        should_revert = (
            score_trend.get("trend") == "declining" and
            current_score < perceptions.get("best_score", 0) - 0.1  # Significant decline from best
        )
        
        # Create the optimization plan
        plan = {
            "strategy": strategy,
            "temperature": temperature,
            "focus_areas": focus_areas,
            "should_revert": should_revert,
            "best_strategy": result_analysis.get("best_strategy", "hybrid_search"),
            "edge_case_count": result_analysis.get("edge_case_count", 0)
        }
        
        # Log the chosen strategy
        self.logger.info(f"Selected optimization strategy: {strategy} (Temperature: {temperature})")
        self.logger.info(f"Focus areas: {', '.join(focus_areas)}")
        
        return plan
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an optimized prompt based on the plan.
        
        Args:
            plan: Plan of action for optimization
            
        Returns:
            Dictionary with optimized prompt
        """
        self.logger.info("Executing optimization plan")
        
        # Get perceptions from memory
        perceptions = self.memory.get_from_short_term("latest_perceptions", {})
        if not perceptions:
            self.logger.error("No perceptions found in memory")
            return {"error": "No perceptions available for optimization"}
            
        # Extract needed data
        current_prompt = perceptions.get("current_prompt", "")
        evaluation_results = perceptions.get("evaluation_results", [])
        current_score = perceptions.get("current_score", 0.0)
        
        # Get optimization parameters from plan
        strategy = plan.get("strategy", "balanced_optimization")
        temperature = plan.get("temperature", 0.6)
        focus_areas = plan.get("focus_areas", ["relevance"])
        should_revert = plan.get("should_revert", False)
        best_strategy = plan.get("best_strategy", "hybrid_search")
        
        # If we should revert to the best prompt
        if should_revert and self.best_prompt:
            self.logger.info("Reverting to best prompt and trying a different approach")
            current_prompt = self.best_prompt
        
        # Try to fetch stories from evaluation results
        stories = []
        for result in evaluation_results:
            if "recommended_stories" in result:
                stories.extend(result["recommended_stories"][:3])  # Take a few example stories
                break
                
        # Optimize the prompt
        optimized_prompt = self._optimize_prompt(
            current_prompt, 
            evaluation_results, 
            stories, 
            best_strategy,
            temperature,
            focus_areas
        )
        
        # If optimization failed, try fallback approach
        if not optimized_prompt or optimized_prompt == current_prompt:
            self.logger.warning("Primary optimization failed, trying fallback approach")
            optimized_prompt = self._fallback_optimization(current_prompt, evaluation_results)
        
        # If fallback also failed, try cached prompt
        if not optimized_prompt or optimized_prompt == current_prompt:
            self.logger.warning("Fallback optimization failed, trying cached prompt")
            best_cached = self._fetch_best_prompt()
            if best_cached and best_cached["score"] > current_score:
                optimized_prompt = best_cached["prompt"]
                
        # Ensure we have a prompt, even if it's the current one
        if not optimized_prompt:
            optimized_prompt = current_prompt
            
        # Store the optimized prompt and score in cache
        if optimized_prompt != current_prompt:
            self.prompt_cache.add_prompt(
                optimized_prompt, 
                current_score,
                {"timestamp": time.time(), "strategy": strategy}
            )
        
        # Return the optimization result
        result = {
            "optimized_prompt": optimized_prompt,
            "optimization_strategy": strategy,
            "focus_areas": focus_areas,
            "timestamp": time.time()
        }
        
        self.logger.info("Optimization complete")
        return result
    
    def _optimize_prompt(self, current_prompt: str, evaluation_results: List[Dict[str, Any]], 
                      stories: List[Dict[str, Any]], strategy: str, temperature: float,
                      focus_areas: List[str]) -> str:
        """
        Optimize the prompt using Intent-based Prompt Calibration.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            stories: List of example stories
            strategy: Recommendation strategy that works best
            temperature: Temperature for generation
            focus_areas: Areas to focus optimization on
            
        Returns:
            Optimized prompt text
        """
        # Extract successful and unsuccessful cases
        successful_cases = [r for r in evaluation_results if r.get("score", 0) > 0.7]
        unsuccessful_cases = [r for r in evaluation_results if r.get("score", 0) <= 0.3]
        
        # If no evaluation data, use a more generic approach
        if not evaluation_results:
            self.logger.warning("No evaluation results available, using generic optimization")
            return self._fallback_optimization(current_prompt, [])

        # Calculate average score        
        avg_score = sum(r.get("score", 0) for r in evaluation_results) / max(1, len(evaluation_results))
            
        # Extract example cases for learning
        example_successes = []
        for case in successful_cases[:2]:  # Limit to 2 examples
            user_id = case.get("user_id", "unknown")
            score = case.get("score", 0)
            strengths = [s.get("description", "") for s in case.get("strengths", [])]
            example_successes.append({
                "user": user_id,
                "score": score,
                "strengths": strengths
            })
            
        example_failures = []
        for case in unsuccessful_cases[:2]:  # Limit to 2 examples
            user_id = case.get("user_id", "unknown")
            score = case.get("score", 0)
            weaknesses = [w.get("description", "") for w in case.get("weaknesses", [])]
            example_failures.append({
                "user": user_id,
                "score": score,
                "weaknesses": weaknesses
            })
        
        # Format story examples    
        story_examples = json.dumps([{"title": s.get("title", ""), "tags": s.get("tags", [])} for s in stories[:3]], indent=2)
        
        # Extract weakness patterns for exploration template
        weakness_patterns = ""
        if evaluation_results:
            weakness_counts = {}
            for result in unsuccessful_cases:
                for weakness in result.get("weaknesses", []):
                    weakness_type = weakness.get("description", "")
                    weakness_counts[weakness_type] = weakness_counts.get(weakness_type, 0) + 1
                    
            # Format weakness patterns
            weakness_patterns = "\n".join([f"- {weakness} (found {count} times)" 
                                          for weakness, count in sorted(weakness_counts.items(), 
                                                                      key=lambda x: x[1], 
                                                                      reverse=True)[:5]])
            if not weakness_patterns:
                weakness_patterns = "No specific weaknesses identified"
        
        # Format edge case examples for refinement template
        edge_case_examples = ""
        if unsuccessful_cases:
            edge_case_tags = set()
            for case in unsuccessful_cases:
                for weakness in case.get("weaknesses", []):
                    edge_case_tags.update(weakness.get("tags", []))
            
            edge_case_examples = ", ".join(list(edge_case_tags)[:10])
            if not edge_case_examples:
                edge_case_examples = "ambiguous preferences, conflicting interests, rare combinations"
            
        # Select the appropriate optimization template based on strategy
        if strategy == "explore_new_approach":
            # Use exploration template for radical changes
            prompt_template = OPTIMIZATION_EXPLORATION_TEMPLATE.format(
                current_prompt=current_prompt,
                average_score=avg_score,
                plateau_counter=self.plateau_counter,
                weakness_patterns=weakness_patterns
            )
        elif strategy == "refine_current_approach":
            # Use refinement template for incremental improvements
            prompt_template = OPTIMIZATION_REFINEMENT_TEMPLATE.format(
                current_prompt=current_prompt,
                average_score=avg_score,
                best_strategy=strategy,
                focus_areas=", ".join(focus_areas),
                edge_case_examples=edge_case_examples
            )
        else:
            # Use standard template for balanced optimization
            prompt_template = OPTIMIZATION_STANDARD_TEMPLATE.format(
                current_prompt=current_prompt,
                successful_count=len(successful_cases),
                unsuccessful_count=len(unsuccessful_cases),
                average_score=avg_score,
                best_strategy=strategy,
                success_examples=json.dumps(example_successes, indent=2),
                failure_examples=json.dumps(example_failures, indent=2),
                focus_areas=", ".join(focus_areas),
                story_examples=story_examples
            )
            
        try:
            # Generate the optimized prompt
            response = generate_with_model(
                prompt=prompt_template,
                model_name=self.model_name,
                system_message=INTENT_BASED_CALIBRATION_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=1000
            )
            
            # Extract the optimized prompt
            optimized_prompt = response.strip()
            
            # Remove any markdown formatting or section headers
            lines = optimized_prompt.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.startswith("#") or line.startswith("```"):
                    continue
                cleaned_lines.append(line)
                
            optimized_prompt = "\n".join(cleaned_lines).strip()
            
            # Ensure the prompt is meaningful
            if len(optimized_prompt) < 20:
                return current_prompt
                
            return optimized_prompt
            
        except Exception as e:
            self.logger.error(f"Error optimizing prompt: {str(e)}")
            return current_prompt
    
    def _fallback_optimization(self, current_prompt: str, evaluation_results: List[Dict[str, Any]]) -> str:
        """
        Fallback method for prompt optimization when primary method fails.
        
        Args:
            current_prompt: Current recommendation prompt
            evaluation_results: List of evaluation results
            
        Returns:
            Optimized prompt text
        """
        # Use the fallback template
        prompt = FALLBACK_OPTIMIZATION_TEMPLATE.format(
            current_prompt=current_prompt
        )
        
        try:
            # Generate the optimized prompt
            response = generate_with_model(
                prompt=prompt,
                model_name=self.model_name,
                system_message=INTENT_BASED_CALIBRATION_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=800
            )
            
            # Clean up the response
            optimized_prompt = response.strip()
            
            # Ensure the prompt is meaningful
            if len(optimized_prompt) < 20:
                return current_prompt
                
            return optimized_prompt
            
        except Exception as e:
            self.logger.error(f"Error in fallback optimization: {str(e)}")
            return current_prompt
    
    def _fetch_best_prompt(self) -> Optional[Dict[str, Any]]:
        """Get the best performing prompt from the cache."""
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
            stories: List of stories in the system
            previous_scores: Optional list of previous scores
            model_name: Optional model name to use
            
        Returns:
            Optimized prompt text
        """
        # Update model if specified
        if model_name:
            self.model_name = model_name
            
        # Prepare input data
        input_data = {
            "current_prompt": current_prompt,
            "evaluation_results": evaluation_results,
            "current_score": sum(r.get("score", 0) for r in evaluation_results) / max(1, len(evaluation_results)) if evaluation_results else 0,
            "iteration": len(self.optimization_history) + 1
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
        agent.optimization_history = data.get("optimization_history", [])
        agent.best_score = data.get("best_score", 0.0)
        agent.best_prompt = data.get("best_prompt", "")
        agent.plateau_counter = data.get("plateau_counter", 0)
        
        return agent 