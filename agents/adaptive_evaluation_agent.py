"""
Adaptive Evaluation Agent that intelligently assesses recommendation quality.
This agent adapts its evaluation strategy based on user preferences and context.
"""
import json
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from utils.data_utils import generate_with_model, create_openai_client
from utils.metrics import calculate_metric, precision_at_k, recall, semantic_overlap
from agents.agent_core import Agent, Memory

class AdaptiveEvaluationAgent(Agent):
    """
    Adaptive agent that evaluates recommendation quality.
    This agent can:
    1. Extract and analyze user preferences from profiles
    2. Generate "ground truth" recommendations 
    3. Evaluate recommendations using multiple strategies
    4. Provide detailed feedback on recommendation quality
    """
    
    def __init__(self, stories: List[Dict[str, Any]], model_name: str = None):
        """
        Initialize the adaptive evaluation agent.
        
        Args:
            stories: List of all available stories
            model_name: Name of the model to use (defaults to config setting)
        """
        super().__init__(name="AdaptiveEvaluator", model_name=model_name or config.EVALUATION_MODEL)
        self.stories = stories
        self.story_dict = {s["id"]: s for s in stories}
        
        # Register available tools
        self.register_tool("extract_preferences", self._extract_preferences)
        self.register_tool("generate_ground_truth", self._generate_ground_truth)
        self.register_tool("evaluate_precision", self._evaluate_precision)
        self.register_tool("evaluate_recall", self._evaluate_recall)
        self.register_tool("evaluate_semantic", self._evaluate_semantic)
        self.register_tool("analyze_quality", self._analyze_quality)
        
        # Set initial goals
        self.set_goal("Accurately assess recommendation quality")
        self.set_goal("Identify strengths and weaknesses in recommendations")
        self.set_goal("Provide actionable feedback for improvement")
    
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to understand the evaluation context.
        
        Args:
            input_data: Dictionary containing user profile, recommendations, etc.
            
        Returns:
            Dictionary of perceptions
        """
        self.logger.info("Perceiving evaluation context")
        
        # Extract input data
        user_profile = input_data.get("user_profile", "")
        recommended_ids = input_data.get("recommended_ids", [])
        recommended_stories = input_data.get("recommended_stories", [])
        metric = input_data.get("metric", config.METRIC)
        existing_tags = input_data.get("extracted_tags", [])
        
        # If we don't have recommended stories but have IDs, get the stories
        if not recommended_stories and recommended_ids:
            recommended_stories = [self.story_dict[sid] for sid in recommended_ids if sid in self.story_dict]
        
        # Extract or use provided tags
        extracted_tags = existing_tags or self._extract_preferences(user_profile)
        
        # Perceptions about the evaluation task
        perceptions = {
            "user_profile": user_profile,
            "extracted_tags": extracted_tags,
            "recommended_stories": recommended_stories,
            "metric": metric,
            "profile_complexity": self._analyze_profile_complexity(user_profile),
            "recommendation_diversity": self._analyze_recommendation_diversity(recommended_stories),
            "tag_relevance": self._analyze_tag_relevance(extracted_tags, recommended_stories)
        }
        
        # Store perceptions in memory
        self.memory.add_to_short_term("latest_perceptions", perceptions)
        
        self.logger.info(f"Perceived {len(extracted_tags)} user preferences and {len(recommended_stories)} recommendations")
        return perceptions
    
    def _analyze_profile_complexity(self, user_profile: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a user profile.
        
        Args:
            user_profile: User profile text
            
        Returns:
            Dictionary with complexity metrics
        """
        if not user_profile:
            return {"level": "unknown", "score": 0.0}
            
        # Simple metrics for complexity
        word_count = len(user_profile.split())
        unique_words = len(set(user_profile.lower().split()))
        avg_word_length = sum(len(word) for word in user_profile.split()) / max(1, word_count)
        
        # Check for specific keywords indicating complexity
        complex_indicators = ["strategy", "ambiguity", "complex", "nuanced", "balance", 
                             "combination", "hybrid", "specific", "detailed"]
        
        indicator_count = sum(1 for indicator in complex_indicators 
                             if indicator in user_profile.lower())
        
        # Calculate overall complexity score
        factors = [
            min(1.0, word_count / 100),  # Normalize word count
            min(1.0, unique_words / 50),  # Normalize unique words
            min(1.0, avg_word_length / 6),  # Normalize avg word length
            min(1.0, indicator_count / 3)  # Normalize indicator count
        ]
        
        complexity_score = sum(factors) / len(factors)
        
        # Determine complexity level
        if complexity_score < 0.3:
            level = "simple"
        elif complexity_score < 0.6:
            level = "moderate"
        else:
            level = "complex"
            
        return {
            "level": level,
            "score": complexity_score,
            "factors": {
                "word_count": word_count,
                "unique_words": unique_words,
                "avg_word_length": avg_word_length,
                "indicator_count": indicator_count
            }
        }
    
    def _analyze_recommendation_diversity(self, stories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the diversity of recommendations.
        
        Args:
            stories: List of recommended stories
            
        Returns:
            Dictionary with diversity metrics
        """
        if not stories:
            return {"level": "unknown", "score": 0.0}
            
        # Collect all tags
        all_tags = []
        for story in stories:
            all_tags.extend(story.get("tags", []))
            
        # Count tag frequencies
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        # Calculate metrics
        unique_tags = len(tag_counts)
        total_tags = len(all_tags)
        
        # Average number of times each tag appears
        avg_tag_frequency = total_tags / max(1, unique_tags)
        
        # Unique tags per story
        avg_unique_tags_per_story = unique_tags / max(1, len(stories))
        
        # Calculate diversity score
        # Higher when many unique tags and lower avg frequency
        diversity_score = min(1.0, (unique_tags / 20) * (1 / max(1, avg_tag_frequency/2)))
        
        # Determine diversity level
        if diversity_score < 0.3:
            level = "low"
        elif diversity_score < 0.7:
            level = "moderate"
        else:
            level = "high"
            
        return {
            "level": level,
            "score": diversity_score,
            "metrics": {
                "unique_tags": unique_tags,
                "total_tags": total_tags,
                "avg_tag_frequency": avg_tag_frequency,
                "avg_unique_tags_per_story": avg_unique_tags_per_story
            }
        }
    
    def _analyze_tag_relevance(self, tags: List[str], stories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze how well tags match with recommended stories.
        
        Args:
            tags: List of user preference tags
            stories: List of recommended stories
            
        Returns:
            Dictionary with tag relevance metrics
        """
        if not tags or not stories:
            return {"level": "unknown", "score": 0.0}
            
        # Count matches for each tag
        tag_matches = {}
        for tag in tags:
            matching_stories = sum(1 for story in stories if tag in story.get("tags", []))
            tag_matches[tag] = matching_stories
            
        # Calculate metrics
        total_matches = sum(tag_matches.values())
        max_possible_matches = len(tags) * len(stories)
        
        # Tags with at least one match
        tags_with_matches = sum(1 for count in tag_matches.values() if count > 0)
        
        # Average matches per tag
        avg_matches_per_tag = total_matches / max(1, len(tags))
        
        # Calculate relevance score
        relevance_score = min(1.0, (total_matches / max(1, max_possible_matches)) * 
                              (tags_with_matches / max(1, len(tags))))
        
        # Determine relevance level
        if relevance_score < 0.3:
            level = "low"
        elif relevance_score < 0.7:
            level = "moderate"
        else:
            level = "high"
            
        return {
            "level": level,
            "score": relevance_score,
            "metrics": {
                "total_matches": total_matches,
                "max_possible_matches": max_possible_matches,
                "tags_with_matches": tags_with_matches,
                "tag_matches": tag_matches,
                "avg_matches_per_tag": avg_matches_per_tag
            }
        }
    
    def reason(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best evaluation strategy based on perceptions.
        
        Args:
            perceptions: Dictionary of perceptions
            
        Returns:
            Plan of action for evaluating recommendations
        """
        self.logger.info("Reasoning about evaluation strategy")
        
        # Extract perceptions
        user_profile = perceptions.get("user_profile", "")
        extracted_tags = perceptions.get("extracted_tags", [])
        recommended_stories = perceptions.get("recommended_stories", [])
        metric = perceptions.get("metric", config.METRIC)
        profile_complexity = perceptions.get("profile_complexity", {})
        recommendation_diversity = perceptions.get("recommendation_diversity", {})
        tag_relevance = perceptions.get("tag_relevance", {})
        
        # Determine what metrics to use based on context
        metrics_to_use = ["precision@10"]  # Always include precision@10
        
        # Add recall if profile is complex and we likely have specific preferences
        if profile_complexity.get("level") == "complex":
            metrics_to_use.append("recall")
            
        # Add semantic overlap if diversity is high or relevance is low
        if (recommendation_diversity.get("level") == "high" or 
            tag_relevance.get("level") == "low"):
            metrics_to_use.append("semantic_overlap")
            
        # Determine if we should use weighted scoring
        use_weighted_scoring = profile_complexity.get("level") in ["moderate", "complex"]
        
        # Set weights for combined scoring (if using weighted scoring)
        metric_weights = {
            "precision@10": 0.6,
            "recall": 0.2,
            "semantic_overlap": 0.2
        }
        
        # Determine how much detail to include in the evaluation
        detail_level = "high" if profile_complexity.get("level") == "complex" else "moderate"
        
        # Combine into a plan
        plan = {
            "metrics_to_use": metrics_to_use,
            "primary_metric": metric,  # User's preferred metric takes precedence
            "use_weighted_scoring": use_weighted_scoring,
            "metric_weights": metric_weights,
            "detail_level": detail_level,
            "analyze_strengths": True,
            "analyze_weaknesses": True,
            "generate_ground_truth": extracted_tags and len(extracted_tags) > 2
        }
        
        self.logger.info(f"Selected metrics: {', '.join(metrics_to_use)}")
        return plan
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate recommendations based on the plan.
        
        Args:
            plan: Plan of action for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Executing evaluation plan")
        
        # Get perceptions from memory
        perceptions = self.memory.get_from_short_term("latest_perceptions", {})
        if not perceptions:
            self.logger.error("No perceptions found in memory")
            return {"error": "No perceptions available for evaluation"}
            
        # Extract needed data
        user_profile = perceptions.get("user_profile", "")
        extracted_tags = perceptions.get("extracted_tags", [])
        recommended_stories = perceptions.get("recommended_stories", [])
        
        # Execute the plan
        results = {}
        
        # Generate ground truth if needed
        if plan.get("generate_ground_truth", True):
            ground_truth_ids = self._generate_ground_truth(user_profile)
            ground_truth_stories = [self.story_dict[sid] for sid in ground_truth_ids if sid in self.story_dict]
            results["ground_truth_ids"] = ground_truth_ids
            results["ground_truth_stories"] = ground_truth_stories
        else:
            # If we don't generate ground truth, we can't calculate metrics
            ground_truth_stories = []
            
        # Calculate requested metrics
        metric_scores = {}
        for metric_name in plan.get("metrics_to_use", ["precision@10"]):
            if metric_name == "precision@10":
                score = self._evaluate_precision(recommended_stories, ground_truth_stories)
            elif metric_name == "recall":
                score = self._evaluate_recall(recommended_stories, ground_truth_stories)
            elif metric_name == "semantic_overlap":
                score = self._evaluate_semantic(recommended_stories, ground_truth_stories)
            else:
                score = 0.0
                
            metric_scores[metric_name] = score
            
        # Calculate primary metric (user's preferred metric)
        primary_metric = plan.get("primary_metric", config.METRIC)
        if primary_metric in metric_scores:
            primary_score = metric_scores[primary_metric]
        else:
            # Calculate the primary metric if not already calculated
            if primary_metric == "precision@10":
                primary_score = self._evaluate_precision(recommended_stories, ground_truth_stories)
            elif primary_metric == "recall":
                primary_score = self._evaluate_recall(recommended_stories, ground_truth_stories)
            elif primary_metric == "semantic_overlap":
                primary_score = self._evaluate_semantic(recommended_stories, ground_truth_stories)
            else:
                primary_score = metric_scores.get("precision@10", 0.0)  # Default to precision@10
                
            metric_scores[primary_metric] = primary_score
            
        # Calculate weighted score if requested
        if plan.get("use_weighted_scoring", False) and len(metric_scores) > 1:
            weights = plan.get("metric_weights", {})
            weighted_sum = sum(metric_scores.get(m, 0) * weights.get(m, 1) for m in metric_scores)
            weight_sum = sum(weights.get(m, 1) for m in metric_scores)
            weighted_score = weighted_sum / weight_sum if weight_sum else 0.0
            results["weighted_score"] = weighted_score
            
        # Add detailed quality analysis if requested
        if plan.get("analyze_strengths", False) or plan.get("analyze_weaknesses", False):
            quality_analysis = self._analyze_quality(
                recommended_stories, 
                ground_truth_stories, 
                extracted_tags,
                plan.get("detail_level", "moderate")
            )
            results["quality_analysis"] = quality_analysis
            
        # Compile results
        evaluation_results = {
            "extracted_tags": extracted_tags,
            "scores": metric_scores,
            "primary_metric": primary_metric,
            "score": primary_score,  # The main score (for compatibility)
            "timestamp": self.memory.last_updated
        }
        
        # Add other results
        evaluation_results.update(results)
        
        # Store the result in memory
        self.memory.add_to_short_term("latest_evaluation", evaluation_results)
        
        self.logger.info(f"Evaluation complete. Primary score: {primary_score:.4f}")
        return evaluation_results
    
    def _extract_preferences(self, user_profile: str) -> List[str]:
        """
        Extract user preference tags from profile.
        
        Args:
            user_profile: User profile text
            
        Returns:
            List of extracted tags
        """
        if not user_profile:
            return []
            
        # Use LLM to extract preference tags
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
        
        try:
            response = generate_with_model(
                prompt=prompt,
                model_name=self.model_name,
                system_message="You are an expert at analyzing user preferences and interests.",
                temperature=0.7
            )
            
            # Extract JSON list
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Error extracting preferences: {str(e)}")
            
            # Fallback to simple extraction
            words = user_profile.lower().split()
            # Simple filtering for meaningful words
            tags = [w for w in words if len(w) > 4 and w not in 
                    ["would", "could", "should", "their", "there", "these", "those", "about", "after", "again", "among"]]
            return list(set(tags))[:15]  # Deduplicate and limit to 15 tags
    
    def _generate_ground_truth(self, user_profile: str) -> List[str]:
        """
        Generate ground truth recommendations based on the full user profile.
        
        Args:
            user_profile: User profile text
            
        Returns:
            List of recommended story IDs
        """
        if not user_profile:
            # If no profile, return random stories
            return [s["id"] for s in random.sample(self.stories, min(10, len(self.stories)))]
            
        # Use LLM to generate ground truth recommendations
        prompt = f"""
        You are an evaluation agent for Sekai, a platform for interactive stories.
        
        Given a user profile and a list of available stories, recommend the 10 most relevant stories for this user.
        Use your understanding of the user's preferences to identify the best matches.
        
        User profile:
        {user_profile}
        
        Available stories:
        {json.dumps(self.stories, indent=2)}
        
        Return ONLY a valid JSON list of the 10 most relevant story IDs, like:
        ["123456", "234567", "345678", ...]
        
        Do not include any explanation or additional text in your response.
        """
        
        try:
            response = generate_with_model(
                prompt=prompt,
                model_name=self.model_name,
                system_message="You are an expert at matching user preferences to content.",
                temperature=0.7
            )
            
            # Extract JSON list
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
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
        except Exception as e:
            self.logger.error(f"Error generating ground truth: {str(e)}")
        
        # Fallback to random stories
        return [s["id"] for s in random.sample(self.stories, min(10, len(self.stories)))]
    
    def _evaluate_precision(self, recommended_stories: List[Dict[str, Any]], 
                          ground_truth_stories: List[Dict[str, Any]]) -> float:
        """
        Evaluate precision@10 metric.
        
        Args:
            recommended_stories: List of recommended stories
            ground_truth_stories: List of ground truth stories
            
        Returns:
            Precision@10 score
        """
        if not recommended_stories or not ground_truth_stories:
            return 0.0
            
        # Extract IDs
        recommended_ids = [s["id"] for s in recommended_stories]
        ground_truth_ids = [s["id"] for s in ground_truth_stories]
        
        return precision_at_k(recommended_ids, ground_truth_ids, k=10)
    
    def _evaluate_recall(self, recommended_stories: List[Dict[str, Any]], 
                       ground_truth_stories: List[Dict[str, Any]]) -> float:
        """
        Evaluate recall metric.
        
        Args:
            recommended_stories: List of recommended stories
            ground_truth_stories: List of ground truth stories
            
        Returns:
            Recall score
        """
        if not recommended_stories or not ground_truth_stories:
            return 0.0
            
        # Extract IDs
        recommended_ids = [s["id"] for s in recommended_stories]
        ground_truth_ids = [s["id"] for s in ground_truth_stories]
        
        return recall(recommended_ids, ground_truth_ids)
    
    def _evaluate_semantic(self, recommended_stories: List[Dict[str, Any]], 
                         ground_truth_stories: List[Dict[str, Any]]) -> float:
        """
        Evaluate semantic overlap metric.
        
        Args:
            recommended_stories: List of recommended stories
            ground_truth_stories: List of ground truth stories
            
        Returns:
            Semantic overlap score
        """
        if not recommended_stories or not ground_truth_stories:
            return 0.0
            
        return semantic_overlap(recommended_stories, ground_truth_stories)
    
    def _analyze_quality(self, recommended_stories: List[Dict[str, Any]], 
                       ground_truth_stories: List[Dict[str, Any]],
                       user_tags: List[str], detail_level: str = "moderate") -> Dict[str, Any]:
        """
        Analyze recommendation quality in detail.
        
        Args:
            recommended_stories: List of recommended stories
            ground_truth_stories: List of ground truth stories
            user_tags: User preference tags
            detail_level: Level of detail for analysis (low, moderate, high)
            
        Returns:
            Dictionary with quality analysis
        """
        if not recommended_stories:
            return {"error": "No recommendations to analyze"}
            
        # Basic metrics
        recommended_ids = [s["id"] for s in recommended_stories]
        ground_truth_ids = [s["id"] for s in ground_truth_stories] if ground_truth_stories else []
        
        # Count overlap with ground truth
        overlapping_ids = [rid for rid in recommended_ids if rid in ground_truth_ids]
        
        # Analyze tag coverage
        tag_coverage = {}
        for tag in user_tags:
            matching_stories = [s for s in recommended_stories if tag in s.get("tags", [])]
            tag_coverage[tag] = len(matching_stories)
            
        # Calculate strengths
        strengths = []
        
        # Strong tag coverage
        well_covered_tags = [tag for tag, count in tag_coverage.items() 
                            if count >= 3]  # At least 3 stories match this tag
        if well_covered_tags:
            strengths.append({
                "type": "tag_coverage",
                "description": f"Good coverage of tags: {', '.join(well_covered_tags[:3])}",
                "tags": well_covered_tags
            })
            
        # Diversity in recommendations
        all_rec_tags = [tag for s in recommended_stories for tag in s.get("tags", [])]
        unique_tags = len(set(all_rec_tags))
        
        if unique_tags >= 15:  # Arbitrary threshold for good diversity
            strengths.append({
                "type": "diversity",
                "description": f"High diversity with {unique_tags} unique tags across recommendations",
                "unique_tags": unique_tags
            })
            
        # Ground truth overlap
        if len(overlapping_ids) >= 3:  # At least 3 matches with ground truth
            strengths.append({
                "type": "ground_truth_overlap",
                "description": f"{len(overlapping_ids)} recommendations match the ground truth",
                "overlap_count": len(overlapping_ids)
            })
            
        # Calculate weaknesses
        weaknesses = []
        
        # Poor tag coverage
        poorly_covered_tags = [tag for tag, count in tag_coverage.items() 
                              if count == 0]  # No stories match this tag
        if poorly_covered_tags:
            weaknesses.append({
                "type": "tag_coverage",
                "description": f"Missing coverage of tags: {', '.join(poorly_covered_tags[:3])}",
                "tags": poorly_covered_tags
            })
            
        # Low diversity
        if unique_tags < 10 and len(recommended_stories) >= 5:  # Low diversity threshold
            weaknesses.append({
                "type": "diversity",
                "description": f"Low diversity with only {unique_tags} unique tags",
                "unique_tags": unique_tags
            })
            
        # Poor ground truth overlap
        if ground_truth_stories and len(overlapping_ids) <= 1:  # At most 1 match with ground truth
            weaknesses.append({
                "type": "ground_truth_overlap",
                "description": f"Only {len(overlapping_ids)} recommendations match the ground truth",
                "overlap_count": len(overlapping_ids)
            })
            
        # Compile analysis
        analysis = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "tag_coverage": tag_coverage,
            "ground_truth_overlap": len(overlapping_ids),
            "unique_tags": unique_tags
        }
        
        # Add detailed tag analysis for high detail level
        if detail_level == "high":
            # Analyze individual stories
            story_analysis = []
            for story in recommended_stories[:5]:  # Analyze top 5 stories
                matching_tags = [tag for tag in user_tags if tag in story.get("tags", [])]
                in_ground_truth = story["id"] in ground_truth_ids
                
                story_analysis.append({
                    "id": story["id"],
                    "title": story["title"],
                    "matching_tags": matching_tags,
                    "matching_tag_count": len(matching_tags),
                    "in_ground_truth": in_ground_truth
                })
                
            analysis["story_analysis"] = story_analysis
            
        return analysis
    
    def evaluate(self, recommendation_agent, user_profile: str, 
                tag_model: str = None, ground_truth_model: str = None, 
                recommendation_model: str = None, metric: str = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate recommendation quality for a user.
        This is a simplified interface for backwards compatibility.
        
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
        # Update model if specified
        if tag_model or ground_truth_model:
            self.model_name = tag_model or ground_truth_model
            
        # Extract tags from user profile
        extracted_tags = self._extract_preferences(user_profile)
        
        # Generate ground truth recommendations
        ground_truth_ids = self._generate_ground_truth(user_profile)
        ground_truth_stories = [self.story_dict[sid] for sid in ground_truth_ids if sid in self.story_dict]
        
        # Get recommendations from the recommendation agent
        recommended_ids = recommendation_agent.recommend(extracted_tags, user_profile, model_name=recommendation_model)
        
        # If we got a list of IDs, convert to stories
        if isinstance(recommended_ids[0], str) if recommended_ids else False:
            recommended_stories = [self.story_dict[sid] for sid in recommended_ids if sid in self.story_dict]
        else:
            # Assume we got a list of stories
            recommended_stories = recommended_ids
            recommended_ids = [s["id"] for s in recommended_stories]
        
        # Prepare input data
        input_data = {
            "user_profile": user_profile,
            "extracted_tags": extracted_tags,
            "recommended_ids": recommended_ids,
            "recommended_stories": recommended_stories,
            "metric": metric or config.METRIC
        }
        
        # Run the full perception-reasoning-action loop
        result = self.run_perception_reasoning_action_loop(input_data)
        
        # Extract score and build compatible result format
        score = result.get("score", 0.0)
        evaluation_details = {
            "extracted_tags": extracted_tags,
            "ground_truth_ids": ground_truth_ids,
            "recommended_ids": recommended_ids,
            "metric": result.get("primary_metric", metric or config.METRIC),
            "score": score
        }
        
        # Add quality analysis if available
        if "quality_analysis" in result:
            evaluation_details["quality_analysis"] = result["quality_analysis"]
            
        return score, evaluation_details
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any], stories: List[Dict[str, Any]]) -> 'AdaptiveEvaluationAgent':
        """
        Create an AdaptiveEvaluationAgent instance from serialized data.
        
        Args:
            data: Serialized agent data
            stories: List of stories
            
        Returns:
            AdaptiveEvaluationAgent instance
        """
        agent = cls(stories, model_name=data.get("model_name"))
        agent.memory = Memory.deserialize(data.get("memory", {}))
        agent.goals = data.get("goals", [])
        
        return agent 