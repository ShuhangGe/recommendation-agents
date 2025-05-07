"""
Adaptive Recommendation Agent that uses dynamic reasoning to suggest stories.
This agent can adapt its strategy based on user preferences and feedback.
"""
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from utils.data_utils import generate_with_model, create_openai_client
from agents.agent_core import Agent, Memory

class AdaptiveRecommendationAgent(Agent):
    """
    Adaptive agent that recommends stories to users based on their preferences.
    This agent can:
    1. Dynamically analyze user preferences
    2. Choose different recommendation strategies based on context
    3. Learn from feedback to improve recommendations
    4. Explain its reasoning process
    """
    
    def __init__(self, stories: List[Dict[str, Any]], model_name: str = None):
        """
        Initialize the adaptive recommendation agent.
        
        Args:
            stories: List of all available stories
            model_name: Name of the model to use (defaults to config setting)
        """
        super().__init__(name="AdaptiveRecommender", model_name=model_name or config.RECOMMENDATION_MODEL)
        self.stories = stories
        self.story_dict = {s["id"]: s for s in stories}
        
        # Register available tools
        self.register_tool("similarity_search", self._similarity_search)
        self.register_tool("tag_based_search", self._tag_based_search)
        self.register_tool("hybrid_search", self._hybrid_search)
        self.register_tool("diversity_boost", self._diversity_boost)
        
        # Save the default prompt for recommendations
        self.default_prompt = self._create_default_prompt()
        self.current_prompt = self.default_prompt
        
        # Set initial goals
        self.set_goal("Provide highly relevant story recommendations to users")
        self.set_goal("Adapt recommendation strategy based on user preferences")
        self.set_goal("Explain recommendation reasoning when needed")
    
    def _create_default_prompt(self) -> str:
        """Create the default recommendation prompt."""
        return f"""
        You are an adaptive recommendation agent for Sekai, a platform for interactive stories. 
        Your task is to recommend the most relevant stories for a user based on their preferences.
        
        Consider the user's explicit preferences but also try to infer implicit preferences.
        Balance relevance with diversity in your recommendations.
        """
    
    def get_current_prompt(self) -> str:
        """
        Get the current recommendation prompt.
        
        Returns:
            Current prompt text
        """
        return self.current_prompt
    
    def update_prompt(self, new_prompt: str):
        """
        Update the recommendation prompt.
        
        Args:
            new_prompt: New prompt text
        """
        if new_prompt and len(new_prompt) > 20:  # Basic validation
            self.current_prompt = new_prompt
            self.logger.info("Updated recommendation prompt")
        else:
            self.logger.warning("Invalid prompt update ignored")
    
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user preferences to understand what they're looking for.
        
        Args:
            input_data: Dictionary containing user preferences and context
            
        Returns:
            Dictionary of perceptions
        """
        self.logger.info("Perceiving user preferences")
        
        # Extract user preferences
        user_preferences = input_data.get("preferences", [])
        user_profile = input_data.get("profile", "")
        past_interactions = input_data.get("past_interactions", [])
        
        # Analyze user preferences
        perceptions = {
            "explicit_preferences": user_preferences,
            "implied_preferences": self._infer_implicit_preferences(user_profile),
            "preference_patterns": self._analyze_preference_patterns(past_interactions),
            "preference_strength": {},
            "apparent_interests": []
        }
        
        # Determine preference strength for each tag
        for tag in user_preferences:
            # Check how many times this preference appears in past interactions
            occurrences = sum(1 for interaction in past_interactions if tag in interaction.get("tags", []))
            perceptions["preference_strength"][tag] = min(1.0, 0.3 + (occurrences * 0.1))
        
        # Identify apparent key interests from profile
        if user_profile:
            interests = self._extract_key_interests(user_profile)
            perceptions["apparent_interests"] = interests
        
        # Store perceptions in memory
        self.memory.add_to_short_term("latest_perceptions", perceptions)
        
        self.logger.info(f"Perceived {len(user_preferences)} explicit preferences")
        return perceptions
    
    def _infer_implicit_preferences(self, user_profile: str) -> List[str]:
        """Infer implicit preferences from user profile."""
        if not user_profile or len(user_profile) < 10:
            return []
            
        # Use LLM to extract implicit preferences
        prompt = f"""
        Based on this user profile, what are 5-7 implied preferences or interests that aren't 
        explicitly stated but can be reasonably inferred?
        
        User profile:
        {user_profile}
        
        Return ONLY a valid JSON list, like: ["action", "romance", "underdog"]
        """
        
        try:
            response = generate_with_model(
                prompt=prompt,
                model_name=self.model_name,
                system_message="You are an expert at understanding user psychology and preferences.",
                temperature=0.7
            )
            
            # Extract JSON list
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Error inferring implicit preferences: {str(e)}")
        
        return []
    
    def _analyze_preference_patterns(self, past_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in past interactions."""
        if not past_interactions:
            return {"trend": "insufficient_data"}
            
        # Simple analysis of recent interactions
        genres = []
        themes = []
        franchises = []
        
        for interaction in past_interactions[-5:]:  # Look at 5 most recent
            tags = interaction.get("tags", [])
            
            # Very simplified categorization
            for tag in tags:
                if tag in ["action", "romance", "comedy", "drama", "horror"]:
                    genres.append(tag)
                elif tag in ["underdog", "rivalry", "betrayal", "redemption"]:
                    themes.append(tag)
                elif tag in ["naruto", "dragon ball", "my hero academia"]:
                    franchises.append(tag)
        
        # Count occurrences
        from collections import Counter
        genre_counts = Counter(genres)
        theme_counts = Counter(themes)
        franchise_counts = Counter(franchises)
        
        # Get most common
        top_genres = genre_counts.most_common(2)
        top_themes = theme_counts.most_common(2)
        top_franchises = franchise_counts.most_common(2)
        
        return {
            "trend": "patterns_found" if (top_genres or top_themes or top_franchises) else "no_clear_patterns",
            "top_genres": top_genres,
            "top_themes": top_themes,
            "top_franchises": top_franchises
        }
    
    def _extract_key_interests(self, user_profile: str) -> List[str]:
        """Extract key interests from user profile."""
        # For demonstration, we'll use a simple keyword extraction
        # In a real system, this would use more sophisticated NLP
        keywords = ["action", "romance", "comedy", "drama", "adventure", 
                   "naruto", "dragon ball", "my hero academia", "underdog", 
                   "rivalry", "betrayal", "redemption"]
        
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in user_profile.lower():
                found_keywords.append(keyword)
                
        return found_keywords
    
    def reason(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best recommendation strategy based on perceptions.
        
        Args:
            perceptions: Dictionary of perceptions about user preferences
            
        Returns:
            Plan of action for generating recommendations
        """
        self.logger.info("Reasoning about recommendation strategy")
        
        # Extract perceptions
        explicit_preferences = perceptions.get("explicit_preferences", [])
        implied_preferences = perceptions.get("implied_preferences", [])
        preference_patterns = perceptions.get("preference_patterns", {})
        preference_strength = perceptions.get("preference_strength", {})
        apparent_interests = perceptions.get("apparent_interests", [])
        
        # Combine all preferences with weights
        combined_preferences = {}
        
        # Add explicit preferences with high weight
        for pref in explicit_preferences:
            combined_preferences[pref] = preference_strength.get(pref, 0.8)
        
        # Add implied preferences with lower weight
        for pref in implied_preferences:
            if pref in combined_preferences:
                combined_preferences[pref] += 0.3
            else:
                combined_preferences[pref] = 0.5
        
        # Add interests from user profile
        for interest in apparent_interests:
            if interest in combined_preferences:
                combined_preferences[interest] += 0.2
            else:
                combined_preferences[interest] = 0.4
                
        # Determine recommendation strategy
        strategy = "hybrid_search"  # Default strategy
        diversify = True
        explanation_mode = "simple"
        
        # Choose strategy based on perceptions
        if not explicit_preferences and not implied_preferences:
            # If we have no preferences, use a diverse recommendation approach
            strategy = "diversity_boost"
            explanation_mode = "exploratory"
        elif len(explicit_preferences) >= 5:
            # Many explicit preferences - use tag-based search for precision
            strategy = "tag_based_search"
            diversify = preference_patterns.get("trend") != "no_clear_patterns"
            explanation_mode = "detailed"
        elif implied_preferences and preference_patterns.get("trend") == "patterns_found":
            # Clear patterns - use similarity search for coherent recommendations
            strategy = "similarity_search"
            diversify = False
            explanation_mode = "pattern_based"
            
        # Create the recommendation plan
        plan = {
            "strategy": strategy,
            "combined_preferences": combined_preferences,
            "diversify": diversify,
            "explanation_mode": explanation_mode,
            "num_recommendations": 10,
            "preference_weights": {
                k: round(v, 2) for k, v in combined_preferences.items()
            }
        }
        
        self.logger.info(f"Selected strategy: {strategy}")
        return plan
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on the selected strategy.
        
        Args:
            plan: Plan of action for generating recommendations
            
        Returns:
            Dictionary with recommended stories and explanation
        """
        self.logger.info(f"Generating recommendations using {plan['strategy']} strategy")
        
        # Extract plan details
        strategy = plan.get("strategy", "hybrid_search")
        combined_preferences = plan.get("combined_preferences", {})
        diversify = plan.get("diversify", True)
        explanation_mode = plan.get("explanation_mode", "simple")
        num_recommendations = plan.get("num_recommendations", 10)
        
        # Use the selected strategy to generate recommendations
        if strategy in self.tools:
            recommendations = self.tools[strategy](combined_preferences, num_recommendations)
        else:
            # Fallback to hybrid search
            self.logger.warning(f"Strategy {strategy} not found, using hybrid_search")
            recommendations = self.tools["hybrid_search"](combined_preferences, num_recommendations)
        
        # Apply diversity boost if needed
        if diversify and "diversity_boost" in self.tools:
            recommendations = self.tools["diversity_boost"](recommendations, combined_preferences)
        
        # For more personalization, we can refine recommendations using the current prompt
        if hasattr(self, 'current_prompt') and self.current_prompt:
            refined_recommendations = self._generate_recommendations_with_prompt(
                recommendations, 
                combined_preferences,
                strategy
            )
            # If refinement successful, use the refined recommendations
            if refined_recommendations:
                recommendations = refined_recommendations
        
        # Generate explanation for recommendations
        explanation = self._generate_explanation(
            recommendations, 
            combined_preferences, 
            strategy, 
            explanation_mode
        )
        
        # Return recommendations and explanation
        result = {
            "recommended_ids": [s["id"] for s in recommendations],
            "recommended_stories": recommendations,
            "explanation": explanation,
            "strategy_used": strategy,
            "timestamp": self.memory.last_updated
        }
        
        # Store the result in memory
        self.memory.add_to_short_term("latest_recommendations", result)
        
        self.logger.info(f"Generated {len(recommendations)} recommendations")
        return result
    
    def _generate_recommendations_with_prompt(self, initial_recommendations: List[Dict[str, Any]], 
                                             preferences: Dict[str, float], 
                                             strategy: str) -> List[Dict[str, Any]]:
        """
        Refine recommendations using the current prompt with LLM.
        
        Args:
            initial_recommendations: Initial recommendations from strategy
            preferences: User preference tags and weights
            strategy: Strategy used for initial recommendations
            
        Returns:
            Refined list of recommended stories
        """
        try:
            # Create a prompt with initial recommendations and user preferences
            user_prompt = f"""
            User preferences: {json.dumps(list(preferences.keys()))}
            
            Initial recommendations: 
            {json.dumps([{"id": s["id"], "title": s["title"], "tags": s["tags"]} for s in initial_recommendations[:5]], indent=2)}
            
            Using strategy: {strategy}
            """
            
            # Combine with the current optimization prompt
            full_prompt = f"{self.current_prompt}\n\n{user_prompt}\n\nReturn the IDs of the top 10 recommended stories in JSON format as a list."
            
            # Generate recommendations using LLM
            response = generate_with_model(
                prompt=full_prompt,
                model_name=self.model_name,
                system_message="You are an expert recommendation system for stories.",
                temperature=0.7
            )
            
            # Extract JSON list of IDs
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                recommended_ids = json.loads(json_str)
                
                # Convert IDs to story objects
                return self.get_stories_by_ids(recommended_ids)
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations with prompt: {str(e)}")
        
        # Return the original recommendations if anything fails
        return initial_recommendations
    
    def _similarity_search(self, preferences: Dict[str, float], num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find stories similar to the user's preferences using semantic similarity.
        
        Args:
            preferences: Dictionary of preference tags and their weights
            num_results: Number of results to return
            
        Returns:
            List of recommended stories
        """
        if not preferences:
            return random.sample(self.stories, min(num_results, len(self.stories)))
            
        # Create a simple vector representation for preferences and stories
        # In a real system, this would use proper embeddings
        preference_tags = list(preferences.keys())
        preference_weights = list(preferences.values())
        
        # Score each story
        scored_stories = []
        for story in self.stories:
            score = 0
            story_tags = story.get("tags", [])
            
            # Calculate tag overlap with weights
            for tag, weight in zip(preference_tags, preference_weights):
                if tag in story_tags:
                    score += weight
            
            # Normalize by number of preferences
            score = score / len(preference_tags) if preference_tags else 0
            scored_stories.append((story, score))
        
        # Sort by score and take top results
        scored_stories.sort(key=lambda x: x[1], reverse=True)
        top_stories = [s[0] for s in scored_stories[:num_results]]
        
        return top_stories
    
    def _tag_based_search(self, preferences: Dict[str, float], num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find stories with the most tag matches to preferences.
        
        Args:
            preferences: Dictionary of preference tags and their weights
            num_results: Number of results to return
            
        Returns:
            List of recommended stories
        """
        if not preferences:
            return random.sample(self.stories, min(num_results, len(self.stories)))
            
        # Score each story based on tag matches
        scored_stories = []
        for story in self.stories:
            score = 0
            matches = 0
            story_tags = story.get("tags", [])
            
            for tag, weight in preferences.items():
                if tag in story_tags:
                    score += weight
                    matches += 1
            
            # Bonus for multiple matches
            if matches > 1:
                score += matches * 0.1
                
            scored_stories.append((story, score))
        
        # Sort by score and take top results
        scored_stories.sort(key=lambda x: x[1], reverse=True)
        top_stories = [s[0] for s in scored_stories[:num_results]]
        
        return top_stories
    
    def _hybrid_search(self, preferences: Dict[str, float], num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid approach combining tag matching and similarity.
        
        Args:
            preferences: Dictionary of preference tags and their weights
            num_results: Number of results to return
            
        Returns:
            List of recommended stories
        """
        if not preferences:
            return random.sample(self.stories, min(num_results, len(self.stories)))
            
        # Get results from both methods
        similarity_results = self._similarity_search(preferences, num_results * 2)
        tag_results = self._tag_based_search(preferences, num_results * 2)
        
        # Combine results with deduplication
        seen_ids = set()
        combined_results = []
        
        # Alternate between results from both approaches
        for i in range(max(len(similarity_results), len(tag_results))):
            if i < len(similarity_results) and similarity_results[i]["id"] not in seen_ids:
                combined_results.append(similarity_results[i])
                seen_ids.add(similarity_results[i]["id"])
                
            if i < len(tag_results) and tag_results[i]["id"] not in seen_ids:
                combined_results.append(tag_results[i])
                seen_ids.add(tag_results[i]["id"])
                
            if len(combined_results) >= num_results:
                break
                
        # If we don't have enough, add random stories
        remaining = num_results - len(combined_results)
        if remaining > 0:
            for story in self.stories:
                if story["id"] not in seen_ids and remaining > 0:
                    combined_results.append(story)
                    seen_ids.add(story["id"])
                    remaining -= 1
        
        return combined_results[:num_results]
    
    def _diversity_boost(self, recommendations: List[Dict[str, Any]], preferences: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Increase diversity in recommendations.
        
        Args:
            recommendations: Current list of recommendations
            preferences: Dictionary of preference tags and their weights
            
        Returns:
            More diverse list of recommendations
        """
        if len(recommendations) <= 1:
            return recommendations
            
        # Extract all tags from current recommendations
        all_tags = []
        for story in recommendations:
            all_tags.extend(story.get("tags", []))
            
        # Count tag frequencies
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        # Identify overrepresented tags (appearing in more than half of recommendations)
        overrepresented = {tag for tag, count in tag_counts.items() 
                          if count > len(recommendations) / 2}
        
        # If no overrepresented tags, return original recommendations
        if not overrepresented:
            return recommendations
            
        # Find replacement stories that have fewer overrepresented tags
        replacements = []
        for story in self.stories:
            if story["id"] not in [s["id"] for s in recommendations]:
                # Count overrepresented tags in this story
                overlap = len(set(story.get("tags", [])) & overrepresented)
                
                # Count preference matches
                preference_matches = sum(1 for tag in story.get("tags", []) if tag in preferences)
                
                # Score based on low overlap with overrepresented tags but still matching preferences
                score = preference_matches - (overlap * 0.5)
                replacements.append((story, score))
        
        # Sort by score
        replacements.sort(key=lambda x: x[1], reverse=True)
        
        # Replace up to 30% of recommendations with diverse alternatives
        num_to_replace = max(1, int(len(recommendations) * 0.3))
        
        # Keep the top 70% of original recommendations
        keep_count = len(recommendations) - num_to_replace
        diverse_recommendations = recommendations[:keep_count]
        
        # Add diverse replacements
        for i in range(min(num_to_replace, len(replacements))):
            diverse_recommendations.append(replacements[i][0])
            
        return diverse_recommendations
    
    def _generate_explanation(self, recommendations: List[Dict[str, Any]], 
                             preferences: Dict[str, float], strategy: str,
                             mode: str = "simple") -> str:
        """
        Generate an explanation for the recommendations.
        
        Args:
            recommendations: List of recommended stories
            preferences: Dictionary of preference tags and their weights
            strategy: Strategy used for recommendations
            mode: Explanation mode (simple, detailed, pattern_based, exploratory)
            
        Returns:
            Explanation text
        """
        if not recommendations:
            return "No recommendations could be generated."
            
        # Basic explanation elements
        strategy_explanations = {
            "similarity_search": "stories that closely match your preferences",
            "tag_based_search": "stories with the most matching tags to your preferences",
            "hybrid_search": "a balanced mix of stories that match your different interests",
            "diversity_boost": "a diverse selection of stories to help you explore"
        }
        
        strategy_text = strategy_explanations.get(strategy, "stories based on your preferences")
        
        # Generate explanation based on mode
        if mode == "simple":
            return f"I recommended {strategy_text}."
            
        elif mode == "detailed":
            # Analyze matches between preferences and recommendations
            tag_matches = {}
            for pref in preferences:
                matching_stories = [s["title"] for s in recommendations 
                                  if pref in s.get("tags", [])]
                if matching_stories:
                    tag_matches[pref] = matching_stories
                    
            # Create detailed explanation
            explanation = f"I recommended {strategy_text}. Here's how your preferences matched:"
            for tag, stories in tag_matches.items():
                explanation += f"\n- '{tag}': {len(stories)} stories, including {', '.join(stories[:2])}"
                if len(stories) > 2:
                    explanation += " and others"
            
            return explanation
            
        elif mode == "pattern_based":
            # Focus on patterns in preferences
            top_tags = sorted(preferences.items(), key=lambda x: x[1], reverse=True)[:3]
            top_tag_names = [t[0] for t in top_tags]
            
            explanation = f"I noticed you have a preference for {', '.join(top_tag_names)}. "
            explanation += f"I recommended {strategy_text} with an emphasis on these themes."
            
            return explanation
            
        elif mode == "exploratory":
            # For users with few or no explicit preferences
            genres = set()
            themes = set()
            
            for story in recommendations:
                for tag in story.get("tags", []):
                    if tag in ["action", "romance", "comedy", "drama", "horror"]:
                        genres.add(tag)
                    elif tag in ["underdog", "rivalry", "betrayal", "redemption"]:
                        themes.add(tag)
            
            explanation = "Since your preferences aren't very specific, "
            explanation += f"I recommended a diverse selection covering genres like {', '.join(genres)} "
            explanation += f"and themes such as {', '.join(themes)}."
            
            return explanation
            
        else:
            return f"I recommended {strategy_text} based on your preferences."
    
    def recommend(self, user_preferences: List[str], user_profile: str = "", 
                 past_interactions: List[Dict[str, Any]] = None,
                 model_name: str = None) -> List[Dict[str, Any]]:
        """
        Generate story recommendations for a user.
        This is a simplified interface for backwards compatibility.
        
        Args:
            user_preferences: List of user preference tags
            user_profile: Optional user profile text
            past_interactions: Optional list of past user interactions
            model_name: Optional model name to use
            
        Returns:
            List of recommended stories
        """
        # Update model if specified
        if model_name:
            self.model_name = model_name
            
        # Prepare input data
        input_data = {
            "preferences": user_preferences,
            "profile": user_profile,
            "past_interactions": past_interactions or []
        }
        
        # Run the full perception-reasoning-action loop
        result = self.run_perception_reasoning_action_loop(input_data)
        
        # Return the recommended stories
        return result.get("recommended_stories", [])
    
    def get_stories_by_ids(self, story_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get story objects by their IDs.
        
        Args:
            story_ids: List of story IDs
            
        Returns:
            List of story objects
        """
        return [self.story_dict[sid] for sid in story_ids if sid in self.story_dict]
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any], stories: List[Dict[str, Any]]) -> 'AdaptiveRecommendationAgent':
        """
        Create an AdaptiveRecommendationAgent instance from serialized data.
        
        Args:
            data: Serialized agent data
            stories: List of stories
            
        Returns:
            AdaptiveRecommendationAgent instance
        """
        agent = cls(stories, model_name=data.get("model_name"))
        agent.memory = Memory.deserialize(data.get("memory", {}))
        agent.goals = data.get("goals", [])
        
        return agent 