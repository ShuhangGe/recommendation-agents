"""
Script to run the new agent-based adaptive recommendation system.
This demonstrates how adaptive agents can dynamically choose different strategies
based on the user's preferences and context.
"""
import argparse
import json
import time
from typing import Dict, Any, List

import config
import utils
from agents import AdaptiveRecommendationAgent, AdaptiveEvaluationAgent

def run_agent_recommendation(user_profile: str, past_interactions: List[Dict[str, Any]] = None,
                          recommendation_model: str = None, evaluation_model: str = None,
                          verbose: bool = False) -> Dict[str, Any]:
    """
    Run the agent-based recommendation process for a user.
    
    Args:
        user_profile: User profile text
        past_interactions: Optional list of past user interactions
        recommendation_model: Optional model for recommendation agent
        evaluation_model: Optional model for evaluation agent
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with results
    """
    # Setup
    utils.load_api_keys()
    
    # Load data
    seed_stories, seed_users = utils.load_seed_data()
    
    # Check if we have expanded data already
    try:
        all_stories = utils.expand_stories(seed_stories)
        all_users = utils.generate_test_users(seed_users)
    except Exception as e:
        print(f"Error loading expanded data: {str(e)}")
        print("Using seed data only.")
        all_stories = seed_stories
        all_users = seed_users
        
    # Find matching user if user_profile is an ID
    if user_profile in [u['id'] for u in all_users]:
        for user in all_users:
            if user['id'] == user_profile:
                user_profile = user['profile']
                print(f"Using profile for user {user['id']}")
                break
    
    # Initialize agents with specified models
    recommendation_agent = AdaptiveRecommendationAgent(all_stories, model_name=recommendation_model)
    evaluation_agent = AdaptiveEvaluationAgent(all_stories, model_name=evaluation_model)
    
    # Set goals for the recommendation agent
    recommendation_agent.set_goal("Recommend stories that match the user's preferences")
    recommendation_agent.set_goal("Provide diverse recommendations")
    recommendation_agent.set_goal("Explain the recommendation process")
    
    # Log start time
    start_time = time.time()
    
    if verbose:
        print("\n=== ADAPTIVE RECOMMENDATION PROCESS ===")
        print(f"User Profile: {user_profile[:100]}...")
    
    # Step 1: Extract preferences
    extracted_tags = evaluation_agent._extract_preferences(user_profile)
    
    if verbose:
        print(f"\n== Extracted User Preferences ==")
        print(f"Tags: {extracted_tags}")
    
    # Step 2: Prepare input for recommendation agent
    recommendation_input = {
        "preferences": extracted_tags,
        "profile": user_profile,
        "past_interactions": past_interactions or []
    }
    
    # Step 3: Run full perception-reasoning-action cycle
    print("\nGenerating recommendations...")
    rec_result = recommendation_agent.run_perception_reasoning_action_loop(recommendation_input)
    
    if verbose:
        # Print perceptions
        perceptions = recommendation_agent.memory.get_from_short_term("latest_perceptions", {})
        print("\n== Agent Perceptions ==")
        print(f"Explicit Preferences: {perceptions.get('explicit_preferences', [])}")
        print(f"Implied Preferences: {perceptions.get('implied_preferences', [])}")
        print(f"Preference Patterns: {perceptions.get('preference_patterns', {}).get('trend', 'unknown')}")
        
        # Print reasoning
        plan = recommendation_agent.memory.context[-2]["content"] if len(recommendation_agent.memory.context) >= 2 else {}
        print("\n== Agent Reasoning ==")
        print(f"Selected Strategy: {plan.get('strategy', 'unknown')}")
        print(f"Diversify: {plan.get('diversify', 'unknown')}")
        print(f"Explanation Mode: {plan.get('explanation_mode', 'unknown')}")
        
        # Print preference weights
        print("\n== Preference Weights ==")
        for tag, weight in plan.get("preference_weights", {}).items():
            print(f"{tag}: {weight}")
    
    # Step 4: Log the results
    recommended_stories = rec_result.get("recommended_stories", [])
    explanation = rec_result.get("explanation", "")
    strategy_used = rec_result.get("strategy_used", "unknown")
    
    # Step 5: Evaluate recommendations
    print("\nEvaluating recommendations...")
    
    # Prepare input for evaluation agent
    evaluation_input = {
        "user_profile": user_profile,
        "recommended_stories": recommended_stories,
        "extracted_tags": extracted_tags,
        "metric": "precision@10"
    }
    
    # Run evaluation
    eval_result = evaluation_agent.run_perception_reasoning_action_loop(evaluation_input)
    
    if verbose:
        # Print evaluation perceptions
        eval_perceptions = evaluation_agent.memory.get_from_short_term("latest_perceptions", {})
        print("\n== Evaluation Perceptions ==")
        print(f"Profile Complexity: {eval_perceptions.get('profile_complexity', {}).get('level', 'unknown')}")
        print(f"Recommendation Diversity: {eval_perceptions.get('recommendation_diversity', {}).get('level', 'unknown')}")
        print(f"Tag Relevance: {eval_perceptions.get('tag_relevance', {}).get('level', 'unknown')}")
        
        # Print evaluation reasoning
        eval_plan = evaluation_agent.memory.context[-2]["content"] if len(evaluation_agent.memory.context) >= 2 else {}
        print("\n== Evaluation Reasoning ==")
        print(f"Metrics Used: {', '.join(eval_plan.get('metrics_to_use', ['unknown']))}")
        print(f"Detail Level: {eval_plan.get('detail_level', 'unknown')}")
        print(f"Weighted Scoring: {eval_plan.get('use_weighted_scoring', False)}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare the final results
    scores = eval_result.get("scores", {})
    quality_analysis = eval_result.get("quality_analysis", {})
    
    results = {
        "recommendations": [{
            "id": story["id"],
            "title": story["title"],
            "tags": story["tags"]
        } for story in recommended_stories],
        "explanation": explanation,
        "strategy": strategy_used,
        "scores": scores,
        "primary_score": eval_result.get("score", 0.0),
        "strengths": quality_analysis.get("strengths", []),
        "weaknesses": quality_analysis.get("weaknesses", []),
        "elapsed_time": elapsed_time
    }
    
    # Print recommendations
    print("\n== Recommended Stories ==")
    for i, story in enumerate(recommended_stories[:5], 1):
        print(f"{i}. {story['title']} (ID: {story['id']})")
        
    # Print explanation
    print(f"\n== Explanation ==\n{explanation}")
    
    # Print scores
    print("\n== Evaluation Scores ==")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    
    # Print strengths and weaknesses
    if quality_analysis.get("strengths"):
        print("\n== Recommendation Strengths ==")
        for strength in quality_analysis.get("strengths", [])[:3]:
            print(f"- {strength.get('description', '')}")
            
    if quality_analysis.get("weaknesses"):
        print("\n== Recommendation Weaknesses ==")
        for weakness in quality_analysis.get("weaknesses", [])[:3]:
            print(f"- {weakness.get('description', '')}")
    
    print(f"\nProcess completed in {elapsed_time:.2f} seconds")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Agent-Based Recommendation System')
    
    # User options
    parser.add_argument('--user-id', type=str, help='User ID to test (e.g., user1)')
    parser.add_argument('--profile', type=str, help='User profile text (alternative to user-id)')
    
    # Model selection
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--recommendation-model', type=str, default=config.RECOMMENDATION_MODEL,
                        help=f'Model for recommendations (default: {config.RECOMMENDATION_MODEL})')
    parser.add_argument('--evaluation-model', type=str, default=config.EVALUATION_MODEL,
                        help=f'Model for evaluations (default: {config.EVALUATION_MODEL})')
    
    # Output options
    parser.add_argument('--json-output', action='store_true',
                        help='Output results in JSON format')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information about the process')
    
    args = parser.parse_args()
    
    # If the user wants to list available models, print them and exit
    if args.list_models:
        print(config.list_available_models())
        return
    
    # Get user profile
    user_profile = None
    
    if args.user_id:
        user_profile = args.user_id  # Will look up the profile in the function
    elif args.profile:
        user_profile = args.profile
    else:
        # Use the default user profile
        user_profile = "user1"
        print(f"Using default user: {user_profile}")
    
    # Run the agent recommendation process
    results = run_agent_recommendation(
        user_profile=user_profile,
        recommendation_model=args.recommendation_model,
        evaluation_model=args.evaluation_model,
        verbose=args.verbose
    )
    
    # Output in JSON format if requested
    if args.json_output:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 