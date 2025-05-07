"""
Script to run the new agent-based adaptive recommendation system.
This demonstrates how adaptive agents can dynamically choose different strategies
based on the user's preferences and context.
"""
import argparse
import json
import time
import os
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
        user_profile: User profile text or "all" to process all users
        past_interactions: Optional list of past user interactions
        recommendation_model: Optional model for recommendation agent
        evaluation_model: Optional model for evaluation agent
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with results
    """
    # Setup
    utils.load_api_keys()
    
    # Check if data files exist
    if not os.path.exists(config.STORY_DATA_FILE):
        raise FileNotFoundError(f"Story data file not found: {config.STORY_DATA_FILE}. Please run ./generate_data.sh first.")
    
    if not os.path.exists(config.USER_PROFILES_FILE):
        raise FileNotFoundError(f"User profiles file not found: {config.USER_PROFILES_FILE}. Please run ./generate_data.sh first.")
    
    # Load data from pre-generated files
    print(f"🔄 Loading data from {config.STORY_DATA_FILE} and {config.USER_PROFILES_FILE}...")
    all_stories = utils.load_stories()
    all_users = utils.load_user_profiles()
    
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
    else:
        print("\n🚀 ADAPTIVE RECOMMENDATION PROCESS")
        print("-" * 50)
    
    print(f"📚 Loaded {len(all_stories)} stories")
    print(f"👥 Loaded {len(all_users)} user profiles")
    print(f"👥 Recommendation model: {recommendation_model}")
    print(f"⚖️ Evaluation model: {evaluation_model}")
    
    # Process all users if requested
    if user_profile == "all":
        # Process all users and return aggregated results
        all_results = []
        total_score = 0.0
        
        print(f"\n👥 Processing all {len(all_users)} users...")
        
        for user in all_users:
            print(f"\n👤 User {user['id']}:")
            user_result = process_single_user(
                user['profile'],
                recommendation_agent,
                evaluation_agent,
                past_interactions,
                verbose
            )
            all_results.append({
                "user_id": user['id'],
                **user_result
            })
            total_score += user_result.get("primary_score", 0.0)
        
        # Calculate average score
        avg_score = total_score / len(all_users) if all_users else 0.0
        print(f"\n📈 Average score across all users: {avg_score:.4f}")
        print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")
        
        # Return aggregated results
        return {
            "results": all_results,
            "average_score": avg_score,
            "elapsed_time": time.time() - start_time,
            "user_count": len(all_users)
        }
    else:
        # Process a single user based on the provided profile
        if verbose:
            print(f"User Profile: {user_profile[:100]}...")
        else:
            print(f"👤 Processing single user profile...")
        
        return process_single_user(
            user_profile,
            recommendation_agent,
            evaluation_agent,
            past_interactions,
            verbose
        )

def process_single_user(user_profile: str, recommendation_agent, evaluation_agent, 
                         past_interactions: List[Dict[str, Any]] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Process a single user for recommendation and evaluation.
    
    Args:
        user_profile: User profile text
        recommendation_agent: Recommendation agent instance
        evaluation_agent: Evaluation agent instance
        past_interactions: Optional list of past user interactions
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    # Step 1: Extract preferences
    extracted_tags = evaluation_agent._extract_preferences(user_profile)
    
    if verbose:
        print(f"\n== Extracted User Preferences ==")
        print(f"Tags: {extracted_tags}")
    else:
        print(f"🔍 Extracting preferences...")
    
    # Step 2: Prepare input for recommendation agent
    recommendation_input = {
        "preferences": extracted_tags,
        "profile": user_profile,
        "past_interactions": past_interactions or []
    }
    
    # Step 3: Run full perception-reasoning-action cycle
    print("📚 Generating recommendations...")
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
    recommendation_strategy = rec_result.get("strategy_used", "unknown")
    explanation = rec_result.get("explanation", "")
    
    # Print minimal recommendation info if not verbose
    if not verbose:
        print(f"✨ Using strategy: {recommendation_strategy}")
        print(f"📋 Found {len(recommended_stories)} stories")
    
    # Step 5: Evaluate recommendations
    print("⚖️ Evaluating recommendations...")
    
    # Prepare evaluation input
    evaluation_input = {
        "user_profile": user_profile,
        "recommended_stories": recommended_stories,
        "extracted_tags": extracted_tags
    }
    
    # Run evaluation agent
    eval_result = evaluation_agent.run_perception_reasoning_action_loop(evaluation_input)
    
    # Process evaluation results
    scores = eval_result.get("scores", {})
    primary_score = eval_result.get("score", 0.0)
    quality_analysis = eval_result.get("quality_analysis", {})
    elapsed_time = time.time() - start_time
    
    # Prepare results
    results = {
        "recommended_stories": recommended_stories,
        "recommendation_strategy": recommendation_strategy,
        "explanation": explanation,
        "scores": scores,
        "primary_score": primary_score,
        "quality_analysis": quality_analysis,
        "elapsed_time": elapsed_time
    }
    
    # Print stories and scores
    print(f"\n📊 Evaluation Score: {primary_score:.4f}")
    
    # Print detailed info if verbose
    if verbose:
        # Print recommended stories
        print("\n== Recommended Stories ==")
        for i, story in enumerate(recommended_stories[:5], 1):
            print(f"{i}. {story['title']} (ID: {story['id']})")
            print(f"   - {story['summary'][:100]}...")
        
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
    else:
        # Print brief strengths/weaknesses
        if quality_analysis.get("strengths"):
            print("✅ Top strength: " + quality_analysis.get("strengths", [])[0].get('description', ''))
            
        if quality_analysis.get("weaknesses"):
            print("⚠️ Top weakness: " + quality_analysis.get("weaknesses", [])[0].get('description', ''))
    
    print(f"⏱️ Process completed in {elapsed_time:.2f} seconds")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Agent-Based Recommendation System')
    
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
    
    # Always use all users
    user_profile = "all"
    print("Using all available users")
    
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