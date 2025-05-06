"""
Main module for the Sekai Recommendation Agent system.
Uses the agent-based architecture for intelligent recommendations.
"""
import argparse
import json
import os
import time
from typing import List, Dict, Any

import config
import utils
from agents import AdaptiveRecommendationAgent, AdaptiveEvaluationAgent

def setup_directories():
    """Create necessary directories."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

def run_agent_system(args):
    """
    Run the agent-based recommendation system.
    
    Args:
        args: Command-line arguments
    """
    # Setup
    setup_directories()
    utils.load_api_keys()
    
    # Load and expand data
    seed_stories, seed_users = utils.load_seed_data()
    
    # Expand stories with specified model
    all_stories = utils.expand_stories(
        seed_stories, 
        target_count=args.story_count,
        model_name=args.story_model
    )
    
    # Generate test users with specified model
    all_users = utils.generate_test_users(
        seed_users, 
        count=args.user_count,
        model_name=args.user_model
    )
    
    print(f"Loaded {len(all_stories)} stories and {len(all_users)} users")
    
    # Initialize agents with specified models
    recommendation_agent = AdaptiveRecommendationAgent(all_stories, model_name=args.recommendation_model)
    evaluation_agent = AdaptiveEvaluationAgent(all_stories, model_name=args.evaluation_model)
    
    # Set goals for agents
    recommendation_agent.set_goal("Provide highly relevant story recommendations to users")
    recommendation_agent.set_goal("Adapt recommendation strategy based on user preferences")
    recommendation_agent.set_goal("Explain the recommendation process")
    
    evaluation_agent.set_goal("Accurately assess recommendation quality")
    evaluation_agent.set_goal("Identify strengths and weaknesses in recommendations")
    evaluation_agent.set_goal("Provide actionable feedback for improvement")
    
    # Run tests for each user
    results = []
    total_score = 0.0
    
    for user in all_users:
        print(f"\nProcessing user {user['id']}...")
        user_profile = user['profile']
        
        # Step 1: Extract preferences
        extracted_tags = evaluation_agent._extract_preferences(user_profile)
        print(f"Extracted tags: {extracted_tags}")
        
        # Step 2: Prepare input for recommendation agent
        recommendation_input = {
            "preferences": extracted_tags,
            "profile": user_profile,
            "past_interactions": []
        }
        
        # Step 3: Run recommendation cycle
        print("\nGenerating recommendations...")
        start_time = time.time()
        rec_result = recommendation_agent.run_perception_reasoning_action_loop(recommendation_input)
        recommended_stories = rec_result.get("recommended_stories", [])
        strategy_used = rec_result.get("strategy_used", "unknown")
        print(f"Used strategy: {strategy_used}")
        
        # Step 4: Evaluate recommendations
        print("\nEvaluating recommendations...")
        evaluation_input = {
            "user_profile": user_profile,
            "recommended_stories": recommended_stories,
            "extracted_tags": extracted_tags,
            "metric": args.metric
        }
        
        eval_result = evaluation_agent.run_perception_reasoning_action_loop(evaluation_input)
        score = eval_result.get("score", 0.0)
        total_score += score
        
        elapsed_time = time.time() - start_time
        
        # Store results
        user_result = {
            "user_id": user['id'],
            "score": score,
            "strategy": strategy_used,
            "elapsed_time": elapsed_time,
            "timestamp": time.time()
        }
        results.append(user_result)
        
        # Print recommendations
        print(f"\nScore: {score:.4f}")
        print("\nTop recommended stories:")
        for i, story in enumerate(recommended_stories[:5], 1):
            print(f"{i}. {story['title']} (ID: {story['id']})")
        
        # Check if we should print strengths and weaknesses
        if args.verbose and "quality_analysis" in eval_result:
            quality_analysis = eval_result.get("quality_analysis", {})
            
            if quality_analysis.get("strengths"):
                print("\nRecommendation Strengths:")
                for strength in quality_analysis.get("strengths", [])[:3]:
                    print(f"- {strength.get('description', '')}")
                    
            if quality_analysis.get("weaknesses"):
                print("\nRecommendation Weaknesses:")
                for weakness in quality_analysis.get("weaknesses", [])[:3]:
                    print(f"- {weakness.get('description', '')}")
    
    # Calculate and print average score
    avg_score = total_score / len(all_users)
    print(f"\nAverage score across all users: {avg_score:.4f}")
    
    # Save results
    if args.save_results:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        results_file = os.path.join(config.RESULTS_DIR, "agent_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "results": results,
                "average_score": avg_score,
                "timestamp": time.time()
            }, f, indent=2)
        print(f"Results saved to {results_file}")
    
    return avg_score, results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sekai Agent-Based Recommendation System')
    
    # Basic configuration
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to a file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    # Model selection
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--recommendation-model', type=str, default=config.RECOMMENDATION_MODEL,
                        help=f'Model for recommendations (default: {config.RECOMMENDATION_MODEL})')
    parser.add_argument('--evaluation-model', type=str, default=config.EVALUATION_MODEL,
                        help=f'Model for evaluations (default: {config.EVALUATION_MODEL})')
    parser.add_argument('--story-model', type=str, default=config.STORY_GENERATION_MODEL,
                        help=f'Model for generating stories (default: {config.STORY_GENERATION_MODEL})')
    parser.add_argument('--user-model', type=str, default=config.USER_GENERATION_MODEL,
                        help=f'Model for generating user profiles (default: {config.USER_GENERATION_MODEL})')
    
    # Data generation settings
    parser.add_argument('--story-count', type=int, default=100,
                        help='Number of stories to generate (default: 100)')
    parser.add_argument('--user-count', type=int, default=5,
                        help='Number of additional test users to generate (default: 5)')
    parser.add_argument('--regenerate-data', action='store_true',
                        help='Force regeneration of stories and users, even if they already exist')
    
    # Evaluation settings
    parser.add_argument('--metric', type=str, choices=['precision@10', 'recall', 'semantic_overlap'],
                        default=config.METRIC, help=f'Evaluation metric to use (default: {config.METRIC})')
    
    args = parser.parse_args()
    
    # If the user wants to list available models, print them and exit
    if args.list_models:
        print(config.list_available_models())
        return
    
    # If regenerate data flag is set, remove existing data files
    if args.regenerate_data:
        story_file_path = os.path.join(config.DATA_DIR, config.STORY_DATA_FILE)
        user_file_path = os.path.join(config.DATA_DIR, config.USER_PROFILES_FILE)
        
        try:
            if os.path.exists(story_file_path):
                os.remove(story_file_path)
                print(f"Removed existing story data file: {story_file_path}")
                
            if os.path.exists(user_file_path):
                os.remove(user_file_path)
                print(f"Removed existing user data file: {user_file_path}")
        except Exception as e:
            print(f"Error clearing data files: {str(e)}")
    
    # Run the agent system with parsed arguments
    run_agent_system(args)

if __name__ == "__main__":
    main() 