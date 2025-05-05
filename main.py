"""
Main module for the Sekai Recommendation Agent system.
Orchestrates the agents in an optimization loop.
"""
import argparse
import csv
import json
import os
import time
from typing import List, Dict, Any

import config
import utils
from agents import RecommendationAgent, EvaluationAgent, PromptOptimizerAgent

def setup_directories():
    """Create necessary directories."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

def save_results(results: List[Dict[str, Any]]):
    """
    Save optimization results to CSV.
    
    Args:
        results: List of optimization results
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(config.RESULTS_DIR, config.RESULTS_FILE)
    
    # Write to CSV
    with open(results_file, 'w', newline='') as f:
        fieldnames = ['iteration', 'score', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {results_file}")
    
    # Save the latest prompt and score
    if results:
        latest_prompt_file = os.path.join(config.RESULTS_DIR, "latest_prompt.txt")
        with open(latest_prompt_file, 'w') as f:
            f.write(results[-1].get('prompt', ''))

def should_stop_optimization(results: List[Dict[str, Any]]) -> bool:
    """
    Determine if optimization should stop.
    
    Args:
        results: List of optimization results
        
    Returns:
        True if optimization should stop, False otherwise
    """
    # If we don't have enough results, continue
    if len(results) < 3:
        return False
    
    # Check if we've reached the maximum number of iterations
    if len(results) >= config.MAX_ITERATIONS:
        print(f"Reached maximum iterations ({config.MAX_ITERATIONS}), stopping")
        return True
    
    # Check if we've reached the score threshold
    latest_score = results[-1]['score']
    if latest_score >= config.SCORE_THRESHOLD:
        print(f"Reached score threshold ({config.SCORE_THRESHOLD}), stopping")
        return True
    
    # Check if the score has plateaued
    recent_scores = [r['score'] for r in results[-3:]]
    improvements = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
    
    if all(imp < config.IMPROVEMENT_THRESHOLD for imp in improvements):
        print(f"Score has plateaued (improvements: {improvements}), stopping")
        return True
    
    return False

def run_optimization_loop(args):
    """
    Run the optimization loop with all agents.
    
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
    recommendation_agent = RecommendationAgent(all_stories, model_name=args.recommendation_model)
    evaluation_agent = EvaluationAgent(all_stories, model_name=args.evaluation_model)
    prompt_optimizer = PromptOptimizerAgent(model_name=args.optimizer_model)
    
    # Optimization loop
    results = []
    current_prompt = recommendation_agent.current_prompt
    
    print("Starting optimization loop...")
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Step 1: Evaluate current prompt on all users
        evaluation_results = []
        total_score = 0.0
        
        for user in all_users:
            print(f"\nEvaluating for user {user['id']}...")
            score, details = evaluation_agent.evaluate(
                recommendation_agent, 
                user['profile'],
                tag_model=args.evaluation_model,
                ground_truth_model=args.evaluation_model,
                recommendation_model=args.recommendation_model
            )
            print(f"Score: {score:.4f}")
            
            evaluation_results.append(details)
            total_score += score
        
        avg_score = total_score / len(all_users)
        print(f"\nAverage score: {avg_score:.4f}")
        
        # Save results
        result = {
            'iteration': iteration,
            'score': avg_score,
            'timestamp': time.time(),
            'prompt': current_prompt
        }
        results.append(result)
        
        # Step 2: Check if we should stop
        if should_stop_optimization(results):
            break
        
        # Step 3: Optimize prompt
        print("\nOptimizing prompt...")
        optimized_prompt = prompt_optimizer.optimize(
            current_prompt, 
            evaluation_results, 
            all_stories,
            model_name=args.optimizer_model
        )
        
        # Step 4: Update recommendation agent with new prompt
        recommendation_agent.update_prompt(optimized_prompt)
        current_prompt = optimized_prompt
        
        print(f"Prompt updated (length: {len(current_prompt)} chars)")
    
    # Save final results
    save_results(results)
    print("\nOptimization complete!")
    
    # Print final stats
    best_result = max(results, key=lambda r: r['score'])
    print(f"\nBest result: Iteration {best_result['iteration']}, Score: {best_result['score']:.4f}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sekai Recommendation Agent System')
    
    # Basic configuration
    parser.add_argument('--max-iterations', type=int, default=config.MAX_ITERATIONS, 
                        help=f'Maximum number of optimization iterations (default: {config.MAX_ITERATIONS})')
    parser.add_argument('--score-threshold', type=float, default=config.SCORE_THRESHOLD,
                        help=f'Score threshold for stopping optimization (default: {config.SCORE_THRESHOLD})')
    
    # Model selection
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--recommendation-model', type=str, default=config.RECOMMENDATION_MODEL,
                        help=f'Model for story recommendations (default: {config.RECOMMENDATION_MODEL})')
    parser.add_argument('--evaluation-model', type=str, default=config.EVALUATION_MODEL,
                        help=f'Model for evaluations (default: {config.EVALUATION_MODEL})')
    parser.add_argument('--optimizer-model', type=str, default=config.OPTIMIZER_MODEL,
                        help=f'Model for prompt optimization (default: {config.OPTIMIZER_MODEL})')
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
    
    args = parser.parse_args()
    
    # If the user wants to list available models, print them and exit
    if args.list_models:
        print(config.list_available_models())
        return
    
    # Override config if command-line arguments are provided
    config.MAX_ITERATIONS = args.max_iterations
    config.SCORE_THRESHOLD = args.score_threshold
    
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
    
    # Run the optimization loop with parsed arguments
    run_optimization_loop(args)

if __name__ == "__main__":
    main() 