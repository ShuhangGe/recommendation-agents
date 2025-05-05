"""
Script to test a specific task with a specific model.
Useful for testing and comparing different models for individual tasks.
"""
import argparse
import json
import time

import config
import utils
from agents import RecommendationAgent, EvaluationAgent, PromptOptimizerAgent

def test_recommendation_task(model_name, user_profile, stories):
    """Test the recommendation task with a specific model."""
    print(f"Testing recommendation task with model: {model_name}")
    start_time = time.time()
    
    agent = RecommendationAgent(stories)
    
    # Extract tags (for simplicity, using the evaluation agent)
    eval_agent = EvaluationAgent(stories)
    tags = eval_agent.extract_tags(user_profile)
    
    # Generate recommendations with specified model
    recommendations = agent.recommend(tags, model_name=model_name)
    
    end_time = time.time()
    print(f"Task completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    recommended_stories = agent.get_stories_by_ids(recommendations)
    for i, story in enumerate(recommended_stories, 1):
        print(f"{i}. {story['title']} (ID: {story['id']})")
    
    return recommendations

def test_evaluation_task(model_name, user_profile, stories):
    """Test the evaluation task with a specific model."""
    print(f"Testing evaluation (ground truth generation) task with model: {model_name}")
    start_time = time.time()
    
    agent = EvaluationAgent(stories)
    
    # Generate ground truth recommendations with specified model
    ground_truth = agent.generate_ground_truth(user_profile, model_name=model_name)
    
    end_time = time.time()
    print(f"Task completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    story_dict = {s["id"]: s for s in stories}
    ground_truth_stories = [story_dict[sid] for sid in ground_truth if sid in story_dict]
    for i, story in enumerate(ground_truth_stories, 1):
        print(f"{i}. {story['title']} (ID: {story['id']})")
    
    return ground_truth

def test_tag_extraction_task(model_name, user_profile, stories):
    """Test the tag extraction task with a specific model."""
    print(f"Testing tag extraction task with model: {model_name}")
    start_time = time.time()
    
    agent = EvaluationAgent(stories)
    
    # Extract tags with specified model
    tags = agent.extract_tags(user_profile, model_name=model_name)
    
    end_time = time.time()
    print(f"Task completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    print(f"Extracted tags: {tags}")
    
    return tags

def test_prompt_optimization_task(model_name, current_prompt, evaluation_results, stories):
    """Test the prompt optimization task with a specific model."""
    print(f"Testing prompt optimization task with model: {model_name}")
    start_time = time.time()
    
    agent = PromptOptimizerAgent()
    
    # Optimize prompt with specified model
    optimized_prompt = agent.optimize(current_prompt, evaluation_results, stories, model_name=model_name)
    
    end_time = time.time()
    print(f"Task completed in {end_time - start_time:.2f} seconds")
    
    # Print results
    print(f"Original prompt length: {len(current_prompt)} chars")
    print(f"Optimized prompt length: {len(optimized_prompt)} chars")
    print(f"First 100 chars of optimized prompt: {optimized_prompt[:100]}...")
    
    return optimized_prompt

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test a specific task with a specific model')
    
    # Basic options
    parser.add_argument('--task', type=str, required=True, 
                        choices=['recommendation', 'evaluation', 'tags', 'optimization'],
                        help='Task to test (recommendation, evaluation, tags, optimization)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use for the task')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    
    # User profile options
    parser.add_argument('--user-id', type=str, default="user1",
                        help='User ID to test with')
    parser.add_argument('--profile', type=str,
                        help='User profile text (alternative to user-id)')
    
    # Output options
    parser.add_argument('--json-output', action='store_true',
                        help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # If the user wants to list available models, print them and exit
    if args.list_models:
        print(config.list_available_models())
        return
    
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
    
    # Get user profile
    user_profile = None
    user_id = None
    
    if args.profile:
        user_profile = args.profile
        user_id = "custom_profile"
    else:
        # Find user by ID
        for user in all_users:
            if user['id'] == args.user_id:
                user_profile = user['profile']
                user_id = user['id']
                break
        
        if not user_profile:
            print(f"User with ID '{args.user_id}' not found. Using first user.")
            user_profile = all_users[0]['profile']
            user_id = all_users[0]['id']
    
    print(f"Using user: {user_id}")
    
    # Run the specified task
    result = None
    
    if args.task == 'recommendation':
        result = test_recommendation_task(args.model, user_profile, all_stories)
    elif args.task == 'evaluation':
        result = test_evaluation_task(args.model, user_profile, all_stories)
    elif args.task == 'tags':
        result = test_tag_extraction_task(args.model, user_profile, all_stories)
    elif args.task == 'optimization':
        # For optimization, we need to generate some evaluation results first
        print("Generating evaluation results for optimization task...")
        rec_agent = RecommendationAgent(all_stories)
        eval_agent = EvaluationAgent(all_stories)
        
        evaluation_results = []
        for user in all_users[:2]:  # Use just 2 users for quick testing
            score, details = eval_agent.evaluate(rec_agent, user['profile'])
            evaluation_results.append(details)
        
        current_prompt = rec_agent.current_prompt
        result = test_prompt_optimization_task(args.model, current_prompt, evaluation_results, all_stories)
    
    # Output results in JSON format if requested
    if args.json_output and result:
        print(json.dumps({
            "user_id": user_id,
            "task": args.task,
            "model": args.model,
            "result": result
        }, indent=2))

if __name__ == "__main__":
    main() 