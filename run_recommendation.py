"""
Script to run a single recommendation for a user profile.
Useful for testing the recommendation agent.
"""
import argparse
import json

import config
import utils
from agents import RecommendationAgent, EvaluationAgent

def main():
    """Run a recommendation test for a user profile."""
    parser = argparse.ArgumentParser(description='Run a recommendation test')
    
    # User options
    parser.add_argument('--user-id', type=str, help='User ID to test (e.g., user1)')
    parser.add_argument('--profile', type=str, help='User profile text (alternative to user-id)')
    
    # Model selection
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--recommendation-model', type=str, default=config.RECOMMENDATION_MODEL,
                        help=f'Model for story recommendations (default: {config.RECOMMENDATION_MODEL})')
    parser.add_argument('--evaluation-model', type=str, default=config.EVALUATION_MODEL,
                        help=f'Model for evaluations (default: {config.EVALUATION_MODEL})')
    
    # Output options
    parser.add_argument('--metric', type=str, choices=['precision@10', 'recall', 'semantic_overlap'],
                        default=config.METRIC, help=f'Evaluation metric to use (default: {config.METRIC})')
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
    
    # Initialize agents with specified models
    print(f"Using recommendation model: {args.recommendation_model}")
    print(f"Using evaluation model: {args.evaluation_model}")
    
    recommendation_agent = RecommendationAgent(all_stories, model_name=args.recommendation_model)
    evaluation_agent = EvaluationAgent(all_stories, model_name=args.evaluation_model)
    
    # Get user profile
    user_profile = None
    user_id = None
    
    if args.user_id:
        # Find user by ID
        for user in all_users:
            if user['id'] == args.user_id:
                user_profile = user['profile']
                user_id = user['id']
                break
        
        if not user_profile:
            print(f"User with ID '{args.user_id}' not found.")
            return
    elif args.profile:
        user_profile = args.profile
        user_id = "custom_profile"
    else:
        # Use the first user profile as default
        user_profile = all_users[0]['profile']
        user_id = all_users[0]['id']
        print(f"Using default user: {user_id}")
    
    # Extract tags from the profile
    print("\nExtracting tags from user profile...")
    tags = evaluation_agent.extract_tags(user_profile, model_name=args.evaluation_model)
    print(f"Extracted tags: {tags}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommended_ids = recommendation_agent.recommend(tags, model_name=args.recommendation_model)
    recommended_stories = recommendation_agent.get_stories_by_ids(recommended_ids)
    
    # Generate ground truth
    print("\nGenerating ground truth recommendations...")
    ground_truth_ids = evaluation_agent.generate_ground_truth(user_profile, model_name=args.evaluation_model)
    story_dict = {s['id']: s for s in all_stories}
    ground_truth_stories = [story_dict[sid] for sid in ground_truth_ids if sid in story_dict]
    
    # Calculate score
    score = utils.calculate_metric(recommended_stories, ground_truth_stories, args.metric)
    print(f"\nScore: {score:.4f} using {args.metric}")
    
    if args.json_output:
        # Output in JSON format
        result = {
            "user_id": user_id,
            "tags": tags,
            "metric": args.metric,
            "score": score,
            "recommended_stories": [{
                "id": story["id"],
                "title": story["title"],
                "tags": story["tags"]
            } for story in recommended_stories],
            "ground_truth_stories": [{
                "id": story["id"],
                "title": story["title"],
                "tags": story["tags"]
            } for story in ground_truth_stories]
        }
        print(json.dumps(result, indent=2))
    else:
        # Print recommendations
        print("\nRecommended stories:")
        for i, story in enumerate(recommended_stories, 1):
            print(f"{i}. {story['title']} (ID: {story['id']})")
            print(f"   Intro: {story['intro']}")
            print(f"   Tags: {', '.join(story['tags'])}")
            print()
        
        # Print ground truth
        print("\nGround truth stories:")
        for i, story in enumerate(ground_truth_stories, 1):
            print(f"{i}. {story['title']} (ID: {story['id']})")
            print(f"   Tags: {', '.join(story['tags'])}")
            print()

if __name__ == "__main__":
    main() 