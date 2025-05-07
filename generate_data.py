#!/usr/bin/env python
"""
Data Generation Script for the Sekai Recommendation Agent System.
Handles generating and expanding story and user profile data from seed examples.

The seed data from the Sekai Take-Home Challenge document serves as "ground truth"
examples and must be preserved as-is in the generated data.

For each ground truth user profile, a specific number of stories are generated that
specifically match their interests, ensuring equal representation across different user preferences.

This script is meant to be run independently before the optimization process.
"""
import argparse
import json
import os
import time
from typing import List, Dict, Any

import config
import utils
from utils.data_utils import load_api_keys, load_seed_data, expand_stories, generate_test_users, generate_user_targeted_stories

def setup_directories():
    """Create necessary directories for data."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    print(f"Data directory created: {config.DATA_DIR}")
    
def generate_data(story_count: int = 100, 
                 user_count: int = 5, 
                 story_model: str = None, 
                 user_model: str = None,
                 stories_per_user: int = 5,
                 regenerate: bool = False) -> Dict[str, Any]:
    """
    Generate and expand story and user profile data.
    
    The seed data from the Sekai Take-Home Challenge (5 stories and 3 user profiles)
    are treated as "ground truth" examples and are always preserved in the generated data.
    Additional stories and user profiles are generated to supplement these truth examples.
    
    For each user in the ground truth examples, an EQUAL number of stories are generated
    that match their specific interests. This ensures balanced representation across 
    different user preferences and is critical for fair optimization and evaluation.
    
    Args:
        story_count: Target number of stories to generate (including seed stories)
        user_count: TOTAL number of users to have (including 3 seed users)
        story_model: Model to use for story generation
        user_model: Model to use for user profile generation
        stories_per_user: Number of stories to generate specifically for each user
                          (EQUAL representation across users)
        regenerate: Whether to regenerate data even if it exists
    
    Returns:
        Dictionary with statistics about the generated data
    """
    start_time = time.time()
    
    # Set up API keys
    load_api_keys()
    
    # Create directories if needed
    setup_directories()
    
    # If regenerate flag is set, remove existing data files
    story_file_path = os.path.join(config.DATA_DIR, config.STORY_DATA_FILE)
    user_file_path = os.path.join(config.DATA_DIR, config.USER_PROFILES_FILE)
    
    if regenerate:
        try:
            if os.path.exists(story_file_path):
                os.remove(story_file_path)
                print(f"Removed existing story data file: {story_file_path}")
                
            if os.path.exists(user_file_path):
                os.remove(user_file_path)
                print(f"Removed existing user data file: {user_file_path}")
        except Exception as e:
            print(f"Error clearing data files: {str(e)}")
    
    print("\n🚀 Starting Data Generation")
    print("-" * 50)
    
    # Load seed data - these are the "ground truth" examples from the challenge
    seed_stories, seed_users = load_seed_data()
    print(f"📋 Loaded {len(seed_stories)} seed stories and {len(seed_users)} seed user profiles as ground truth examples")
    
    # Calculate how many additional users to generate
    additional_user_count = max(0, user_count - len(seed_users))
    print(f"👥 Will generate {additional_user_count} additional users beyond the {len(seed_users)} ground truth users")
    
    # Use specified models or defaults from config
    story_model = story_model or config.STORY_GENERATION_MODEL
    user_model = user_model or config.USER_GENERATION_MODEL
    
    # FIRST: Generate user profiles including seed users and additional ones
    print(f"\n👤 Generating User Profiles")
    print(f"  Target: {user_count} total users ({len(seed_users)} ground truth examples + {additional_user_count} additional)")
    print(f"  Model: {user_model}")
    
    # Generate test users while preserving seed users
    users_start = time.time()
    all_users = generate_test_users(seed_users, count=additional_user_count, model_name=user_model)
    users_elapsed = time.time() - users_start
    
    # Verify that seed users are preserved
    seed_user_ids = {user["id"] for user in seed_users}
    preserved_user_count = sum(1 for user in all_users if user["id"] in seed_user_ids)
    if preserved_user_count != len(seed_users):
        print(f"⚠️ Warning: Some ground truth user profiles were not preserved! Expected {len(seed_users)}, found {preserved_user_count}")
    else:
        print(f"  ✓ All {len(seed_users)} ground truth user profiles preserved")
    
    print(f"  ✓ Generated {len(all_users) - len(seed_users)} additional profiles in {users_elapsed:.1f} seconds")
    
    # SECOND: Generate stories including user-targeted stories for ALL users (not just seed users)
    print(f"\n📚 Generating Stories")
    print(f"  Target: {story_count} total stories (including {len(seed_stories)} ground truth examples)")
    print(f"  EQUAL REPRESENTATION: {stories_per_user} targeted stories for EACH of {len(all_users)} users")
    print(f"  Model: {story_model}")
    
    # Expand stories while preserving seed stories
    stories_start = time.time()
    all_stories = expand_stories(
        seed_stories, 
        target_count=story_count, 
        model_name=story_model,
        user_profiles=all_users,  # Use ALL users, not just seed users
        stories_per_user=stories_per_user
    )
    stories_elapsed = time.time() - stories_start
    
    # Verify that seed stories are preserved
    seed_ids = {story["id"] for story in seed_stories}
    preserved_seed_count = sum(1 for story in all_stories if story["id"] in seed_ids)
    if preserved_seed_count != len(seed_stories):
        print(f"⚠️ Warning: Some ground truth stories were not preserved! Expected {len(seed_stories)}, found {preserved_seed_count}")
    else:
        print(f"  ✓ All {len(seed_stories)} ground truth stories preserved")
    
    # Calculate how many additional general stories were generated
    user_targeted_count = len(all_users) * stories_per_user
    general_stories_count = len(all_stories) - len(seed_stories) - user_targeted_count
    
    print(f"  ✓ Created {user_targeted_count} user-targeted stories ({stories_per_user} per user for equal representation)")
    print(f"  ✓ Generated {general_stories_count} additional general stories")
    print(f"  ✓ Total processing time: {stories_elapsed:.1f} seconds")
    
    # Calculate statistics
    story_stats = calculate_story_statistics(all_stories)
    
    # Print some statistics
    print(f"\n📊 Data Generation Summary")
    print(f"  Total stories: {len(all_stories)} stories broken down as:")
    print(f"    - {len(seed_stories)} ground truth examples")
    print(f"    - {user_targeted_count} user-targeted stories (EXACTLY {stories_per_user} stories per user for {len(all_users)} users)")
    print(f"    - {general_stories_count} general stories")
    print(f"  Average story tags: {story_stats['avg_tags_per_story']:.1f}")
    print(f"  Most common tags: {', '.join(story_stats['top_tags'][:5])}")
    print(f"  Total user profiles: {len(all_users)} ({len(seed_users)} ground truth + {len(all_users) - len(seed_users)} generated)")
    print(f"  Total processing time: {time.time() - start_time:.1f} seconds")
    
    result = {
        "stories_count": len(all_stories),
        "seed_stories_count": len(seed_stories),
        "user_targeted_stories_count": user_targeted_count,
        "stories_per_user": stories_per_user,
        "general_stories_count": general_stories_count,
        "users_count": len(all_users),
        "seed_users_count": len(seed_users),
        "additional_users_count": len(all_users) - len(seed_users),
        "story_stats": story_stats,
        "elapsed_time": time.time() - start_time
    }
    
    print(f"\n✅ Data Generation Complete")
    print(f"  Stories saved to: {story_file_path}")
    print(f"  User profiles saved to: {user_file_path}")
    print(f"  Equal representation achieved: {stories_per_user} tailored stories per user for all {len(all_users)} users")
    
    return result

def calculate_story_statistics(stories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics about the generated stories."""
    if not stories:
        return {
            "avg_tags_per_story": 0,
            "top_tags": [],
            "unique_tags": 0
        }
    
    # Extract all tags
    all_tags = []
    for story in stories:
        all_tags.extend(story.get("tags", []))
    
    # Count tags per story
    tags_per_story = [len(story.get("tags", [])) for story in stories]
    avg_tags = sum(tags_per_story) / len(stories) if stories else 0
    
    # Get frequency of each tag
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Sort tags by frequency
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    top_tags = [tag for tag, count in sorted_tags[:10]]
    
    return {
        "avg_tags_per_story": avg_tags,
        "top_tags": top_tags,
        "unique_tags": len(tag_counts)
    }

def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(description='Generate Data for Sekai Recommendation Agent System')
    
    # Data generation options
    parser.add_argument('--story-count', type=int, default=100,
                        help='Number of stories to generate (default: 100)')
    parser.add_argument('--user-count', type=int, default=5,
                        help='TOTAL number of users including the 3 ground truth users (default: 5)')
    parser.add_argument('--story-model', type=str, default=config.STORY_GENERATION_MODEL,
                        help=f'Model for generating stories (default: {config.STORY_GENERATION_MODEL})')
    parser.add_argument('--user-model', type=str, default=config.USER_GENERATION_MODEL,
                        help=f'Model for generating user profiles (default: {config.USER_GENERATION_MODEL})')
    parser.add_argument('--stories-per-user', type=int, default=5,
                        help='Number of stories to generate specifically for each user')
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regeneration of data even if it exists')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and exit')
    
    args = parser.parse_args()
    
    # If the user wants to list available models, print them and exit
    if args.list_models:
        print(config.list_available_models())
        return
    
    # Generate data
    generate_data(
        story_count=args.story_count,
        user_count=args.user_count,
        story_model=args.story_model,
        user_model=args.user_model,
        stories_per_user=args.stories_per_user,
        regenerate=args.regenerate
    )

if __name__ == "__main__":
    main() 