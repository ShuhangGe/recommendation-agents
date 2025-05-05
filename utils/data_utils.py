"""
Utility functions for data handling and management.
"""
import json
import os
import random
import google.generativeai as genai
import openai
from typing import Dict, List, Any, Tuple

import config

def load_api_keys():
    """Configure API keys for generative models."""
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    # Configure OpenAI client properly for newer versions
    import openai
    
    # Initialize client with API key
    if config.OPENAI_API_KEY:
        try:
            # Create a client using our helper function
            client = create_openai_client()
            # Make the client available globally
            openai.client = client
            print("OpenAI client initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
    else:
        print("Warning: OpenAI API key not found. OpenAI models will not work.")

def load_seed_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load the seed data (sample stories and user profiles) from the challenge.
    Returns:
        Tuple of (stories, users)
    """
    # Sample Sekai Stories
    stories = [
        {
            "id": "217107",
            "title": "Stranger Who Fell From The Sky",
            "intro": "You are Devin, plummeting towards Orario with no memory of how you got here...",
            "tags": ["danmachi", "reincarnation", "heroic aspirations", "mystery origin", "teamwork", "loyalty", "protectiveness"]
        },
        {
            "id": "273613",
            "title": "Trapped Between Four Anime Legends!",
            "intro": "You're caught in a dimensional rift with four anime icons. Goku wants to spar...",
            "tags": ["crossover", "jujutsu kaisen", "dragon ball", "naruto", "isekai", "dimensional travel", "reverse harem"]
        },
        {
            "id": "235701",
            "title": "New Transfer Students vs. Class 1-A Bully",
            "intro": "You and Zeroku watch in disgust as Bakugo torments Izuku again...",
            "tags": ["my hero academia", "challenging authority", "bullying", "underdog", "disruptors"]
        },
        {
            "id": "214527",
            "title": "Zenitsu Touched Your Sister's WHAT?!",
            "intro": "Your peaceful afternoon at the Butterfly Estate shatters when Zenitsu accidentally gropes Nezuko...",
            "tags": ["demon slayer", "protective instincts", "comedic panic", "violent reactions"]
        },
        {
            "id": "263242",
            "title": "Principal's Daughter Dating Contest",
            "intro": "You are Yuji Itadori, facing off against Tanjiro and Naruto for Ochako's heart...",
            "tags": ["crossover", "romantic comedy", "forced proximity", "harem", "dating competition"]
        }
    ]
    
    # Sample User Profiles
    users = [
        {
            "id": "user1",
            "profile": "choice-driven high-agency dominant protector strategist; underdog, rivalry, team-vs-team, hero-vs-villain, internal-struggle, tournament conflicts; master-servant, royalty-commoner, captor-captive power-dynamics; high-immersion lore-expander, community-engagement; power-fantasy, moral-ambiguity; isekai escapism; romance, forbidden-love, love-triangles, found-family, reverse-harem; enemies-to-lovers, slow-burn; reincarnation, devil-powers, jujitsu-sorcerer; betrayal, loyalty, survival, redemption; Naruto, Dragon Ball, Jujutsu-Kaisen, Genshin-Impact, One-Piece, Demon-Slayer, Chainsaw-Man, Marvel/DC; crossover, anti-hero, strategy, fan-groups."
        },
        {
            "id": "user2",
            "profile": "Self-insert choice-driven narrator as reluctant/supportive guardian, disguised royalty, rookie competitor. Likes Re:Zero/Naruto/MyHeroAcademia. Prefers cafes, academies, fantasy kingdoms (Konoha, Hogwarts, Teyvat), cities. Genres: supernatural/contemporary/historical romance, fantasy, action, horror. Enjoys supernatural beings, magic/curses/quirks. Favors harem, love triangles, power imbalance, enemies-to-lovers, underdog, redemption. Emotional catalysts: forbidden desires, rival advances, legacy. Content: action, romance."
        },
        {
            "id": "user3",
            "profile": "Male roleplayer seeking immersive, choice-driven narratives; self-insert underdog, reluctant hero, dominant protector power fantasy. Prefers one-on-one romance, found-family bonds, intense angst, trauma healing. Loves supernatural—nine-tailed foxes, vampires, magic. Achievement-hunter chasing epic conclusions. Morally flexible exploration sans non-consensual, gore, character death. Co-creative, supportive, detail-rich storytelling. Leaderboard climber, protective sibling loyalty, guilt."
        }
    ]
    
    return stories, users

def create_openai_client():
    """
    Create and return an OpenAI client, handling any proxy configuration issues.
    
    Returns:
        OpenAI client instance
    """
    from openai import OpenAI
    
    if not config.OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Cannot create client.")
        
    try:
        # Try creating client with explicit parameters only
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            # Explicitly set parameters, avoiding any environment/proxy settings
            base_url="https://api.openai.com/v1",
            timeout=60.0,
            max_retries=2
        )
        return client
    except TypeError as e:
        if 'proxies' in str(e):
            print("Warning: Proxy settings causing issue with OpenAI client")
            # Create a client object directly, bypassing any proxy settings
            import httpx
            http_client = httpx.Client(
                base_url="https://api.openai.com/v1",
                timeout=60.0,
                follow_redirects=True,
            )
            client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                http_client=http_client
            )
            return client
        else:
            raise e
    
def generate_with_model(prompt: str, model_name: str, system_message: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """
    Generate content using either Google or OpenAI model based on the model name.
    
    Args:
        prompt: The prompt text
        model_name: Name of the model to use
        system_message: System message for OpenAI models
        temperature: Creativity setting (0-1)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    model_type = config.get_model_type(model_name)
    
    if model_type == config.MODEL_TYPE_GOOGLE:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    
    elif model_type == config.MODEL_TYPE_OPENAI:
        import openai
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Try to use the client attribute first (set in load_api_keys)
            if hasattr(openai, 'client') and openai.client:
                client = openai.client
            else:
                # Create a new client
                client = create_openai_client()
                
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            raise
    
    else:
        raise ValueError(f"Unknown model type for model: {model_name}")

def expand_stories(seed_stories: List[Dict[str, Any]], target_count: int = 100, model_name: str = None) -> List[Dict[str, Any]]:
    """
    Expand the seed stories to a larger set using AI.
    
    Args:
        seed_stories: The original seed stories from the challenge
        target_count: Target number of stories to generate
        model_name: Name of the model to use (defaults to config setting)
        
    Returns:
        List of expanded stories
    """
    # If we already have expanded stories saved, load them
    story_file_path = os.path.join(config.DATA_DIR, config.STORY_DATA_FILE)
    if os.path.exists(story_file_path):
        with open(story_file_path, 'r') as f:
            return json.load(f)
    
    # Use the specified model or default from config
    model_name = model_name or config.STORY_GENERATION_MODEL
    print(f"Using model '{model_name}' for story generation")
    
    all_stories = seed_stories.copy()
    stories_to_generate = target_count - len(seed_stories)
    
    # Generate new stories
    for i in range(stories_to_generate):
        try:
            # Create a prompt for generating a new story
            prompt = f"""
            Based on these example stories from Sekai:
            
            {json.dumps(seed_stories, indent=2)}
            
            Generate a NEW and ORIGINAL story entry with the following format:
            - Unique ID (6 digits, different from existing IDs)
            - Creative title
            - Engaging intro that hooks readers
            - Relevant tags (5-7 tags)
            
            Make it similar in style but with different content, characters, and themes.
            Return ONLY the JSON format with fields: id, title, intro, tags (as list).
            """
            
            # Generate content using the selected model
            system_message = "You are a creative writer for interactive stories focused on anime, fantasy, and fan fiction."
            story_text = generate_with_model(prompt, model_name, system_message, temperature=0.8)
            
            # Extract the JSON part
            try:
                if '```json' in story_text:
                    json_str = story_text.split('```json')[1].split('```')[0].strip()
                elif '```' in story_text:
                    json_str = story_text.split('```')[1].strip()
                else:
                    json_str = story_text
                
                new_story = json.loads(json_str)
                
                # Ensure required fields are present
                if all(k in new_story for k in ["id", "title", "intro", "tags"]):
                    all_stories.append(new_story)
                    print(f"Generated story {i+1}/{stories_to_generate}: {new_story['title']}")
                else:
                    print(f"Generated story {i+1} missing required fields, skipping.")
            except json.JSONDecodeError:
                print(f"Generated story {i+1} has invalid JSON format, skipping.")
                
        except Exception as e:
            print(f"Error generating story {i+1}: {str(e)}")
    
    # Save the expanded stories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(story_file_path, 'w') as f:
        json.dump(all_stories, f, indent=2)
    
    return all_stories

def generate_test_users(seed_users: List[Dict[str, Any]], count: int = 5, model_name: str = None) -> List[Dict[str, Any]]:
    """
    Generate additional test users based on seed profiles.
    
    Args:
        seed_users: Original user profiles from the challenge
        count: Number of test users to generate
        model_name: Name of the model to use (defaults to config setting)
        
    Returns:
        List of user profiles including original and new ones
    """
    # If we already have expanded users saved, load them
    user_file_path = os.path.join(config.DATA_DIR, config.USER_PROFILES_FILE)
    if os.path.exists(user_file_path):
        with open(user_file_path, 'r') as f:
            return json.load(f)
    
    # Use the specified model or default from config
    model_name = model_name or config.USER_GENERATION_MODEL
    print(f"Using model '{model_name}' for user profile generation")
    
    all_users = seed_users.copy()
    
    # Generate additional users
    for i in range(count):
        try:
            # Create a prompt for generating a new user profile
            prompt = f"""
            Based on these example user profiles from Sekai:
            
            {json.dumps(seed_users, indent=2)}
            
            Generate a NEW and UNIQUE user profile with the following attributes:
            - User preferences for story types
            - Preferred character roles and dynamics
            - Content interests (themes, genres, franchises)
            - Relationship dynamics preferences
            
            Make it realistic and distinct from the example profiles.
            Return ONLY the JSON format with fields: id (like "user{len(all_users)+1}"), profile (as a single text string).
            """
            
            # Generate content using the selected model
            system_message = "You are an expert in user profiling for interactive fiction."
            user_text = generate_with_model(prompt, model_name, system_message, temperature=0.7, max_tokens=400)
            
            # Extract the JSON part
            try:
                if '```json' in user_text:
                    json_str = user_text.split('```json')[1].split('```')[0].strip()
                elif '```' in user_text:
                    json_str = user_text.split('```')[1].strip()
                else:
                    json_str = user_text
                
                new_user = json.loads(json_str)
                
                # Ensure required fields are present
                if all(k in new_user for k in ["id", "profile"]):
                    all_users.append(new_user)
                    print(f"Generated user {i+1}/{count}: {new_user['id']}")
                else:
                    print(f"Generated user {i+1} missing required fields, skipping.")
            except json.JSONDecodeError:
                print(f"Generated user {i+1} has invalid JSON format, skipping.")
                
        except Exception as e:
            print(f"Error generating user {i+1}: {str(e)}")
    
    # Save the expanded users
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(user_file_path, 'w') as f:
        json.dump(all_users, f, indent=2)
    
    return all_users 