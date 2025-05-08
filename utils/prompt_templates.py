"""
Prompt templates for various agents and tasks in the Sekai Recommendation Agent system.
"""

RECOMMENDATION_SYSTEM_PROMPT = """
You are an intelligent recommendation agent for Sekai, a platform with interactive stories.
Your goal is to recommend the most relevant stories to users based on their preferences.
Analyze user profiles carefully and match them with the most appropriate stories.
Be systematic, thorough, and explain your recommendation process.
"""

EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluation agent for Sekai's recommendation system.
Your goal is to assess how well story recommendations match a user's preferences.
Analyze user profiles thoroughly, identify their preferences, and evaluate recommendation quality.
Be objective, detailed, and provide constructive feedback.
"""

# AutoPrompt-inspired Intent-based Prompt Calibration template
INTENT_BASED_CALIBRATION_SYSTEM_PROMPT = """
You are an expert prompt engineer specializing in recommendation systems.
Your task is to optimize recommendation prompts using Intent-based Prompt Calibration.

Apply the following optimization principles:
1. Identify edge cases and boundary examples where the system underperforms
2. Extract patterns from successful vs. unsuccessful recommendations
3. Generate improved prompts that handle these edge cases while maintaining core functionality
4. Ensure prompts are concise yet comprehensive
5. Focus on robustness across diverse user preferences
"""

OPTIMIZATION_STANDARD_TEMPLATE = """
# PROMPT OPTIMIZATION TASK

## Current Recommendation Prompt:
```
{current_prompt}
```

## Evaluation Results Summary:
- Successful recommendations: {successful_count} cases
- Unsuccessful recommendations: {unsuccessful_count} cases
- Average score: {average_score:.4f}
- Most effective strategy: {best_strategy}

## Example Success Cases:
{success_examples}

## Example Failure Cases:
{failure_examples}

## Focus Areas for Improvement:
{focus_areas}

## Example Stories:
{story_examples}

# OPTIMIZATION INSTRUCTIONS

Based on the evaluation results, create an improved recommendation prompt that:
1. Addresses the weaknesses in the failure cases
2. Preserves the strengths from successful cases
3. Explicitly encourages the use of "{best_strategy}" strategy when appropriate
4. Is optimized for recommending the most relevant stories to users
5. Provides clear instructions on how to analyze user preferences
6. Explains how to balance relevance with diversity
7. Includes guidance on handling different types of user profiles

## IMPROVED RECOMMENDATION PROMPT:

"""

OPTIMIZATION_EXPLORATION_TEMPLATE = """
# PROMPT EXPLORATION TASK

The current recommendation prompt has plateaued or is underperforming. Try a substantially different approach.

## Current Prompt (to be reimagined):
```
{current_prompt}
```

## Current Performance:
- Average score: {average_score:.4f}
- Plateau detected: Yes
- Iterations without improvement: {plateau_counter}

## Key Weaknesses Identified:
{weakness_patterns}

## OPTIMIZATION INSTRUCTIONS

Create a completely new recommendation prompt with a fresh approach that:
1. Takes a different structural approach to the recommendation problem
2. Addresses the identified weaknesses in a novel way
3. Includes explicit reasoning steps for matching user preferences to stories
4. Creates a balance between relevance and diversity in recommendations
5. Implements clear guidance on handling complex or ambiguous user preferences

## REDESIGNED RECOMMENDATION PROMPT:

"""

OPTIMIZATION_REFINEMENT_TEMPLATE = """
# PROMPT REFINEMENT TASK

The current recommendation prompt is performing well but can be further optimized.

## Current Recommendation Prompt:
```
{current_prompt}
```

## Evaluation Results:
- Average score: {average_score:.4f} (improving)
- Successful strategies: {best_strategy}

## Areas to Enhance:
{focus_areas}

## OPTIMIZATION INSTRUCTIONS

Refine the current prompt to:
1. Enhance the most successful elements while maintaining the overall structure
2. Make the reasoning process more explicit where helpful
3. Add more specific guidance on handling edge cases like:
   {edge_case_examples}
4. Fine-tune instructions for balancing user preferences
5. Preserve what works well while making targeted improvements

## REFINED RECOMMENDATION PROMPT:

"""

FALLBACK_OPTIMIZATION_TEMPLATE = """
I need to improve this recommendation prompt:

```
{current_prompt}
```

Please enhance this prompt to:
1. Better identify user preferences from their profile
2. Balance relevance with diversity in recommendations
3. Be more specific about how to analyze tags and themes
4. Provide clearer instructions for ranking and selecting stories

Return ONLY the improved prompt without explanations or additional text.
"""

# Template for generating user-targeted stories
USER_TARGETED_STORY_TEMPLATE = """
Create a new story that would appeal to this user profile:

{user_profile}

The story should:
1. Match their interests, preferences and favorite genres
2. Include characters and themes they would enjoy
3. Fit within the style of the example stories
4. Be unique and not duplicate existing stories

Please format your response as a JSON object with these fields:
- "title": A catchy title for the story
- "intro": A brief introduction paragraph (1-3 sentences)
- "tags": 5-10 relevant tags describing the story content
- "characters": Brief descriptions of 1-3 main characters
- "plot": A summary of the main plot elements (3-5 sentences)
"""

# Edge case testing template for prompt robustness
EDGE_CASE_TEST_TEMPLATE = """
# EDGE CASE TESTING

Let's test this recommendation prompt against a challenging user profile:

## Prompt to Test:
```
{prompt_to_test}
```

## Challenging User Profile:
{challenging_profile}

## Testing Instructions:
1. Act as the recommendation agent using exactly the prompt above
2. Process the challenging user profile
3. Generate recommendations based solely on the prompt instructions
4. Rate the recommendations on a scale from 1-10 for:
   - Relevance to user interests
   - Diversity of recommendations
   - Handling of ambiguous preferences

## RECOMMENDATION RESULTS:

""" 