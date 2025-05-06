"""
Agent modules for the Sekai Recommendation Agent system.
"""

# Legacy agents (workflow-based)
from agents.recommendation_agent import RecommendationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.prompt_optimizer import PromptOptimizerAgent

# New adaptive agents (agent-based)
from agents.agent_core import Agent, Memory
from agents.adaptive_recommendation_agent import AdaptiveRecommendationAgent
from agents.adaptive_evaluation_agent import AdaptiveEvaluationAgent

__all__ = [
    # Legacy agents
    'RecommendationAgent',
    'EvaluationAgent',
    'PromptOptimizerAgent',
    
    # Agent core
    'Agent',
    'Memory',
    
    # New adaptive agents
    'AdaptiveRecommendationAgent',
    'AdaptiveEvaluationAgent'
] 