"""
Core agent framework for the Sekai Recommendation Agent system.
This provides the foundation for building intelligent, adaptive agents.
"""
import time
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Memory:
    """Memory system for agents to store and retrieve information."""
    
    def __init__(self):
        self.short_term = {}  # For temporary, session-based information
        self.long_term = {}   # For persistent information across sessions
        self.context = []     # For tracking conversation/action history
        self.last_updated = time.time()
    
    def add_to_short_term(self, key: str, value: Any):
        """Add information to short-term memory."""
        self.short_term[key] = value
        self.last_updated = time.time()
    
    def add_to_long_term(self, key: str, value: Any):
        """Add information to long-term memory."""
        self.long_term[key] = value
        self.last_updated = time.time()
    
    def get_from_short_term(self, key: str, default: Any = None) -> Any:
        """Retrieve information from short-term memory."""
        return self.short_term.get(key, default)
    
    def get_from_long_term(self, key: str, default: Any = None) -> Any:
        """Retrieve information from long-term memory."""
        return self.long_term.get(key, default)
    
    def add_to_context(self, entry: Dict[str, Any]):
        """Add an entry to the context history."""
        self.context.append({
            **entry,
            "timestamp": time.time()
        })
        self.last_updated = time.time()
    
    def get_recent_context(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent context entries."""
        return self.context[-n:] if self.context else []
    
    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term = {}
        self.last_updated = time.time()
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize memory to a dictionary."""
        return {
            "short_term": self.short_term,
            "long_term": self.long_term,
            "context": self.context,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Memory':
        """Create a Memory instance from serialized data."""
        memory = cls()
        memory.short_term = data.get("short_term", {})
        memory.long_term = data.get("long_term", {})
        memory.context = data.get("context", [])
        memory.last_updated = data.get("last_updated", time.time())
        return memory

    def add_user_perception(self, user_id: str, perceptions: Dict[str, Any]):
        """
        Add user-specific perceptions to short-term memory.
        
        Args:
            user_id: Identifier for the user
            perceptions: Dictionary of perceptions for this user
        """
        # Create a perceptions dictionary if it doesn't exist
        if "user_perceptions" not in self.short_term:
            self.short_term["user_perceptions"] = {}
            
        # Store the perceptions for this specific user
        self.short_term["user_perceptions"][user_id] = perceptions
        
        # Also store the most recent perceptions in the standard location for backward compatibility
        self.short_term["latest_perceptions"] = perceptions
        
        # Update timestamp
        self.last_updated = time.time()
    
    def get_user_perception(self, user_id: str, default: Any = None) -> Dict[str, Any]:
        """
        Retrieve user-specific perceptions from short-term memory.
        
        Args:
            user_id: Identifier for the user
            default: Default value to return if no perceptions found
            
        Returns:
            Dictionary of perceptions for the specified user
        """
        if "user_perceptions" not in self.short_term:
            return default
            
        return self.short_term["user_perceptions"].get(user_id, default)
    
    def get_all_user_perceptions(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all user perceptions from short-term memory.
        
        Returns:
            Dictionary mapping user IDs to their perception dictionaries
        """
        return self.short_term.get("user_perceptions", {})


class Agent(ABC):
    """
    Abstract base class for intelligent agents.
    
    An agent can:
    1. Perceive its environment
    2. Reason about what to do next
    3. Act to achieve goals
    4. Learn from experience
    5. Communicate with other agents
    """
    
    def __init__(self, name: str, model_name: str = None):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent
            model_name: Name of the LLM model to use
        """
        self.name = name
        self.model_name = model_name
        self.memory = Memory()
        self.logger = logging.getLogger(f"Agent:{name}")
        self.goals = []
        self.tools = {}
    
    def set_goal(self, goal: str):
        """Set a goal for the agent."""
        self.goals.append(goal)
        self.logger.info(f"Goal set: {goal}")
        self.memory.add_to_context({"type": "goal", "content": goal})
    
    def register_tool(self, tool_name: str, tool_function: callable):
        """Register a tool that the agent can use."""
        self.tools[tool_name] = tool_function
        self.logger.info(f"Tool registered: {tool_name}")
    
    @abstractmethod
    def perceive(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data to update the agent's understanding of its environment.
        
        Args:
            input_data: Data from the environment
            
        Returns:
            Dictionary of perceptions
        """
        pass
    
    @abstractmethod
    def reason(self, perceptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason about what to do based on perceptions.
        
        Args:
            perceptions: Dictionary of perceptions
            
        Returns:
            Plan of action
        """
        pass
    
    @abstractmethod
    def act(self, plan: Dict[str, Any]) -> Any:
        """
        Take action based on a plan.
        
        Args:
            plan: Plan of action
            
        Returns:
            Result of the action
        """
        pass
    
    def learn(self, experience: Dict[str, Any]):
        """
        Learn from experience.
        
        Args:
            experience: Dictionary containing action, result, and feedback
        """
        # By default, just store the experience in memory
        self.memory.add_to_context(experience)
        self.logger.info(f"Learned from experience: {experience.get('action', 'Unknown action')}")
    
    def communicate(self, message: Dict[str, Any], recipient: Optional['Agent'] = None) -> Dict[str, Any]:
        """
        Communicate with another agent or the environment.
        
        Args:
            message: Message to communicate
            recipient: Recipient agent (if any)
            
        Returns:
            Response message
        """
        if recipient:
            self.logger.info(f"Sending message to {recipient.name}")
            return recipient.receive_message(message, self)
        
        # Default implementation for environment communication
        self.logger.info("Sending message to environment")
        self.memory.add_to_context({"type": "outgoing_message", "content": message})
        return {"status": "sent", "timestamp": time.time()}
    
    def receive_message(self, message: Dict[str, Any], sender: Optional['Agent'] = None) -> Dict[str, Any]:
        """
        Receive a message from another agent or the environment.
        
        Args:
            message: Received message
            sender: Sender agent (if any)
            
        Returns:
            Response message
        """
        sender_name = sender.name if sender else "environment"
        self.logger.info(f"Received message from {sender_name}")
        self.memory.add_to_context({
            "type": "incoming_message", 
            "content": message, 
            "sender": sender_name
        })
        
        # Default response
        return {"status": "received", "timestamp": time.time()}
    
    def run_perception_reasoning_action_loop(self, input_data: Any) -> Any:
        """
        Run a complete cycle of perception, reasoning, and action.
        
        Args:
            input_data: Input data for perception
            
        Returns:
            Result of the action
        """
        self.logger.info("Starting perception-reasoning-action cycle")
        
        # Perceive
        perceptions = self.perceive(input_data)
        self.memory.add_to_context({"type": "perception", "content": perceptions})
        
        # Reason
        plan = self.reason(perceptions)
        self.memory.add_to_context({"type": "plan", "content": plan})
        
        # Act
        result = self.act(plan)
        self.memory.add_to_context({
            "type": "action_result", 
            "plan": plan, 
            "result": result
        })
        
        self.logger.info("Completed perception-reasoning-action cycle")
        return result
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize agent state to a dictionary."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "memory": self.memory.serialize(),
            "goals": self.goals,
            "tools": list(self.tools.keys())  # Just save tool names, not the functions
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Agent':
        """Create an Agent instance from serialized data (to be implemented by subclasses)."""
        raise NotImplementedError("Deserialization must be implemented by subclasses") 