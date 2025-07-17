# /app/recommendation_error_handler.py
"""
Error Handling and Fallback Mechanisms for Smart Property Recommendations

This module provides comprehensive error handling for the recommendation system,
including graceful degradation and fallback to standard chat flow.
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class RecommendationErrorType(Enum):
    """Types of errors that can occur in the recommendation system."""
    INTENT_DETECTION_FAILED = "intent_detection_failed"
    USER_CONTEXT_UNAVAILABLE = "user_context_unavailable"
    CONVERSATION_STATE_ERROR = "conversation_state_error"
    PROPERTY_RETRIEVAL_FAILED = "property_retrieval_failed"
    RECOMMENDATION_GENERATION_FAILED = "recommendation_generation_failed"
    WORKFLOW_SESSION_ERROR = "workflow_session_error"
    DATABASE_CONNECTION_ERROR = "database_connection_error"
    LLM_API_ERROR = "llm_api_error"


@dataclass
class RecommendationErrorContext:
    """Context information for recommendation system errors."""
    error_type: RecommendationErrorType
    user_id: str
    session_id: Optional[str] = None
    original_message: Optional[str] = None
    error_message: str = ""
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RecommendationErrorHandler:
    """
    Error handler for the recommendation system with comprehensive fallback strategies.
    """
    
    # Fallback messages for different error types
    FALLBACK_MESSAGES = {
        RecommendationErrorType.INTENT_DETECTION_FAILED: 
            "I'm having trouble understanding your request. Could you please rephrase what you're looking for in a property?",
        
        RecommendationErrorType.USER_CONTEXT_UNAVAILABLE: 
            "I couldn't access your preference history. Let me ask you a few quick questions to help find the right property for you.",
        
        RecommendationErrorType.CONVERSATION_STATE_ERROR: 
            "There was an issue with our conversation. Let's start fresh - what kind of property are you looking for?",
        
        RecommendationErrorType.PROPERTY_RETRIEVAL_FAILED: 
            "I'm having trouble accessing the property database right now. Let me try a different search approach.",
        
        RecommendationErrorType.RECOMMENDATION_GENERATION_FAILED: 
            "I encountered an issue generating recommendations. Let me search for properties using your criteria instead.",
        
        RecommendationErrorType.WORKFLOW_SESSION_ERROR: 
            "There was a session error. Let's restart your property search with a fresh conversation.",
        
        RecommendationErrorType.DATABASE_CONNECTION_ERROR: 
            "I'm experiencing connectivity issues. Let me try to help you with a basic property search.",
        
        RecommendationErrorType.LLM_API_ERROR: 
            "I'm having trouble processing your request right now. Could you try asking in a different way?"
    }
    
    def __init__(self):
        self.error_count = {}  # Track error frequency by type
        self.user_error_history = {}  # Track errors per user
    
    async def handle_recommendation_error(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """
        Handle a recommendation system error with appropriate fallback strategy.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            Fallback response with alternative action
        """
        logger.error(f"Recommendation error for user {error_context.user_id}: "
                    f"{error_context.error_type.value} - {error_context.error_message}")
        
        # Track error for monitoring
        self._track_error(error_context)
        
        # Determine fallback strategy based on error type and user history
        fallback_strategy = self._determine_fallback_strategy(error_context)
        
        # Execute fallback strategy
        fallback_response = await self._execute_fallback_strategy(error_context, fallback_strategy)
        
        return fallback_response
    
    def _track_error(self, error_context: RecommendationErrorContext) -> None:
        """Track error for monitoring and pattern analysis."""
        error_type = error_context.error_type.value
        
        # Track global error count
        if error_type not in self.error_count:
            self.error_count[error_type] = 0
        self.error_count[error_type] += 1
        
        # Track user-specific errors
        user_id = error_context.user_id
        if user_id not in self.user_error_history:
            self.user_error_history[user_id] = []
        
        self.user_error_history[user_id].append({
            'error_type': error_type,
            'timestamp': error_context.timestamp,
            'session_id': error_context.session_id,
            'message': error_context.error_message
        })
        
        # Keep only last 10 errors per user
        self.user_error_history[user_id] = self.user_error_history[user_id][-10:]
    
    def _determine_fallback_strategy(self, error_context: RecommendationErrorContext) -> str:
        """Determine the best fallback strategy based on error type and user history."""
        error_type = error_context.error_type
        user_id = error_context.user_id
        
        # Check if user has had recent errors
        recent_errors = self._get_recent_user_errors(user_id)
        has_recent_errors = len(recent_errors) >= 2
        
        # Determine strategy based on error type
        if error_type == RecommendationErrorType.INTENT_DETECTION_FAILED:
            return "clarify_intent" if not has_recent_errors else "fallback_to_standard_chat"
        
        elif error_type == RecommendationErrorType.USER_CONTEXT_UNAVAILABLE:
            return "manual_preference_gathering" if not has_recent_errors else "basic_search"
        
        elif error_type == RecommendationErrorType.CONVERSATION_STATE_ERROR:
            return "restart_conversation" if not has_recent_errors else "fallback_to_standard_chat"
        
        elif error_type == RecommendationErrorType.PROPERTY_RETRIEVAL_FAILED:
            return "alternative_search" if not has_recent_errors else "manual_search_guidance"
        
        elif error_type == RecommendationErrorType.RECOMMENDATION_GENERATION_FAILED:
            return "basic_property_listing" if not has_recent_errors else "search_guidance"
        
        elif error_type in [
            RecommendationErrorType.WORKFLOW_SESSION_ERROR,
            RecommendationErrorType.DATABASE_CONNECTION_ERROR,
            RecommendationErrorType.LLM_API_ERROR
        ]:
            return "fallback_to_standard_chat"
        
        else:
            return "fallback_to_standard_chat"
    
    async def _execute_fallback_strategy(self, error_context: RecommendationErrorContext, 
                                       strategy: str) -> Dict[str, Any]:
        """Execute the determined fallback strategy."""
        
        if strategy == "clarify_intent":
            return await self._clarify_intent_fallback(error_context)
        
        elif strategy == "manual_preference_gathering":
            return await self._manual_preference_gathering_fallback(error_context)
        
        elif strategy == "restart_conversation":
            return await self._restart_conversation_fallback(error_context)
        
        elif strategy == "alternative_search":
            return await self._alternative_search_fallback(error_context)
        
        elif strategy == "basic_property_listing":
            return await self._basic_property_listing_fallback(error_context)
        
        elif strategy == "basic_search":
            return await self._basic_search_fallback(error_context)
        
        elif strategy == "search_guidance":
            return await self._search_guidance_fallback(error_context)
        
        elif strategy == "manual_search_guidance":
            return await self._manual_search_guidance_fallback(error_context)
        
        else:  # fallback_to_standard_chat
            return await self._standard_chat_fallback(error_context)
    
    async def _clarify_intent_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to clarify user intent."""
        return {
            "strategy": "clarify_intent",
            "message": "I want to help you find the perfect property! Are you looking for:\n\n"
                      "• A specific type of apartment or property?\n"
                      "• Properties in a particular area?\n"
                      "• Something within a certain budget range?\n\n"
                      "Let me know what you have in mind!",
            "fallback_to_standard": False,
            "continue_recommendations": True
        }
    
    async def _manual_preference_gathering_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to manually gather user preferences."""
        return {
            "strategy": "manual_preference_gathering",
            "message": "Let me help you find the right property by asking a few questions:\n\n"
                      "1. What's your budget range for monthly rent?\n"
                      "2. Which areas or neighborhoods interest you?\n"
                      "3. Any specific features you're looking for?\n\n"
                      "Feel free to answer any or all of these!",
            "fallback_to_standard": False,
            "continue_recommendations": True
        }
    
    async def _restart_conversation_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to restart the conversation."""
        return {
            "strategy": "restart_conversation",
            "message": "Let's start fresh with your property search! What kind of property are you looking for? "
                      "I can help you find apartments, lofts, or other rental properties based on your preferences.",
            "fallback_to_standard": False,
            "continue_recommendations": True
        }
    
    async def _alternative_search_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to alternative search method."""
        return {
            "strategy": "alternative_search",
            "message": "I'm trying a different search approach for you. "
                      "Could you tell me the most important thing you're looking for in a property? "
                      "For example: location, price range, or specific features.",
            "fallback_to_standard": True,
            "use_basic_rag": True,
            "continue_recommendations": False
        }
    
    async def _basic_property_listing_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to basic property listing."""
        return {
            "strategy": "basic_property_listing",
            "message": "Let me search for available properties for you. "
                      "I'll look for listings that might match your needs. "
                      "What specific details are you most interested in?",
            "fallback_to_standard": True,
            "use_basic_rag": True,
            "continue_recommendations": False
        }
    
    async def _basic_search_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to basic search functionality."""
        return {
            "strategy": "basic_search",
            "message": "I'll help you search for properties. "
                      "What specific information can I help you find about our available listings?",
            "fallback_to_standard": True,
            "use_basic_rag": True,
            "continue_recommendations": False
        }
    
    async def _search_guidance_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to providing search guidance."""
        return {
            "strategy": "search_guidance",
            "message": "I can help you search for properties! Try asking me things like:\n\n"
                      "• 'Show me properties under $3000'\n"
                      "• 'Find apartments in downtown'\n"
                      "• 'What's available with parking?'\n"
                      "• 'Tell me about [specific address]'\n\n"
                      "What would you like to know?",
            "fallback_to_standard": True,
            "use_basic_rag": True,
            "continue_recommendations": False
        }
    
    async def _manual_search_guidance_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to manual search guidance."""
        return {
            "strategy": "manual_search_guidance",
            "message": "I'm having some technical difficulties with the property search. "
                      "You can try:\n\n"
                      "• Asking about specific addresses\n"
                      "• Searching by neighborhood or area\n"
                      "• Looking for properties with specific features\n\n"
                      "What specific property information can I help you find?",
            "fallback_to_standard": True,
            "use_basic_rag": True,
            "continue_recommendations": False
        }
    
    async def _standard_chat_fallback(self, error_context: RecommendationErrorContext) -> Dict[str, Any]:
        """Fallback to standard chat functionality."""
        fallback_message = self.FALLBACK_MESSAGES.get(
            error_context.error_type,
            "I'm experiencing some technical issues. How can I help you with your property search?"
        )
        
        return {
            "strategy": "standard_chat",
            "message": fallback_message,
            "fallback_to_standard": True,
            "use_basic_rag": True,
            "continue_recommendations": False
        }
    
    def _get_recent_user_errors(self, user_id: str, minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors for a user within the specified time window."""
        if user_id not in self.user_error_history:
            return []
        
        recent_errors = []
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        for error in self.user_error_history[user_id]:
            if error['timestamp'].timestamp() > cutoff_time:
                recent_errors.append(error)
        
        return recent_errors
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors_by_type": self.error_count.copy(),
            "users_with_errors": len(self.user_error_history),
            "most_common_error": max(self.error_count.items(), key=lambda x: x[1])[0] if self.error_count else None
        }


# Global error handler instance
recommendation_error_handler = RecommendationErrorHandler() 