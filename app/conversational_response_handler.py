# /app/conversational_response_handler.py
"""
Conversational Response Handler for Performance Optimization

This service handles non-property related conversations efficiently using
response templates and personalization without triggering expensive RAG workflows.
"""

import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from app.models import MessageType, ConversationalContext
import app.crm as crm_module

logger = logging.getLogger(__name__)


class ConversationalResponseHandler:
    """
    Service for handling conversational messages with quick, personalized responses.
    
    Provides appropriate responses for greetings, thanks, help requests, and general
    chat without triggering expensive property search or RAG workflows.
    """
    
    # Greeting response templates
    GREETING_RESPONSES = {
        "morning": [
            "Good morning! 🌅 I'm here to help you find the perfect property. How can I assist you today?",
            "Hello and good morning! ☀️ Ready to explore some amazing properties?",
            "Good morning! I hope you're having a great start to your day. What can I help you with?",
        ],
        "afternoon": [
            "Good afternoon! 🌤️ I'm here to help with all your property needs. What can I do for you?",
            "Hello! Hope you're having a wonderful afternoon. How can I assist you today?",
            "Good afternoon! Ready to find your next home or investment property?",
        ],
        "evening": [
            "Good evening! 🌆 I'm here to help you explore properties. What are you looking for?",
            "Hello! Thanks for stopping by this evening. How can I help you today?",
            "Good evening! Perfect time to browse some properties. What interests you?",
        ],
        "general": [
            "Hello! 👋 Welcome to Okada Leasing. I'm here to help you find amazing properties. What can I do for you?",
            "Hi there! I'm your property assistant. Whether you're looking to rent or just browsing, I'm here to help!",
            "Hey! Great to meet you. I specialize in helping people find their perfect property. What are you looking for?",
            "Hello! I'm excited to help you with your property search. What kind of place interests you?",
        ]
    }
    
    # Thank you response templates
    THANK_YOU_RESPONSES = [
        "You're very welcome! 😊 Is there anything else I can help you with regarding properties?",
        "My pleasure! Feel free to ask if you need help finding specific properties or areas.",
        "Happy to help! Let me know if you'd like to explore more property options.",
        "You're welcome! I'm here whenever you need property assistance.",
        "Glad I could help! Ready to look at more properties or have other questions?",
    ]
    
    # Help response templates
    HELP_RESPONSES = [
        """I'm your property assistant! Here's what I can help you with:

🏠 **Property Search**: Find apartments, houses, and rentals based on your preferences
📍 **Location-Based Search**: Discover properties in specific neighborhoods or areas  
💰 **Budget Planning**: Find properties within your price range
📊 **Property Details**: Get information about rent, size, amenities, and more
📅 **Appointment Booking**: Schedule property viewings and consultations

Just tell me what you're looking for! For example:
• "Show me 2-bedroom apartments under $3000"
• "Find properties near downtown"
• "I want to book an appointment to view properties"

What would you like to explore?""",

        """I'm here to make your property search easy! I can help you:

✨ **Find Properties**: Search by location, price, size, or features
🔍 **Property Details**: Get comprehensive information about any listing
📅 **Schedule Viewings**: Book appointments to see properties in person
💡 **Recommendations**: Get personalized property suggestions
📊 **Market Insights**: Learn about different neighborhoods and pricing

Try asking me something like:
• "What properties do you have in Brooklyn?"
• "Show me luxury apartments"
• "I need help finding a family-friendly neighborhood"

What's your property goal today?""",
    ]
    
    # Conversational response templates  
    CONVERSATIONAL_RESPONSES = [
        "I'm doing great, thank you for asking! 😊 I'm here and ready to help you find amazing properties. What brings you here today?",
        "Thanks for asking! I'm here and excited to help you with your property search. Are you looking for something specific?",
        "I'm wonderful, thanks! Perfect day to explore some fantastic properties. What kind of place are you interested in?",
        "Doing fantastic! I love helping people find their ideal properties. What's your property goal today?",
    ]
    
    # Follow-up suggestions
    PROPERTY_SUGGESTIONS = [
        "Would you like me to show you some popular properties in the area?",
        "I can help you find properties based on your budget, location, or specific features.",
        "Are you looking for a specific type of property - apartment, house, or something else?",
        "I have access to many great listings. What neighborhood interests you?",
        "Would you like to see some recently available properties?",
    ]
    
    def __init__(self):
        self.response_count = 0
    
    def handle_greeting(self, message: str, user_context: Optional[ConversationalContext] = None) -> str:
        """
        Handle greeting messages with personalized responses.
        
        Args:
            message: User's greeting message
            user_context: Optional user context for personalization
            
        Returns:
            Appropriate greeting response with property guidance
        """
        logger.debug(f"Handling greeting: {message}")
        
        # Determine time of day for appropriate greeting
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            time_category = "morning"
        elif 12 <= current_hour < 17:
            time_category = "afternoon"
        elif 17 <= current_hour < 22:
            time_category = "evening"
        else:
            time_category = "general"
        
        # Select base response
        responses = self.GREETING_RESPONSES.get(time_category, self.GREETING_RESPONSES["general"])
        base_response = random.choice(responses)
        
        # Add personalization if user context available
        if user_context:
            base_response = self._personalize_greeting(base_response, user_context)
        
        self.response_count += 1
        return base_response
    
    def handle_thanks(self, message: str, user_context: Optional[ConversationalContext] = None) -> str:
        """
        Handle thank you messages with follow-up offers.
        
        Args:
            message: User's thank you message
            user_context: Optional user context for personalization
            
        Returns:
            Appreciation response with follow-up suggestions
        """
        logger.debug(f"Handling thanks: {message}")
        
        base_response = random.choice(self.THANK_YOU_RESPONSES)
        
        # Add a property suggestion
        suggestion = random.choice(self.PROPERTY_SUGGESTIONS)
        full_response = f"{base_response}\n\n{suggestion}"
        
        self.response_count += 1
        return full_response
    
    def handle_help_request(self, message: str, user_context: Optional[ConversationalContext] = None) -> str:
        """
        Handle help requests with comprehensive guidance.
        
        Args:
            message: User's help request message
            user_context: Optional user context for personalization
            
        Returns:
            Detailed help response with capabilities and examples
        """
        logger.debug(f"Handling help request: {message}")
        
        # Select appropriate help response
        help_response = random.choice(self.HELP_RESPONSES)
        
        # Add personalization if user has context
        if user_context and user_context.previous_interactions > 0:
            personal_note = f"\n\n💡 Since we've chatted before, I remember you're interested in property searches. Feel free to be specific about what you're looking for!"
            help_response += personal_note
        
        self.response_count += 1
        return help_response
    
    def handle_general_chat(self, message: str, user_context: Optional[ConversationalContext] = None) -> str:
        """
        Handle general conversational messages.
        
        Args:
            message: User's conversational message
            user_context: Optional user context for personalization
            
        Returns:
            Friendly conversational response with property redirection
        """
        logger.debug(f"Handling general chat: {message}")
        
        base_response = random.choice(self.CONVERSATIONAL_RESPONSES)
        
        # Add a property suggestion to guide conversation
        suggestion = random.choice(self.PROPERTY_SUGGESTIONS)
        full_response = f"{base_response}\n\n{suggestion}"
        
        self.response_count += 1
        return full_response
    
    def get_response_for_type(self, message_type: MessageType, message: str, 
                            user_context: Optional[ConversationalContext] = None) -> str:
        """
        Get appropriate response based on message type.
        
        Args:
            message_type: Classified message type
            message: Original user message
            user_context: Optional user context
            
        Returns:
            Appropriate response for the message type
        """
        try:
            if message_type == MessageType.GREETING:
                return self.handle_greeting(message, user_context)
            elif message_type == MessageType.THANK_YOU:
                return self.handle_thanks(message, user_context)
            elif message_type == MessageType.HELP_REQUEST:
                return self.handle_help_request(message, user_context)
            elif message_type == MessageType.CONVERSATIONAL:
                return self.handle_general_chat(message, user_context)
            else:
                # Fallback for unknown types
                return self._get_fallback_response(message, user_context)
                
        except Exception as e:
            logger.error(f"Error generating response for type {message_type}: {e}")
            return self._get_error_fallback_response()
    
    async def get_user_context(self, user_id: str) -> Optional[ConversationalContext]:
        """
        Get user context for personalization.
        
        Args:
            user_id: User identifier
            
        Returns:
            ConversationalContext if available, None otherwise
        """
        try:
            # Get user info from CRM
            user_profile = await crm_module.get_user_profile(user_id)
            
            if user_profile:
                return ConversationalContext(
                    user_id=user_id,
                    user_name=user_profile.get('name'),
                    previous_interactions=user_profile.get('interaction_count', 0),
                    last_interaction_time=user_profile.get('last_interaction'),
                    user_preferences=user_profile.get('preferences', {}),
                    conversation_stage="returning" if user_profile.get('interaction_count', 0) > 0 else "initial"
                )
            else:
                return ConversationalContext(
                    user_id=user_id,
                    conversation_stage="initial"
                )
                
        except Exception as e:
            logger.error(f"Error getting user context for {user_id}: {e}")
            return ConversationalContext(user_id=user_id, conversation_stage="initial")
    
    def _personalize_greeting(self, base_response: str, context: ConversationalContext) -> str:
        """Add personalization to greeting responses."""
        
        # Add name if available
        if context.user_name:
            base_response = base_response.replace("Hello!", f"Hello, {context.user_name}!")
            base_response = base_response.replace("Hi there!", f"Hi, {context.user_name}!")
        
        # Add returning user note
        if context.previous_interactions > 0:
            if context.previous_interactions == 1:
                personal_note = " Great to see you back!"
            elif context.previous_interactions < 5:
                personal_note = " Welcome back!"
            else:
                personal_note = " Always a pleasure to help you with properties!"
            
            base_response += personal_note
        
        # Add recent interaction note
        if context.last_interaction_time:
            time_diff = datetime.now() - context.last_interaction_time
            if time_diff < timedelta(hours=1):
                base_response += " Continuing where we left off..."
            elif time_diff < timedelta(days=1):
                base_response += " Back for more property exploration?"
        
        return base_response
    
    def _get_fallback_response(self, message: str, context: Optional[ConversationalContext]) -> str:
        """Generate fallback response for unknown message types."""
        fallback_responses = [
            "I understand you're reaching out! I'm here to help you find great properties. What are you looking for today?",
            "Thanks for your message! I specialize in property searches and recommendations. How can I assist you?",
            "I'm here to help with all your property needs! What kind of place are you interested in?",
        ]
        
        base_response = random.choice(fallback_responses)
        suggestion = random.choice(self.PROPERTY_SUGGESTIONS)
        
        return f"{base_response}\n\n{suggestion}"
    
    def _get_error_fallback_response(self) -> str:
        """Generate error fallback response."""
        return ("I'm here to help you find amazing properties! "
                "You can ask me about available apartments, houses, or schedule property viewings. "
                "What interests you today?")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the response handler."""
        return {
            "total_responses_generated": self.response_count,
            "avg_response_time_estimate": "< 1 second",
            "response_types_supported": [
                "greetings", "thanks", "help_requests", "general_conversation"
            ]
        } 