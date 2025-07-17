# /app/conversation_state_manager.py
"""
Conversation State Manager for Smart Property Recommendations

This service manages the conversational flow and state during the recommendation process,
including session management, question generation, and conversation completion detection.
"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

from app.models import ConversationSession, ConversationState, UserContext
from app.database import get_database
from app.user_context_analyzer import user_context_analyzer

logger = logging.getLogger(__name__)


class ConversationStateManager:
    """
    Service for managing conversation state and workflow during recommendation process.
    
    Handles session persistence, question generation, and conversation progression.
    """
    
    # Maximum number of clarifying questions to ask
    MAX_QUESTIONS = 3
    
    # Session timeout (30 minutes)
    SESSION_TIMEOUT = timedelta(minutes=30)
    
    # Question templates for different preference categories
    QUESTION_TEMPLATES = {
        'budget': [
            "What's your budget range for monthly rent?",
            "How much are you looking to spend per month?",
            "What's your preferred price range?",
        ],
        'location': [
            "Which areas or neighborhoods interest you?",
            "Do you have any location preferences?",
            "Where would you like to be located?",
        ],
        'property_type': [
            "Are you looking for a studio, one-bedroom, or larger apartment?",
            "What type of property interests you most?",
            "Do you prefer apartments, lofts, or something specific?",
        ],
        'features': [
            "Are there any specific amenities that are important to you?",
            "What features matter most to you in a property?",
            "Do you have any must-have amenities in mind?",
        ],
        'size': [
            "How much space do you need?",
            "Do you have any size requirements?",
            "What's your preferred square footage or room count?",
        ]
    }
    
    def __init__(self):
        pass
    
    async def create_session(self, user_id: str, initial_context: UserContext) -> ConversationSession:
        """
        Create a new conversation session for recommendation workflow.
        
        Args:
            user_id: User's email address
            initial_context: Initial user context from analysis
            
        Returns:
            New ConversationSession
        """
        session_id = str(uuid.uuid4())
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            state=ConversationState.INITIATED,
            collected_preferences={},
            questions_asked=[],
            responses_received=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store session in database
        await self._store_session(session)
        
        logger.info(f"Created new conversation session {session_id} for user {user_id}")
        return session
    
    async def update_session(self, session_id: str, user_response: str) -> ConversationSession:
        """
        Update conversation session with user response.
        
        Args:
            session_id: Session identifier
            user_response: User's response to the last question
            
        Returns:
            Updated ConversationSession
        """
        session = await self._get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add response to session
        session.responses_received.append(user_response)
        session.updated_at = datetime.now()
        
        # Extract preferences from response
        extracted_prefs = await self._extract_preferences_from_response(
            user_response, 
            session.questions_asked[-1] if session.questions_asked else ""
        )
        
        # Merge with collected preferences
        session.collected_preferences.update(extracted_prefs)
        
        # Update session state
        await self._update_session_state(session)
        
        # Store updated session
        await self._store_session(session)
        
        logger.info(f"Updated session {session_id} with new response. State: {session.state}")
        return session
    
    async def get_next_question(self, session: ConversationSession) -> Optional[str]:
        """
        Generate the next clarifying question for the conversation.
        
        Args:
            session: Current conversation session
            
        Returns:
            Next question string or None if conversation is complete
        """
        if session.state in [ConversationState.COMPLETED, ConversationState.FAILED]:
            return None
        
        if len(session.questions_asked) >= self.MAX_QUESTIONS:
            return None
        
        # Get user context to identify missing preferences
        user_context = await user_context_analyzer.analyze_user_context(session.user_id)
        
        # Merge session preferences with user context
        merged_preferences = {**user_context.historical_preferences, **session.collected_preferences}
        
        # Identify what we still need to ask about
        missing_categories = await self._identify_missing_categories(merged_preferences)
        
        if not missing_categories:
            return None  # We have enough information
        
        # Generate personalized question for the first missing category
        next_category = missing_categories[0]
        question = await self._generate_personalized_question(
            session, 
            next_category, 
            merged_preferences
        )
        
        # Add question to session
        session.questions_asked.append(question)
        session.updated_at = datetime.now()
        
        # Store updated session
        await self._store_session(session)
        
        logger.info(f"Generated question for session {session.session_id}: {question}")
        return question
    
    async def is_conversation_complete(self, session: ConversationSession) -> bool:
        """
        Check if we have gathered sufficient information to proceed with recommendations.
        
        Args:
            session: Current conversation session
            
        Returns:
            True if conversation is complete
        """
        # Get user context
        user_context = await user_context_analyzer.analyze_user_context(session.user_id)
        
        # Merge session preferences with user context
        merged_preferences = {**user_context.historical_preferences, **session.collected_preferences}
        
        # Check if we have essential preferences
        missing_categories = await self._identify_missing_categories(merged_preferences)
        
        # Complete if we have enough info or hit question limit
        is_complete = (
            len(missing_categories) == 0 or 
            len(session.questions_asked) >= self.MAX_QUESTIONS
        )
        
        if is_complete:
            session.state = ConversationState.COMPLETED
            session.updated_at = datetime.now()
            await self._store_session(session)
        
        return is_complete
    
    async def _identify_missing_categories(self, preferences: Dict[str, Any]) -> List[str]:
        """Identify which essential preference categories are missing."""
        missing = []
        
        # Check budget
        if not preferences.get('budget') and not preferences.get('budget_range'):
            missing.append('budget')
        
        # Check location (lower priority, not required)
        if not preferences.get('location') and not preferences.get('preferred_locations'):
            if len(missing) < 2:  # Only ask about location if we have room
                missing.append('location')
        
        # Check property type
        if not preferences.get('property_type'):
            missing.append('property_type')
        
        # Check features (optional, only if we have room)
        if not preferences.get('features') and not preferences.get('required_features'):
            if len(missing) < 2:
                missing.append('features')
        
        return missing[:2]  # Limit to 2 most important categories
    
    async def _generate_personalized_question(self, session: ConversationSession, 
                                            category: str, 
                                            preferences: Dict[str, Any]) -> str:
        """Generate a personalized question for a preference category."""
        
        # Get base questions for the category
        base_questions = self.QUESTION_TEMPLATES.get(category, ["What are your preferences?"])
        
        # If we have some context, personalize the question
        if preferences:
            personalization_prompt = f"""
            Generate a natural, conversational question to ask a user about their {category} preferences for property recommendations.
            
            Context about the user's existing preferences:
            {json.dumps(preferences, indent=2)}
            
            Questions they've already been asked:
            {json.dumps(session.questions_asked, indent=2)}
            
            The question should:
            1. Be friendly and conversational
            2. Reference their known preferences if relevant
            3. Be specific to {category}
            4. Be easy to answer
            5. Not repeat previously asked questions
            
            Generate ONE personalized question (just the question text, no extra formatting):
            """
            
            try:
                response = await Settings.llm.achat([ChatMessage(role="user", content=personalization_prompt)])
                personalized_question = response.message.content.strip()
                
                # Validate the response
                if (personalized_question and 
                    not personalized_question.startswith('{') and 
                    len(personalized_question) < 200 and
                    '?' in personalized_question):
                    return personalized_question
            except Exception as e:
                logger.error(f"Error generating personalized question: {e}")
        
        # Fallback to template questions
        return base_questions[0]
    
    async def _extract_preferences_from_response(self, response: str, question: str) -> Dict[str, Any]:
        """Extract preferences from user's response to a question."""
        
        extraction_prompt = f"""
        Extract preference information from the user's response to a clarifying question.
        
        Question asked: "{question}"
        User's response: "{response}"
        
        Extract relevant preferences and respond with ONLY a JSON object. Examples:
        
        For budget questions:
        {{"budget": {{"min": 2000, "max": 3000}}}}
        
        For location questions:
        {{"location": ["downtown", "near subway"]}}
        
        For property type questions:
        {{"property_type": "apartment"}}
        
        For feature questions:
        {{"required_features": ["kitchen", "parking"]}}
        
        If no clear preferences can be extracted, return {{}}.
        """
        
        try:
            llm_response = await Settings.llm.achat([ChatMessage(role="user", content=extraction_prompt)])
            preferences = json.loads(llm_response.message.content or '{}')
            
            logger.debug(f"Extracted preferences from response: {preferences}")
            return preferences if isinstance(preferences, dict) else {}
        except Exception as e:
            logger.error(f"Error extracting preferences from response: {e}")
            return {}
    
    async def _update_session_state(self, session: ConversationSession) -> None:
        """Update session state based on progress."""
        if session.state == ConversationState.INITIATED:
            session.state = ConversationState.GATHERING_PREFERENCES
        elif session.state == ConversationState.GATHERING_PREFERENCES:
            if len(session.questions_asked) >= self.MAX_QUESTIONS:
                session.state = ConversationState.GENERATING_RECOMMENDATIONS
            else:
                session.state = ConversationState.CLARIFYING_DETAILS
        elif session.state == ConversationState.CLARIFYING_DETAILS:
            if await self.is_conversation_complete(session):
                session.state = ConversationState.GENERATING_RECOMMENDATIONS
    
    async def _store_session(self, session: ConversationSession) -> None:
        """Store conversation session in database."""
        try:
            db = get_database()
            sessions_collection = db["conversation_sessions"]
            
            session_dict = session.model_dump()
            session_dict['_id'] = session.session_id  # Use session_id as document ID
            
            await sessions_collection.replace_one(
                {"_id": session.session_id},
                session_dict,
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error storing session {session.session_id}: {e}")
    
    async def _get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve conversation session from database."""
        try:
            db = get_database()
            sessions_collection = db["conversation_sessions"]
            
            session_data = await sessions_collection.find_one({"_id": session_id})
            if session_data:
                # Remove MongoDB _id field and restore session_id
                session_data.pop('_id', None)
                session_data['session_id'] = session_id
                return ConversationSession(**session_data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired conversation sessions. Returns number of sessions cleaned."""
        try:
            db = get_database()
            sessions_collection = db["conversation_sessions"]
            
            cutoff_time = datetime.now() - self.SESSION_TIMEOUT
            
            result = await sessions_collection.delete_many({
                "updated_at": {"$lt": cutoff_time}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} expired conversation sessions")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0


# Global service instance
conversation_state_manager = ConversationStateManager() 