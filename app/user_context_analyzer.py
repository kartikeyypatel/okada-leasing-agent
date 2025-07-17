# /app/user_context_analyzer.py
"""
User Context Analyzer for Smart Property Recommendations

This service analyzes user history and preferences from CRM data to inform
personalized property recommendations.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

from app.models import UserContext
from app.crm import get_user_by_email, create_or_update_user
from app.history import get_user_history

logger = logging.getLogger(__name__)


class UserContextAnalyzer:
    """
    Service for analyzing user context and preferences for personalized recommendations.
    
    Integrates with existing CRM system and conversation history to build
    comprehensive user profiles for recommendations.
    """
    
    # Essential preference categories that we need for good recommendations
    ESSENTIAL_PREFERENCES = [
        'budget_range',
        'preferred_locations',
        'required_features',
        'property_type'
    ]
    
    # Property features that users commonly care about
    COMMON_FEATURES = [
        'kitchen', 'chef kitchen', 'modern kitchen', 'updated kitchen',
        'balcony', 'terrace', 'patio', 'outdoor space',
        'parking', 'garage', 'parking spot',
        'laundry', 'washer/dryer', 'in-unit laundry',
        'gym', 'fitness center', 'workout room',
        'pool', 'swimming pool',
        'elevator', 'doorman', 'concierge',
        'pet friendly', 'pets allowed', 'dog friendly',
        'hardwood floors', 'marble', 'granite counters',
        'stainless steel appliances', 'dishwasher',
        'air conditioning', 'heating', 'central air'
    ]
    
    def __init__(self):
        pass
    
    async def analyze_user_context(self, user_id: str) -> UserContext:
        """
        Analyze comprehensive user context for recommendations.
        
        Args:
            user_id: User's email address
            
        Returns:
            UserContext with historical preferences and analysis
        """
        logger.info(f"Analyzing user context for: {user_id}")
        
        # Get user profile from CRM
        user_profile = await self._get_user_profile(user_id)
        
        # Get conversation history
        conversation_history = await self._get_conversation_history(user_id)
        
        # Extract preferences from history
        historical_preferences = await self._extract_preferences_from_history(conversation_history)
        
        # Merge with stored preferences
        combined_preferences = self._merge_preferences(
            user_profile.get('recommendation_preferences', {}),
            historical_preferences
        )
        
        # Build user context
        user_context = UserContext(
            user_id=user_id,
            historical_preferences=combined_preferences,
            budget_range=self._extract_budget_range(combined_preferences),
            preferred_locations=self._extract_locations(combined_preferences),
            required_features=self._extract_features(combined_preferences, 'required'),
            excluded_features=self._extract_features(combined_preferences, 'excluded'),
            last_updated=datetime.now()
        )
        
        logger.info(f"User context analysis complete. Found {len(combined_preferences)} preference categories")
        return user_context
    
    async def identify_missing_preferences(self, context: UserContext) -> List[str]:
        """
        Identify which essential preferences are missing from user context.
        
        Args:
            context: Current user context
            
        Returns:
            List of missing preference categories
        """
        missing = []
        
        # Check budget
        if not context.budget_range and 'budget' not in context.historical_preferences:
            missing.append('budget')
        
        # Check location preferences
        if not context.preferred_locations and 'location' not in context.historical_preferences:
            missing.append('location')
        
        # Check property type
        if 'property_type' not in context.historical_preferences:
            missing.append('property_type')
        
        # Check if we have any feature preferences
        if not context.required_features and 'features' not in context.historical_preferences:
            missing.append('features')
        
        logger.info(f"Missing preferences for {context.user_id}: {missing}")
        return missing
    
    async def merge_new_preferences(self, user_id: str, new_prefs: Dict[str, Any]) -> UserContext:
        """
        Merge new preferences with existing user context.
        
        Args:
            user_id: User's email address
            new_prefs: New preferences to merge
            
        Returns:
            Updated UserContext
        """
        logger.info(f"Merging new preferences for {user_id}: {list(new_prefs.keys())}")
        
        # Get current context
        current_context = await self.analyze_user_context(user_id)
        
        # Merge preferences
        updated_preferences = self._merge_preferences(current_context.historical_preferences, new_prefs)
        
        # Update user context
        updated_context = UserContext(
            user_id=user_id,
            historical_preferences=updated_preferences,
            budget_range=self._extract_budget_range(updated_preferences),
            preferred_locations=self._extract_locations(updated_preferences),
            required_features=self._extract_features(updated_preferences, 'required'),
            excluded_features=self._extract_features(updated_preferences, 'excluded'),
            last_updated=datetime.now()
        )
        
        # Store updated preferences in CRM
        await self._update_user_preferences(user_id, updated_preferences)
        
        return updated_context
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile from CRM system."""
        try:
            user = await get_user_by_email(user_id)
            if user:
                return {
                    'full_name': user.full_name,
                    'email': user.email,
                    'company_name': user.company_name,
                    'recommendation_preferences': user.recommendation_preferences or {},
                    'last_recommendation_date': user.last_recommendation_date,
                    'recommendation_history': user.recommendation_history or []
                }
            else:
                logger.warning(f"User {user_id} not found in CRM")
                return {}
        except Exception as e:
            logger.error(f"Error getting user profile for {user_id}: {e}")
            return {}
    
    async def _get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's conversation history."""
        try:
            history = await get_user_history(user_id)
            return history if history else []
        except Exception as e:
            logger.error(f"Error getting conversation history for {user_id}: {e}")
            return []
    
    async def _extract_preferences_from_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract preferences from conversation history using LLM analysis.
        
        Args:
            history: User's conversation history
            
        Returns:
            Dictionary of extracted preferences
        """
        if not history:
            return {}
        
        # Prepare conversation text for analysis
        recent_conversations = history[-10:]  # Analyze last 10 conversations
        conversation_text = "\n".join([
            f"User: {conv.get('user_message', '')}\nAssistant: {conv.get('assistant_message', '')}"
            for conv in recent_conversations
        ])
        
        if not conversation_text.strip():
            return {}
        
        # LLM-based preference extraction
        extraction_prompt = f"""
        Analyze the following conversation history to extract user preferences for property recommendations.
        Look for mentions of:
        - Budget range or price preferences
        - Location preferences (neighborhoods, streets, areas)
        - Property features they like or want (kitchen, parking, gym, etc.)
        - Property features they dislike or want to avoid
        - Property type preferences (apartment, studio, loft, etc.)
        - Size requirements (bedrooms, bathrooms, square footage)
        
        Conversation History:
        {conversation_text}
        
        Extract preferences and respond with ONLY a JSON object like this:
        {{
            "budget": {{"min": 2000, "max": 3000, "currency": "USD"}},
            "location": ["downtown", "near subway", "Manhattan"],
            "required_features": ["kitchen", "parking", "gym"],
            "excluded_features": ["noisy", "small kitchen"],
            "property_type": "apartment",
            "size": {{"bedrooms": 2, "bathrooms": 1, "min_sqft": 800}}
        }}
        
        If no clear preferences are found, return {{}}.
        """
        
        try:
            response = await Settings.llm.achat([ChatMessage(role="user", content=extraction_prompt)])
            preferences = json.loads(response.message.content or '{}')
            
            logger.debug(f"Extracted preferences from history: {preferences}")
            return preferences
        except Exception as e:
            logger.error(f"Error extracting preferences from history: {e}")
            return {}
    
    def _merge_preferences(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge preference dictionaries, with new preferences taking priority.
        
        Args:
            existing: Existing preferences
            new: New preferences to merge
            
        Returns:
            Merged preferences dictionary
        """
        merged = existing.copy()
        
        for key, value in new.items():
            if key in merged:
                # For lists, merge and deduplicate
                if isinstance(value, list) and isinstance(merged[key], list):
                    merged[key] = list(set(merged[key] + value))
                # For dicts, merge recursively
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **value}
                # For other types, new value takes priority
                else:
                    merged[key] = value
            else:
                merged[key] = value
        
        # Add timestamp
        merged['last_updated'] = datetime.now().isoformat()
        
        return merged
    
    def _extract_budget_range(self, preferences: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract budget range from preferences."""
        budget_info = preferences.get('budget')
        if budget_info and isinstance(budget_info, dict):
            min_budget = budget_info.get('min')
            max_budget = budget_info.get('max')
            if min_budget is not None and max_budget is not None:
                return (int(min_budget), int(max_budget))
            elif max_budget is not None:
                return (0, int(max_budget))
            elif min_budget is not None:
                return (int(min_budget), 999999)  # Large upper bound
        return None
    
    def _extract_locations(self, preferences: Dict[str, Any]) -> List[str]:
        """Extract location preferences from preferences."""
        locations = preferences.get('location', [])
        if isinstance(locations, list):
            return [str(loc).strip() for loc in locations if str(loc).strip()]
        elif isinstance(locations, str):
            return [locations.strip()] if locations.strip() else []
        return []
    
    def _extract_features(self, preferences: Dict[str, Any], feature_type: str) -> List[str]:
        """Extract feature preferences from preferences."""
        if feature_type == 'required':
            features = preferences.get('required_features', [])
        elif feature_type == 'excluded':
            features = preferences.get('excluded_features', [])
        else:
            return []
        
        if isinstance(features, list):
            return [str(feature).strip() for feature in features if str(feature).strip()]
        elif isinstance(features, str):
            return [features.strip()] if features.strip() else []
        return []
    
    async def _update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Update user preferences in CRM system."""
        try:
            await create_or_update_user(
                email=user_id,
                full_name=None,  # Don't update name
                company_name=None,  # Don't update company
                preferences=None,  # Don't update general preferences
                recommendation_preferences=preferences
            )
            logger.info(f"Updated recommendation preferences for {user_id}")
        except Exception as e:
            logger.error(f"Error updating user preferences for {user_id}: {e}")
    
    async def generate_clarifying_question(self, context: UserContext, missing_category: str) -> str:
        """
        Generate a natural clarifying question for a missing preference category.
        
        Args:
            context: Current user context
            missing_category: The preference category that's missing
            
        Returns:
            Natural language question string
        """
        questions = {
            'budget': [
                "What's your budget range for rent?",
                "How much are you looking to spend monthly?",
                "What's your preferred price range?",
            ],
            'location': [
                "Which areas or neighborhoods are you interested in?",
                "Do you have any location preferences?",
                "Where would you like to be located?",
            ],
            'property_type': [
                "Are you looking for a studio, one-bedroom, or larger apartment?",
                "What type of property interests you most?",
                "Do you prefer apartments, lofts, or something else?",
            ],
            'features': [
                "Are there any specific features that are important to you?",
                "What amenities matter most to you?",
                "Do you have any must-have features in mind?",
            ]
        }
        
        # Get appropriate questions for the category
        category_questions = questions.get(missing_category, ["What are your preferences?"])
        
        # Use LLM to personalize the question if we have some context
        if context.historical_preferences:
            personalization_prompt = f"""
            Generate a natural, personalized clarifying question for a user about their {missing_category} preferences.
            
            What we know about the user:
            {json.dumps(context.historical_preferences, indent=2)}
            
            The question should:
            1. Be conversational and friendly
            2. Reference their known preferences if relevant
            3. Be specific to the {missing_category} category
            4. Be easy to answer
            
            Example questions for {missing_category}: {category_questions}
            
            Generate ONE personalized question:
            """
            
            try:
                response = await Settings.llm.achat([ChatMessage(role="user", content=personalization_prompt)])
                personalized_question = response.message.content.strip()
                if personalized_question and not personalized_question.startswith('{'):
                    return personalized_question
            except Exception as e:
                logger.error(f"Error generating personalized question: {e}")
        
        # Fallback to default questions
        return category_questions[0]


# Global service instance
user_context_analyzer = UserContextAnalyzer() 