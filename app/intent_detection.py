# /app/intent_detection.py
"""
Intent Detection Service for Smart Property Recommendations

This service identifies when user input should trigger the recommendation workflow
and extracts initial preferences from user messages.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

# Import rag module to ensure LLM is properly configured
import app.rag
from app.models import RecommendationIntent

logger = logging.getLogger(__name__)


class IntentDetectionService:
    """
    Service for detecting recommendation intents in user messages.
    
    Uses both pattern matching and LLM-based classification to identify
    when users want property recommendations.
    """
    
    # Trigger phrases that indicate recommendation requests
    RECOMMENDATION_TRIGGERS = [
        # Direct recommendation requests
        r'\b(?:suggest|recommend|find|show)\s+(?:me\s+)?(?:a\s+|some\s+)?(?:property|properties|apartment|apartments|listing|listings|place|places)\b',
        r'\b(?:any|got any)\s+(?:good\s+)?(?:property|properties|apartment|apartments|listing|listings|place|places)\s+(?:for\s+me|available)\b',
        r'\b(?:what\s+do\s+you\s+have|what\s+properties)\b',
        r'\b(?:looking\s+for|searching\s+for)\s+(?:a\s+|some\s+)?(?:property|apartment|place)\b',
        r'\b(?:help\s+me\s+find|can\s+you\s+find)\s+(?:a\s+|some\s+)?(?:property|apartment|place)\b',
        
        # Casual recommendation requests
        r'\b(?:any\s+)?(?:listings|properties|apartments|places)\s+(?:for\s+me|you\s+suggest)\b',
        r'\b(?:what\s+would\s+you\s+recommend|what\s+do\s+you\s+recommend)\b',
        r'\b(?:show\s+me\s+your|what\s+are\s+your)\s+(?:best\s+)?(?:options|listings|properties)\b',
        
        # Question-based triggers
        r'\b(?:do\s+you\s+have\s+any|are\s+there\s+any)\s+(?:good\s+)?(?:properties|apartments|listings)\b',
        r'\b(?:what\s+kinds?\s+of|what\s+types?\s+of)\s+(?:properties|apartments)\s+(?:do\s+you\s+have|are\s+available)\b',
    ]
    
    # Preference extraction patterns
    PREFERENCE_PATTERNS = {
        'budget': [
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:to|[-–])\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:budget|price|rent)\s+(?:of\s+|around\s+|up\s+to\s+)?\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'under\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'max\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        ],
        'location': [
            r'\b(?:in|near|around|close\s+to)\s+([A-Za-z\s]+(?:street|st|avenue|ave|road|rd|blvd|boulevard|drive|dr|lane|ln|court|ct|place|pl)?)\b',
            r'\b(?:downtown|midtown|uptown|brooklyn|manhattan|queens|bronx|staten\s+island)\b',
            r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|blvd|boulevard|drive|dr|lane|ln|court|ct|place|pl)\b',
        ],
        'size': [
            r'(\d+(?:,\d{3})*)\s*(?:sq\s*ft|square\s+feet|sf)\b',
            r'(\d+)\s*(?:bedroom|bed|br)\b',
            r'(\d+)\s*(?:bathroom|bath|ba)\b',
        ],
        'features': [
            r'\b(kitchen|chef\s+kitchen|modern\s+kitchen|updated\s+kitchen)\b',
            r'\b(balcony|terrace|patio|outdoor\s+space)\b',
            r'\b(parking|garage|parking\s+spot)\b',
            r'\b(laundry|washer|dryer|in-unit\s+laundry)\b',
            r'\b(gym|fitness|fitness\s+center|workout\s+room)\b',
            r'\b(pool|swimming\s+pool)\b',
            r'\b(elevator|doorman|concierge)\b',
            r'\b(pet\s+friendly|pets\s+allowed|dog\s+friendly)\b',
        ]
    }
    
    def __init__(self):
        self.compiled_triggers = [re.compile(pattern, re.IGNORECASE) for pattern in self.RECOMMENDATION_TRIGGERS]
        self.compiled_preferences = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.PREFERENCE_PATTERNS.items()
        }
    
    async def detect_recommendation_intent(self, message: str) -> RecommendationIntent:
        """
        Detect if a message contains a recommendation request.
        
        Args:
            message: User's input message
            
        Returns:
            RecommendationIntent with detection results
        """
        logger.info(f"Analyzing message for recommendation intent: '{message[:100]}...'")
        
        # Step 1: Pattern-based detection
        pattern_result = self._detect_patterns(message)
        
        # Step 2: LLM-based validation and confidence scoring
        llm_result = await self._llm_based_detection(message)
        
        # Step 3: Extract initial preferences
        initial_preferences = self.extract_initial_preferences(message)
        
        # Step 4: Combine results
        final_intent = self._combine_detection_results(pattern_result, llm_result, initial_preferences)
        
        logger.info(f"Intent detection result: is_recommendation={final_intent.is_recommendation_request}, "
                   f"confidence={final_intent.confidence:.2f}")
        
        return final_intent
    
    def _detect_patterns(self, message: str) -> Dict[str, Any]:
        """Use regex patterns to detect recommendation triggers."""
        matched_triggers = []
        
        for i, pattern in enumerate(self.compiled_triggers):
            if pattern.search(message):
                matched_triggers.append(self.RECOMMENDATION_TRIGGERS[i])
        
        return {
            'is_recommendation': len(matched_triggers) > 0,
            'confidence': min(0.8, len(matched_triggers) * 0.3),  # Pattern-based confidence
            'triggers': matched_triggers
        }
    
    async def _llm_based_detection(self, message: str) -> Dict[str, Any]:
        """Use LLM to validate and score recommendation intent."""
        detection_prompt = f"""
        Analyze this user message to determine if they are requesting property recommendations.
        
        A recommendation request is when the user wants you to suggest or find properties for them,
        rather than asking about a specific property or general information.
        
        Examples of recommendation requests:
        - "Suggest me a property"
        - "Find me an apartment"
        - "Any good listings for me?"
        - "What properties do you have?"
        - "Show me your best options"
        - "I'm looking for a place"
        
        Examples of NON-recommendation requests:
        - "Tell me about 123 Main St" (specific property inquiry)
        - "What is the rent for the 2-bedroom?" (specific information)
        - "How do I schedule a viewing?" (scheduling)
        - "Thank you" (general chat)
        
        User message: "{message}"
        
        Respond with ONLY a JSON object:
        {{
            "is_recommendation_request": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = await Settings.llm.achat([ChatMessage(role="user", content=detection_prompt)])
            response_content = response.message.content or '{}'
            logger.debug(f"LLM response for intent detection: {response_content}")
            
            # Try to extract JSON from the response
            if response_content.strip().startswith('{'):
                result = json.loads(response_content)
            else:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^}]*\}', response_content)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: parse manually
                    result = {'is_recommendation_request': True, 'confidence': 0.8, 'reasoning': 'Pattern-based fallback'}
            
            return {
                'is_recommendation': result.get('is_recommendation_request', False),
                'confidence': result.get('confidence', 0.0),
                'reasoning': result.get('reasoning', '')
            }
        except Exception as e:
            logger.error(f"LLM-based intent detection failed: {e}")
            logger.error(f"Response content was: {response.message.content if 'response' in locals() else 'No response'}")
            return {
                'is_recommendation': False,
                'confidence': 0.0,
                'reasoning': 'LLM detection failed'
            }
    
    def extract_initial_preferences(self, message: str) -> Dict[str, Any]:
        """
        Extract any explicit preferences mentioned in the user's message.
        
        Args:
            message: User's input message
            
        Returns:
            Dictionary of extracted preferences
        """
        preferences = {}
        
        # Extract budget information
        budget_info = self._extract_budget(message)
        if budget_info:
            preferences['budget'] = budget_info
        
        # Extract location preferences
        locations = self._extract_locations(message)
        if locations:
            preferences['preferred_locations'] = locations
        
        # Extract size requirements
        size_info = self._extract_size_requirements(message)
        if size_info:
            preferences.update(size_info)
        
        # Extract feature preferences
        features = self._extract_features(message)
        if features:
            preferences['required_features'] = features
        
        logger.debug(f"Extracted preferences: {preferences}")
        return preferences
    
    def _extract_budget(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract budget information from message."""
        for pattern in self.compiled_preferences['budget']:
            match = pattern.search(message)
            if match:
                if len(match.groups()) >= 2:  # Range match
                    min_budget = int(match.group(1).replace(',', ''))
                    max_budget = int(match.group(2).replace(',', ''))
                    return {'min': min_budget, 'max': max_budget, 'type': 'range'}
                else:  # Single value match
                    value = int(match.group(1).replace(',', ''))
                    if 'under' in match.group(0).lower() or 'max' in match.group(0).lower():
                        return {'max': value, 'type': 'max'}
                    else:
                        return {'target': value, 'type': 'target'}
        return None
    
    def _extract_locations(self, message: str) -> List[str]:
        """Extract location preferences from message."""
        locations = []
        for pattern in self.compiled_preferences['location']:
            matches = pattern.findall(message)
            locations.extend([match.strip() for match in matches if match.strip()])
        return list(set(locations))  # Remove duplicates
    
    def _extract_size_requirements(self, message: str) -> Dict[str, Any]:
        """Extract size requirements from message."""
        size_info = {}
        
        for pattern in self.compiled_preferences['size']:
            match = pattern.search(message)
            if match:
                value = int(match.group(1).replace(',', ''))
                match_text = match.group(0).lower()
                
                if 'sq' in match_text or 'sf' in match_text:
                    size_info['square_feet'] = value
                elif 'bedroom' in match_text or 'bed' in match_text:
                    size_info['bedrooms'] = value
                elif 'bathroom' in match_text or 'bath' in match_text:
                    size_info['bathrooms'] = value
        
        return size_info
    
    def _extract_features(self, message: str) -> List[str]:
        """Extract feature preferences from message."""
        features = []
        for pattern in self.compiled_preferences['features']:
            matches = pattern.findall(message)
            features.extend([match.strip() for match in matches if match.strip()])
        return list(set(features))  # Remove duplicates
    
    def _combine_detection_results(self, pattern_result: Dict[str, Any], 
                                 llm_result: Dict[str, Any],
                                 preferences: Dict[str, Any]) -> RecommendationIntent:
        """Combine pattern and LLM detection results."""
        
        # Determine if it's a recommendation request
        is_recommendation = pattern_result['is_recommendation'] or llm_result['is_recommendation']
        
        # Calculate combined confidence
        pattern_confidence = pattern_result['confidence']
        llm_confidence = llm_result['confidence']
        
        # Weighted combination (LLM gets higher weight)
        combined_confidence = (pattern_confidence * 0.3) + (llm_confidence * 0.7)
        
        # Boost confidence if preferences are found
        if preferences and is_recommendation:
            combined_confidence = min(1.0, combined_confidence + 0.1)
        
        return RecommendationIntent(
            is_recommendation_request=is_recommendation,
            confidence=combined_confidence,
            initial_preferences=preferences,
            trigger_phrases=pattern_result.get('triggers', [])
        )


# Global service instance
intent_detection_service = IntentDetectionService() 