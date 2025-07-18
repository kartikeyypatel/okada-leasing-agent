# /app/appointment_intent_detection.py
"""
Appointment Intent Detection Service

This service detects when users want to book appointments through natural language
and extracts relevant appointment details from their messages.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

from app.models import AppointmentIntent

logger = logging.getLogger(__name__)


class AppointmentIntentDetectionService:
    """
    Service for detecting appointment booking intents and extracting appointment details.
    
    Integrates with existing intent detection patterns to recognize when users
    want to schedule appointments, meetings, or calls.
    """
    
    # High-confidence appointment trigger phrases
    APPOINTMENT_TRIGGERS = [
        # Direct appointment requests
        r"\b(book|schedule|set up|arrange|make)\s+(an?\s+)?(appointment|meeting|call|session)\b",
        r"\bi\s+(want|need|would like)\s+to\s+(book|schedule|set up|arrange|make)",
        r"\b(can|could)\s+(i|we)\s+(book|schedule|set up|arrange|make)",
        
        # Meeting-specific phrases
        r"\b(let's|lets)\s+(meet|schedule|set up a meeting)\b",
        r"\b(need|want)\s+to\s+(meet|have a meeting)\b",
        r"\b(schedule|set up)\s+(a|the)\s+(meeting|call|appointment)\b",
        
        # Time-based requests
        r"\b(free|available)\s+(on|at|for|tomorrow|next week|this week)\b",
        r"\b(when\s+(can|are you)\s+)?(available|free)\b",
        r"\bmeet\s+(on|at|tomorrow|next week|this week)\b",
        
        # Calendar-related phrases
        r"\b(put it in|add to|block)\s+(my|the)\s+calendar\b",
        r"\bcalendar\s+(invite|invitation|meeting)\b",
        r"\bsend\s+(me\s+)?(a\s+)?(calendar\s+)?(invite|invitation)\b"
    ]
    
    # Detail extraction patterns - FIXED to prevent incorrect extractions
    DETAIL_PATTERNS = {
        'date': [
            r"\b(tomorrow|today|next week|this week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}(st|nd|rd|th)?\s+(of\s+)?\w+)\b",
            r"\bon\s+(\w+,?\s*\w*\s*\d{1,2}(st|nd|rd|th)?)\b"
        ],
        'time': [
            r"\b(\d{1,2}(:\d{2})?\s*(am|pm|AM|PM))\b",
            r"\bat\s+(\d{1,2}(:\d{2})?\s*(am|pm|AM|PM))\b",
            r"\b(morning|afternoon|evening|noon)\b"
        ],
        'location': [
            # More specific location patterns to avoid false matches
            r"\bat\s+(?:the\s+)?(office|headquarters|building|boardroom|conference room)\b",
            r"\bin\s+(?:the\s+)?(conference room|meeting room|office|boardroom)\b",
            r"\bat\s+(\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr))\b",
            r"\bat\s+([A-Z][A-Za-z\s,.-]{5,}(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr))\b"
        ],
        'email': [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\bwith\s+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b"
        ],
        'title': [
            # More specific title patterns to avoid false matches from trigger phrases
            r"\b(meeting|appointment|call|session)\s+(about|for|regarding)\s+([A-Za-z\s]{3,})",
            r"\b(discuss|review|talk about|go over)\s+([A-Za-z\s]{3,})",
            r"\b([A-Za-z\s]{3,})\s+(consultation|interview|presentation|demo)\b"
        ]
    }
    
    def __init__(self):
        pass
    
    async def detect_appointment_intent(self, message: str) -> AppointmentIntent:
        """
        Detect if a message contains an appointment booking intent.
        
        Args:
            message: User's message to analyze
            
        Returns:
            AppointmentIntent with detection results and extracted details
        """
        logger.info(f"Analyzing message for appointment intent: '{message}'")
        
        # Step 1: Pattern-based detection for high confidence
        pattern_confidence = self._calculate_pattern_confidence(message)
        
        # Step 2: LLM-based intent classification for nuanced detection
        llm_confidence = await self._llm_intent_classification(message)
        
        # Step 3: Combine confidences (weighted toward patterns for reliability)
        combined_confidence = (pattern_confidence * 0.7) + (llm_confidence * 0.3)
        
        # Step 4: Extract appointment details if intent detected
        extracted_details = {}
        missing_fields = []
        
        if combined_confidence > 0.6:  # High confidence threshold
            extracted_details = self.extract_appointment_details(message)
            missing_fields = self._identify_missing_fields(extracted_details)
        
        is_appointment_request = combined_confidence > 0.6
        
        logger.info(f"Appointment intent detection: {is_appointment_request} (confidence: {combined_confidence:.2f})")
        
        return AppointmentIntent(
            is_appointment_request=is_appointment_request,
            confidence=combined_confidence,
            extracted_details=extracted_details,
            missing_fields=missing_fields
        )
    
    def extract_appointment_details(self, message: str) -> Dict[str, Any]:
        """
        Extract specific appointment details from a message.
        
        Args:
            message: User's message to analyze
            
        Returns:
            Dictionary with extracted appointment details
        """
        details = {}
        message_lower = message.lower()
        
        # Extract each type of detail
        for detail_type, patterns in self.DETAIL_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, message_lower, re.IGNORECASE)
                if matches:
                    if detail_type == 'email':
                        details[detail_type] = [match if isinstance(match, str) else match[0] for match in matches]
                    elif detail_type == 'title':
                        # Special handling for title to avoid generic appointment words
                        match_value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                        # Filter out generic appointment trigger words
                        generic_words = ['schedule', 'book', 'arrange', 'make', 'set up', 'an', 'a', 'the']
                        if isinstance(match_value, str) and match_value.strip().lower() not in generic_words:
                            details[detail_type] = match_value.strip()
                    else:
                        # Take the first match for single-value fields
                        match_value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                        # Additional validation for location to avoid false matches
                        if detail_type == 'location':
                            # Don't accept single common words as locations
                            if isinstance(match_value, str) and len(match_value.strip().split()) >= 1:
                                # Avoid obvious false matches
                                false_location_words = ['schedule', 'book', 'an', 'appointment', 'meeting', 'call']
                                if match_value.strip().lower() not in false_location_words:
                                    details[detail_type] = match_value.strip()
                        else:
                            details[detail_type] = match_value
                    break
        
        logger.info(f"Extracted appointment details: {details}")
        return details
    
    def _calculate_pattern_confidence(self, message: str) -> float:
        """Calculate confidence based on pattern matching."""
        message_lower = message.lower()
        
        # Check for appointment trigger phrases
        trigger_matches = 0
        strong_triggers = 0
        for pattern in self.APPOINTMENT_TRIGGERS:
            if re.search(pattern, message_lower):
                trigger_matches += 1
                # Some patterns are stronger indicators
                if any(word in pattern for word in ["book", "schedule", "arrange", "make"]):
                    strong_triggers += 1
        
        # Base confidence from trigger patterns - increased for strong triggers
        if trigger_matches > 0:
            if strong_triggers > 0:
                base_confidence = min(0.9, 0.6 + (strong_triggers * 0.1))
            else:
                base_confidence = min(0.8, 0.4 + (trigger_matches * 0.2))
        else:
            base_confidence = 0.0
        
        # Boost confidence if specific appointment details are mentioned
        detail_boost = 0.0
        for detail_type, patterns in self.DETAIL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detail_boost += 0.1
                    break
        
        total_confidence = min(1.0, base_confidence + detail_boost)
        
        logger.debug(f"Pattern confidence: {total_confidence:.2f} (triggers: {trigger_matches}, strong: {strong_triggers}, detail_boost: {detail_boost:.1f})")
        return total_confidence
    
    async def _llm_intent_classification(self, message: str) -> float:
        """Use LLM for nuanced intent classification."""
        classification_prompt = f"""
        Analyze this message to determine if the user wants to book, schedule, or arrange an appointment, meeting, or call.

        Consider these as appointment requests:
        - Direct requests to book/schedule meetings
        - Asking about availability
        - Wanting to set up calls or sessions
        - Planning to meet someone
        - Calendar-related requests

        Do NOT consider these as appointment requests:
        - General questions about properties
        - Asking for information
        - Casual conversation
        - Already scheduled meeting references

        Message: "{message}"

        Respond with only a number between 0.0 and 1.0 representing confidence that this is an appointment booking request.
        Examples:
        - "I want to book an appointment" → 0.95
        - "Can we schedule a meeting tomorrow?" → 0.90
        - "When are you available?" → 0.75
        - "Tell me about this property" → 0.05
        - "Thanks for the information" → 0.0

        Confidence score:
        """
        
        try:
            response = await Settings.llm.achat([ChatMessage(role="user", content=classification_prompt)])
            confidence_text = response.message.content.strip() if response.message.content else "0.0"
            
            # Extract numeric value
            confidence_match = re.search(r'(\d+\.?\d*)', confidence_text)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
                logger.debug(f"LLM classification confidence: {confidence:.2f}")
                return confidence
            else:
                logger.warning(f"Could not parse LLM confidence from: {confidence_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {e}")
            return 0.0
    
    def _identify_missing_fields(self, extracted_details: Dict[str, Any]) -> List[str]:
        """Identify which required appointment fields are missing."""
        required_fields = ['title', 'date', 'time', 'location']
        missing = []
        
        for field in required_fields:
            if field not in extracted_details or not extracted_details[field]:
                missing.append(field)
        
        # Special handling for title - can be inferred as "Meeting" if not specified
        if 'title' in missing:
            missing.remove('title')  # We can provide a default title
        
        logger.debug(f"Missing appointment fields: {missing}")
        return missing


# Global service instance
appointment_intent_detection_service = AppointmentIntentDetectionService() 