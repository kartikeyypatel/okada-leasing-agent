# /app/fast_message_classifier.py
"""
Fast Message Classifier for Performance Optimization

This service quickly categorizes user messages using rule-based pattern matching
to avoid expensive LLM operations for simple conversational messages.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from app.models import MessageType, MessageClassification, ProcessingStrategy, ConversationalContext

logger = logging.getLogger(__name__)


class FastMessageClassifier:
    """
    Service for fast message classification using pattern matching.
    
    Provides quick categorization of user messages to determine appropriate
    processing strategy and avoid expensive operations for simple conversations.
    """
    
    # High-confidence greeting patterns
    GREETING_PATTERNS = [
        r'\b(hi|hello|hey|hiya|greetings|good\s+(morning|afternoon|evening|day))\b',
        r'\b(what\'s\s+up|how\s+(are\s+you|ya\s+doing)|how\s+do\s+you\s+do)\b',
        r'\b(nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you)\b',
        r'^\s*(hi|hello|hey)\s*[!.,]*\s*$',
        r'\b(howdy|salutations|aloha)\b'
    ]
    
    # Thank you and appreciation patterns
    THANK_YOU_PATTERNS = [
        r'\b(thank\s+you|thanks|thx|ty|appreciate|grateful)\b',
        r'\b(cheers|much\s+appreciated|awesome|perfect|great)\b',
        r'\b(that\'s\s+helpful|very\s+helpful|exactly\s+what\s+i\s+needed)\b',
        r'\b(excellent|fantastic|wonderful|amazing)\s+(help|service|response)\b'
    ]
    
    # Help request patterns
    HELP_PATTERNS = [
        r'\b(help|assist|support|guide|explain|how\s+do\s+i)\b',
        r'\b(can\s+you\s+help|need\s+assistance|show\s+me\s+how)\b',
        r'\b(what\s+can\s+you\s+do|what\s+are\s+your\s+capabilities)\b',
        r'\b(how\s+does\s+this\s+work|getting\s+started)\b'
    ]
    
    # Property search patterns (high confidence indicators)
    PROPERTY_SEARCH_PATTERNS = [
        r'\b(find|search|look|show)\s+(me\s+)?(properties|apartments|listings|places)\b',
        r'\b(property|apartment|house|listing|rental)\s+(at|on|in|near)\b',
        r'\b(rent|lease|available|for\s+rent|to\s+rent)\b',
        r'\b\d+\s+(bedroom|bed|br)\b',
        r'\$\d+(\,\d+)*(\.\d+)?\s*(per\s+month|monthly|rent)',
        r'\b(square\s+feet|sq\s*ft|sf)\b',
        r'\b\d+\s+[NSEW]?\s*\w+\s+(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive)\b'
    ]
    
    # Appointment request patterns
    APPOINTMENT_PATTERNS = [
        r'\b(book|schedule|set\s+up|arrange|make)\s+(an?\s+)?(appointment|meeting|call)\b',
        r'\b(can\s+we\s+meet|let\'s\s+meet|available|free)\b',
        r'\b(calendar|schedule|appointment|meeting)\b'
    ]
    
    # General conversational patterns
    CONVERSATIONAL_PATTERNS = [
        r'\b(how\s+are\s+you|what\'s\s+new|how\'s\s+it\s+going)\b',
        r'\b(nice\s+weather|good\s+day|beautiful\s+day)\b',
        r'\b(just\s+chatting|just\s+saying\s+hi|checking\s+in)\b',
        r'\b(have\s+a\s+good\s+day|take\s+care|bye|goodbye)\b'
    ]
    
    def __init__(self):
        # Compile patterns for better performance
        self.compiled_greetings = [re.compile(pattern, re.IGNORECASE) for pattern in self.GREETING_PATTERNS]
        self.compiled_thanks = [re.compile(pattern, re.IGNORECASE) for pattern in self.THANK_YOU_PATTERNS]
        self.compiled_help = [re.compile(pattern, re.IGNORECASE) for pattern in self.HELP_PATTERNS]
        self.compiled_property = [re.compile(pattern, re.IGNORECASE) for pattern in self.PROPERTY_SEARCH_PATTERNS]
        self.compiled_appointment = [re.compile(pattern, re.IGNORECASE) for pattern in self.APPOINTMENT_PATTERNS]
        self.compiled_conversational = [re.compile(pattern, re.IGNORECASE) for pattern in self.CONVERSATIONAL_PATTERNS]
        
        # Performance tracking
        self.classification_times = []
    
    def classify_message(self, message: str, user_context: Optional[ConversationalContext] = None) -> MessageClassification:
        """
        Classify a user message quickly using pattern matching.
        
        Args:
            message: User's input message
            user_context: Optional user context for personalization
            
        Returns:
            MessageClassification with type, confidence, and processing strategy
        """
        start_time = time.time()
        
        try:
            # Clean and normalize message
            normalized_message = self._normalize_message(message)
            
            # Pattern-based classification
            classification_results = self._run_all_classifications(normalized_message)
            
            # Select best classification
            best_classification = self._select_best_classification(classification_results, user_context)
            
            # Track performance
            duration_ms = (time.time() - start_time) * 1000
            self.classification_times.append(duration_ms)
            
            logger.debug(f"Fast classification completed in {duration_ms:.2f}ms: {best_classification.message_type} (confidence: {best_classification.confidence:.2f})")
            
            return best_classification
            
        except Exception as e:
            logger.error(f"Error in fast message classification: {e}")
            # Return fallback classification
            return MessageClassification(
                message_type=MessageType.UNKNOWN,
                confidence=0.0,
                processing_strategy=ProcessingStrategy.FALLBACK_RESPONSE,
                estimated_response_time=5000.0,  # 5 seconds fallback
                requires_index=False,
                reasoning="Classification error - using fallback"
            )
    
    def is_greeting(self, message: str) -> bool:
        """Quick check if message is a greeting."""
        normalized = self._normalize_message(message)
        return any(pattern.search(normalized) for pattern in self.compiled_greetings)
    
    def is_property_query(self, message: str) -> bool:
        """Quick check if message is property-related."""
        normalized = self._normalize_message(message)
        return any(pattern.search(normalized) for pattern in self.compiled_property)
    
    def is_appointment_request(self, message: str) -> bool:
        """Quick check if message is appointment-related."""
        normalized = self._normalize_message(message)
        return any(pattern.search(normalized) for pattern in self.compiled_appointment)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the classifier."""
        if not self.classification_times:
            return {"message": "No classifications performed yet"}
        
        times = self.classification_times[-100:]  # Last 100 classifications
        return {
            "total_classifications": len(self.classification_times),
            "recent_classifications": len(times),
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "target_met": sum(1 for t in times if t < 100) / len(times) * 100  # % under 100ms
        }
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message for pattern matching."""
        if not message:
            return ""
        
        # Basic normalization
        normalized = message.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation clusters but keep single punctuation
        normalized = re.sub(r'[!.?]{2,}', '.', normalized)
        
        return normalized
    
    def _is_greeting(self, normalized_message: str) -> bool:
        """Check if message is a greeting."""
        return any(pattern.search(normalized_message) for pattern in self.compiled_greetings)
    
    def _is_thank_you(self, normalized_message: str) -> bool:
        """Check if message is a thank you."""
        return any(pattern.search(normalized_message) for pattern in self.compiled_thanks)
    
    def _is_help_request(self, normalized_message: str) -> bool:
        """Check if message is a help request."""
        return any(pattern.search(normalized_message) for pattern in self.compiled_help)
    
    def _is_appointment_request(self, normalized_message: str) -> bool:
        """Check if message is an appointment request."""
        return any(pattern.search(normalized_message) for pattern in self.compiled_appointment)
    
    def _is_conversational(self, normalized_message: str) -> bool:
        """Check if message is conversational."""
        return any(pattern.search(normalized_message) for pattern in self.compiled_conversational)
    
    def _run_all_classifications(self, normalized_message: str) -> List[Tuple[MessageType, float, str]]:
        """Run all classification methods and return results."""
        results = []
        
        # Test for greetings
        if self._is_greeting(normalized_message):
            results.append((MessageType.GREETING, 0.9, "greeting patterns"))
        
        # Test for thank you messages
        if self._is_thank_you(normalized_message):
            results.append((MessageType.THANK_YOU, 0.9, "thank you patterns"))
        
        # Test for help requests
        if self._is_help_request(normalized_message):
            results.append((MessageType.HELP_REQUEST, 0.8, "help request patterns"))
        
        # Test for appointment requests
        if self._is_appointment_request(normalized_message):
            results.append((MessageType.APPOINTMENT_REQUEST, 0.8, "appointment patterns"))
        
        # Test for property searches with distinction between direct queries and recommendations
        property_confidence, property_reasoning = self._classify_property_search(normalized_message)
        if property_confidence > 0.3:
            # Check if this is a direct query (like "top 3 properties") vs recommendation request
            if "direct query pattern" in property_reasoning or "top N pattern" in property_reasoning:
                results.append((MessageType.DIRECT_PROPERTY_QUERY, property_confidence, property_reasoning))
            else:
                results.append((MessageType.PROPERTY_SEARCH, property_confidence, property_reasoning))
        
        # Test for conversational messages
        if self._is_conversational(normalized_message):
            results.append((MessageType.CONVERSATIONAL, 0.6, "conversational patterns"))
        
        # If no strong classification, mark as unknown
        if not results or max(result[1] for result in results) < 0.4:
            results.append((MessageType.UNKNOWN, 0.3, "no clear patterns found"))
        
        return results
    
    def _find_pattern_matches(self, message: str, compiled_patterns: List[re.Pattern], 
                            original_patterns: List[str]) -> List[str]:
        """Find all matching patterns in a message."""
        matches = []
        for i, pattern in enumerate(compiled_patterns):
            if pattern.search(message):
                matches.append(original_patterns[i])
        return matches
    
    def _select_best_classification(self, results: List[Tuple[MessageType, float, str]], 
                                  user_context: Optional[ConversationalContext]) -> MessageClassification:
        """Select the best classification from results."""
        if not results:
            return MessageClassification(
                message_type=MessageType.UNKNOWN,
                confidence=0.0,
                processing_strategy=ProcessingStrategy.FALLBACK_RESPONSE,
                estimated_response_time=5000.0,
                requires_index=False,
                reasoning="No patterns matched"
            )
        
        # Sort by confidence
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        best_type, best_confidence, best_reasoning = sorted_results[0]
        
        # Determine processing strategy and estimated time
        strategy, estimated_time, requires_index = self._determine_processing_strategy(best_type, best_confidence)
        
        # Apply user context adjustments
        if user_context:
            strategy, estimated_time = self._apply_user_context(strategy, estimated_time, user_context)
        
        return MessageClassification(
            message_type=best_type,
            confidence=best_confidence,
            processing_strategy=strategy,
            estimated_response_time=estimated_time,
            requires_index=requires_index,
            reasoning=best_reasoning
        )
    
    def _determine_processing_strategy(self, message_type: MessageType, 
                                     confidence: float) -> Tuple[ProcessingStrategy, float, bool]:
        """Determine processing strategy based on message type and confidence."""
        
        if message_type in [MessageType.GREETING, MessageType.THANK_YOU, MessageType.CONVERSATIONAL]:
            return ProcessingStrategy.QUICK_RESPONSE, 1000.0, False  # 1 second
        
        elif message_type == MessageType.HELP_REQUEST:
            return ProcessingStrategy.QUICK_RESPONSE, 1500.0, False  # 1.5 seconds
        
        elif message_type == MessageType.DIRECT_PROPERTY_QUERY:
            # Direct property queries should use basic RAG search, not recommendation workflow
            return ProcessingStrategy.DIRECT_SEARCH, 3000.0, True  # 3 seconds, needs index
        
        elif message_type == MessageType.PROPERTY_SEARCH:
            # General property searches can trigger recommendation workflow
            if confidence > 0.8:
                return ProcessingStrategy.PROPERTY_WORKFLOW, 4000.0, True  # 4 seconds
            else:
                return ProcessingStrategy.FALLBACK_RESPONSE, 2000.0, False  # 2 seconds
        
        elif message_type == MessageType.APPOINTMENT_REQUEST:
            return ProcessingStrategy.APPOINTMENT_WORKFLOW, 3000.0, False  # 3 seconds
        
        else:  # UNKNOWN
            return ProcessingStrategy.FALLBACK_RESPONSE, 5000.0, False  # 5 seconds
    
    def _apply_user_context(self, strategy: ProcessingStrategy, estimated_time: float, 
                          context: ConversationalContext) -> Tuple[ProcessingStrategy, float]:
        """Apply user context to refine processing strategy."""
        
        # Personalization can reduce response time slightly
        if context.previous_interactions > 5:
            estimated_time *= 0.9  # 10% faster for familiar users
        
        # If user has preferences, we might be able to respond faster
        if context.user_preferences and strategy == ProcessingStrategy.PROPERTY_WORKFLOW:
            estimated_time *= 0.8  # 20% faster with known preferences
        
        return strategy, estimated_time 

    def _classify_property_search(self, normalized_message: str) -> Tuple[float, str]:
        """Classify property search requests with distinction between direct queries and recommendation requests."""
        
        # Direct property query patterns (should bypass recommendation workflow)
        direct_query_patterns = [
            r'\b(?:top|best|cheapest|most expensive|lowest|highest|largest|smallest)\s+\d*\s*(?:property|properties|apartment|apartments|listing|listings)\b',
            r'\b(?:show|list|find)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:property|properties|apartment|apartments|listing|listings)\b',
            r'\b(?:properties|apartments|listings)\s+(?:under|over|above|below)\s+\$?\d+\b',
            r'\b(?:tell\s+me\s+about|what\s+is|info\s+about|details\s+about)\s+.+(?:street|st|avenue|ave|road|rd|drive|dr|place|pl)\b',
            r'\b\d+\s+\w+\s+(?:street|st|avenue|ave|road|rd|drive|dr|place|pl)\b',  # Specific addresses
            r'\b(?:search|filter)\s+(?:for\s+)?(?:property|properties|apartment|apartments)\b',
        ]
        
        # Recommendation request patterns (should trigger recommendation workflow)
        recommendation_patterns = [
            r'\b(?:suggest|recommend)\s+(?:me\s+)?(?:a\s+|some\s+)?(?:property|properties|apartment|place)\b',
            r'\b(?:help\s+me\s+find)\s+(?:me\s+)?(?:a\s+|some\s+)?(?:property|properties|apartment|place)\b',
            r'\b(?:what\s+do\s+you\s+have|what\s+would\s+you\s+recommend)\b',
            r'\b(?:any\s+)?(?:good\s+)?(?:properties|apartments|places)\s+(?:for\s+me|you\s+suggest)\b',
            r'\b(?:looking\s+for|searching\s+for)\s+(?:a\s+)?(?:property|apartment|place)\s+(?:to\s+rent|for\s+rent)?\b',
            r'\b(?:i\s+need|i\s+want)\s+(?:a\s+|some\s+)?(?:property|apartment|place)\b'
        ]
        
        direct_query_score = 0.0
        recommendation_score = 0.0
        reasoning_parts = []
        
        # Check for direct query patterns
        for pattern in direct_query_patterns:
            if re.search(pattern, normalized_message):
                direct_query_score += 0.3
                reasoning_parts.append("direct query pattern")
        
        # Check for recommendation patterns
        for pattern in recommendation_patterns:
            if re.search(pattern, normalized_message):
                recommendation_score += 0.4  # Increased from 0.3
                reasoning_parts.append("recommendation pattern")
        
        # Additional scoring based on keywords
        property_keywords = ['property', 'properties', 'apartment', 'apartments', 'listing', 'listings', 'rent', 'rental']
        action_keywords = ['show', 'list', 'find', 'search', 'tell me', 'what is', 'top', 'best', 'cheapest']
        suggest_keywords = ['suggest', 'recommend', 'help']  # Added suggest keywords
        
        property_count = sum(1 for keyword in property_keywords if keyword in normalized_message)
        action_count = sum(1 for keyword in action_keywords if keyword in normalized_message)
        suggest_count = sum(1 for keyword in suggest_keywords if keyword in normalized_message)
        
        # Boost direct query score for action + property combinations
        if property_count > 0 and action_count > 0:
            direct_query_score += 0.4
            reasoning_parts.append("action+property keywords")
        
        # Boost recommendation score for suggest + property combinations
        if property_count > 0 and suggest_count > 0:
            recommendation_score += 0.5
            reasoning_parts.append("suggest+property keywords")
        
        # Check for specific patterns that indicate direct queries
        if re.search(r'\b(?:top|cheapest|most expensive|lowest|highest|best)\s+\d+', normalized_message):
            direct_query_score += 0.5
            reasoning_parts.append("top N pattern")
        
        if re.search(r'\$\d+', normalized_message):  # Price mentioned
            direct_query_score += 0.2
            reasoning_parts.append("price mentioned")
        
        if re.search(r'\b\d+\s+(?:bedroom|bed|bath|sqft|sf)\b', normalized_message):  # Specific features
            direct_query_score += 0.2
            reasoning_parts.append("specific features")
        
        # Address patterns strongly indicate direct queries
        if re.search(r'\b\d+\s+\w+\s+(?:street|st|avenue|ave|road|rd)\b', normalized_message):
            direct_query_score += 0.6
            reasoning_parts.append("specific address")
        
        # Determine final classification
        if direct_query_score > recommendation_score and direct_query_score > 0.5:
            # This is a direct property query, not a recommendation request
            confidence = min(0.9, direct_query_score)
            reasoning = f"Direct property query: {', '.join(reasoning_parts)}"
            return confidence, reasoning
        elif recommendation_score > 0.3:
            # This is a recommendation request
            confidence = min(0.8, recommendation_score)
            reasoning = f"Recommendation request: {', '.join(reasoning_parts)}"
            return confidence, reasoning
        else:
            # General property search
            confidence = min(0.7, max(direct_query_score, recommendation_score))
            reasoning = f"General property search: {', '.join(reasoning_parts) if reasoning_parts else 'property keywords found'}"
            return confidence, reasoning 