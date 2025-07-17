# /app/enhanced_intent_detection.py
"""
Enhanced Intent Detection with Error Handling

This service improves intent detection reliability by adding robust JSON parsing,
error recovery, and rule-based fallback classification when LLM fails.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from app.models import MessageType, MessageClassification, ProcessingStrategy
from app.fast_message_classifier import FastMessageClassifier
import app.intent_detection as original_intent_detection
import app.appointment_intent_detection as appointment_intent_detection

logger = logging.getLogger(__name__)


class EnhancedIntentDetectionService:
    """
    Enhanced intent detection service with robust error handling and fallbacks.
    
    Provides reliable intent classification by combining fast pattern matching,
    improved LLM-based detection with error recovery, and comprehensive fallbacks.
    """
    
    def __init__(self):
        self.fast_classifier = FastMessageClassifier()
        self.original_intent_service = original_intent_detection.IntentDetectionService()
        self.appointment_service = appointment_intent_detection.AppointmentIntentDetectionService()
        
        # Error tracking
        self.llm_failures = []
        self.json_parse_failures = []
        self.fallback_usage = []
    
    async def detect_intent_with_fallback(self, message: str, user_id: Optional[str] = None) -> MessageClassification:
        """
        Detect intent with comprehensive error handling and fallbacks.
        
        Args:
            message: User's input message
            user_id: Optional user ID for context
            
        Returns:
            MessageClassification with reliable intent detection
        """
        logger.info(f"Enhanced intent detection for: '{message[:100]}...'")
        
        try:
            # Step 1: Fast classification for common cases
            fast_result = self.fast_classifier.classify_message(message)
            
            # If fast classifier is confident, use it
            if fast_result.confidence > 0.8:
                logger.info(f"Fast classifier confident: {fast_result.message_type} ({fast_result.confidence:.2f})")
                return fast_result
            
            # Step 2: Enhanced LLM-based detection for complex cases
            llm_result = await self._enhanced_llm_detection(message, user_id)
            
            # Step 3: Combine results intelligently
            combined_result = self._combine_detection_results(fast_result, llm_result, message)
            
            logger.info(f"Enhanced detection result: {combined_result.message_type} (confidence: {combined_result.confidence:.2f})")
            return combined_result
            
        except Exception as e:
            logger.error(f"Critical error in enhanced intent detection: {e}")
            return self._emergency_fallback_classification(message)
    
    async def _enhanced_llm_detection(self, message: str, user_id: Optional[str]) -> Optional[MessageClassification]:
        """Enhanced LLM detection with robust error handling."""
        
        try:
            # Try appointment detection first (more specific)
            appointment_intent = await self.appointment_service.detect_appointment_intent(message)
            if appointment_intent.is_appointment_request and appointment_intent.confidence > 0.6:
                return MessageClassification(
                    message_type=MessageType.APPOINTMENT_REQUEST,
                    confidence=appointment_intent.confidence,
                    processing_strategy=ProcessingStrategy.APPOINTMENT_WORKFLOW,
                    estimated_response_time=3000.0,
                    requires_index=False,
                    reasoning="LLM appointment detection"
                )
            
            # Try property recommendation detection
            recommendation_intent = await self.original_intent_service.detect_recommendation_intent(message)
            if recommendation_intent.is_recommendation_request and recommendation_intent.confidence > 0.6:
                return MessageClassification(
                    message_type=MessageType.PROPERTY_SEARCH,
                    confidence=recommendation_intent.confidence,
                    processing_strategy=ProcessingStrategy.PROPERTY_WORKFLOW,
                    estimated_response_time=4000.0,
                    requires_index=True,
                    reasoning="LLM property detection"
                )
            
            # Try general intent classification with robust parsing
            general_result = await self._robust_general_intent_classification(message)
            return general_result
            
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            self.llm_failures.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "message_preview": message[:50]
            })
            return None
    
    async def _robust_general_intent_classification(self, message: str) -> Optional[MessageClassification]:
        """Robust general intent classification with improved error handling."""
        
        from llama_index.core.llms import ChatMessage
        from llama_index.core import Settings
        
        classification_prompt = f"""
        Classify this user message into one of these categories with confidence scoring.
        
        Categories:
        - greeting: Hello, hi, good morning, how are you
        - thank_you: Thanks, thank you, appreciate it
        - help_request: Help, assist, what can you do, how does this work
        - conversational: General chat, small talk, casual conversation
        - property_search: Looking for properties, apartments, rentals
        - appointment_request: Want to book, schedule, meet, arrange appointment
        - unknown: Unclear or ambiguous intent
        
        Message: "{message}"
        
        Respond with ONLY a valid JSON object:
        {{
            "category": "one_of_the_categories_above",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        
        Example responses:
        {{"category": "greeting", "confidence": 0.95, "reasoning": "Clear greeting message"}}
        {{"category": "property_search", "confidence": 0.80, "reasoning": "Looking for rental property"}}
        """
        
        try:
            response = await Settings.llm.achat([ChatMessage(role="user", content=classification_prompt)])
            response_text = response.message.content.strip() if response.message.content else ""
            
            # Enhanced JSON parsing with multiple recovery strategies
            parsed_result = self._robust_json_parse(response_text, message)
            
            if parsed_result:
                return self._convert_to_message_classification(parsed_result)
            else:
                logger.warning(f"Failed to parse LLM response: {response_text[:100]}")
                return None
                
        except Exception as e:
            logger.error(f"General intent classification failed: {e}")
            return None
    
    def _robust_json_parse(self, response_text: str, original_message: str) -> Optional[Dict[str, Any]]:
        """Robust JSON parsing with multiple recovery strategies."""
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from text
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response_text)
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Parse individual fields
        try:
            category_match = re.search(r'"category":\s*"([^"]+)"', response_text)
            confidence_match = re.search(r'"confidence":\s*([\d.]+)', response_text)
            reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', response_text)
            
            if category_match:
                result = {
                    "category": category_match.group(1),
                    "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Partial parse"
                }
                return result
        except Exception:
            pass
        
        # Strategy 4: Pattern-based extraction
        category_keywords = {
            "greeting": ["greeting", "hello", "hi"],
            "thank_you": ["thank", "thanks", "appreciate"],
            "help_request": ["help", "assist", "support"],
            "conversational": ["conversational", "chat", "talk"],
            "property_search": ["property", "search", "find", "apartment"],
            "appointment_request": ["appointment", "schedule", "book", "meet"]
        }
        
        response_lower = response_text.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return {
                    "category": category,
                    "confidence": 0.4,  # Lower confidence for pattern matching
                    "reasoning": "Pattern extraction from malformed response"
                }
        
        # Log failure for monitoring
        self.json_parse_failures.append({
            "timestamp": datetime.now().isoformat(),
            "response_text": response_text[:200],
            "original_message": original_message[:100]
        })
        
        return None
    
    def _convert_to_message_classification(self, parsed_result: Dict[str, Any]) -> MessageClassification:
        """Convert parsed LLM result to MessageClassification."""
        
        category = parsed_result.get("category", "unknown")
        confidence = float(parsed_result.get("confidence", 0.5))
        reasoning = parsed_result.get("reasoning", "LLM classification")
        
        # Map LLM categories to MessageType enum
        category_mapping = {
            "greeting": MessageType.GREETING,
            "thank_you": MessageType.THANK_YOU,
            "help_request": MessageType.HELP_REQUEST,
            "conversational": MessageType.CONVERSATIONAL,
            "property_search": MessageType.PROPERTY_SEARCH,
            "appointment_request": MessageType.APPOINTMENT_REQUEST,
            "unknown": MessageType.UNKNOWN
        }
        
        message_type = category_mapping.get(category, MessageType.UNKNOWN)
        
        # Determine processing strategy
        if message_type in [MessageType.GREETING, MessageType.THANK_YOU, MessageType.CONVERSATIONAL]:
            strategy = ProcessingStrategy.QUICK_RESPONSE
            estimated_time = 1000.0
            requires_index = False
        elif message_type == MessageType.HELP_REQUEST:
            strategy = ProcessingStrategy.QUICK_RESPONSE
            estimated_time = 1500.0
            requires_index = False
        elif message_type == MessageType.PROPERTY_SEARCH:
            strategy = ProcessingStrategy.PROPERTY_WORKFLOW
            estimated_time = 4000.0
            requires_index = True
        elif message_type == MessageType.APPOINTMENT_REQUEST:
            strategy = ProcessingStrategy.APPOINTMENT_WORKFLOW
            estimated_time = 3000.0
            requires_index = False
        else:
            strategy = ProcessingStrategy.FALLBACK_RESPONSE
            estimated_time = 2000.0
            requires_index = False
        
        return MessageClassification(
            message_type=message_type,
            confidence=confidence,
            processing_strategy=strategy,
            estimated_response_time=estimated_time,
            requires_index=requires_index,
            reasoning=reasoning
        )
    
    def _combine_detection_results(self, fast_result: MessageClassification, 
                                 llm_result: Optional[MessageClassification], 
                                 message: str) -> MessageClassification:
        """Intelligently combine fast and LLM detection results."""
        
        # If LLM failed, use fast result with fallback reasoning
        if not llm_result:
            fast_result.reasoning = f"Fast classification (LLM failed): {fast_result.reasoning}"
            self.fallback_usage.append({
                "timestamp": datetime.now().isoformat(),
                "reason": "LLM failure",
                "fast_result": fast_result.message_type.value,
                "message_preview": message[:50]
            })
            return fast_result
        
        # If both agree, use higher confidence
        if fast_result.message_type == llm_result.message_type:
            best_confidence = max(fast_result.confidence, llm_result.confidence)
            result = fast_result if fast_result.confidence >= llm_result.confidence else llm_result
            result.confidence = best_confidence
            result.reasoning = f"Fast + LLM agreement: {result.reasoning}"
            return result
        
        # If they disagree, use the more confident one
        if fast_result.confidence > llm_result.confidence:
            fast_result.reasoning = f"Fast classification (higher confidence): {fast_result.reasoning}"
            return fast_result
        else:
            llm_result.reasoning = f"LLM classification (higher confidence): {llm_result.reasoning}"
            return llm_result
    
    def _emergency_fallback_classification(self, message: str) -> MessageClassification:
        """Emergency fallback when all detection methods fail."""
        
        # Basic keyword-based emergency classification
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hi", "hello", "hey", "good morning"]):
            message_type = MessageType.GREETING
            strategy = ProcessingStrategy.QUICK_RESPONSE
            estimated_time = 1000.0
        elif any(word in message_lower for word in ["thank", "thanks", "appreciate"]):
            message_type = MessageType.THANK_YOU
            strategy = ProcessingStrategy.QUICK_RESPONSE
            estimated_time = 1000.0
        elif any(word in message_lower for word in ["property", "apartment", "rent", "listing"]):
            message_type = MessageType.PROPERTY_SEARCH
            strategy = ProcessingStrategy.PROPERTY_WORKFLOW
            estimated_time = 4000.0
        elif any(word in message_lower for word in ["appointment", "schedule", "book", "meet"]):
            message_type = MessageType.APPOINTMENT_REQUEST
            strategy = ProcessingStrategy.APPOINTMENT_WORKFLOW
            estimated_time = 3000.0
        else:
            message_type = MessageType.UNKNOWN
            strategy = ProcessingStrategy.FALLBACK_RESPONSE
            estimated_time = 2000.0
        
        return MessageClassification(
            message_type=message_type,
            confidence=0.3,  # Low confidence for emergency fallback
            processing_strategy=strategy,
            estimated_response_time=estimated_time,
            requires_index=(message_type == MessageType.PROPERTY_SEARCH),
            reasoning="Emergency fallback classification"
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "llm_failures": {
                "count": len(self.llm_failures),
                "recent": self.llm_failures[-5:] if self.llm_failures else []
            },
            "json_parse_failures": {
                "count": len(self.json_parse_failures),
                "recent": self.json_parse_failures[-5:] if self.json_parse_failures else []
            },
            "fallback_usage": {
                "count": len(self.fallback_usage),
                "recent": self.fallback_usage[-5:] if self.fallback_usage else []
            },
            "fast_classifier_stats": self.fast_classifier.get_performance_stats()
        } 