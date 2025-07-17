# /test/test_chatbot_performance_optimization.py
"""
Unit Tests for Chatbot Performance Optimization

Comprehensive tests for fast message classification, conversational responses,
enhanced intent detection, index health validation, and performance monitoring.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from app.fast_message_classifier import FastMessageClassifier
from app.conversational_response_handler import ConversationalResponseHandler
from app.enhanced_intent_detection import EnhancedIntentDetectionService
from app.index_health_validator import IndexHealthValidator, IndexHealthResult
from app.async_index_manager import AsyncIndexManager, RebuildStatus
from app.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from app.performance_logger import PerformanceLogger, LogLevel
from app.models import (
    MessageType, ProcessingStrategy, MessageClassification, 
    ConversationalContext, PerformanceMetrics
)


class TestFastMessageClassifier:
    """Test fast message classification functionality."""
    
    def test_greeting_detection(self):
        """Test greeting message detection."""
        classifier = FastMessageClassifier()
        
        # Test various greeting patterns
        greetings = [
            "hi",
            "hello there",
            "hey!",
            "good morning",
            "good afternoon",
            "what's up",
            "how are you"
        ]
        
        for greeting in greetings:
            assert classifier.is_greeting(greeting), f"Failed to detect greeting: {greeting}"
            
            classification = classifier.classify_message(greeting)
            assert classification.message_type == MessageType.GREETING
            assert classification.confidence > 0.7
            assert classification.processing_strategy == ProcessingStrategy.QUICK_RESPONSE
            assert not classification.requires_index
    
    def test_property_query_detection(self):
        """Test property search query detection."""
        classifier = FastMessageClassifier()
        
        property_queries = [
            "find me a 2 bedroom apartment",
            "show me properties under $3000",
            "looking for apartments near downtown",
            "any properties with parking",
            "84 Mulberry St details"
        ]
        
        for query in property_queries:
            assert classifier.is_property_query(query), f"Failed to detect property query: {query}"
            
            classification = classifier.classify_message(query)
            assert classification.message_type == MessageType.PROPERTY_SEARCH
            assert classification.processing_strategy == ProcessingStrategy.PROPERTY_WORKFLOW
            assert classification.requires_index
    
    def test_appointment_request_detection(self):
        """Test appointment request detection."""
        classifier = FastMessageClassifier()
        
        appointment_requests = [
            "book an appointment",
            "schedule a meeting",
            "can we meet tomorrow",
            "set up a call",
            "arrange a viewing"
        ]
        
        for request in appointment_requests:
            assert classifier.is_appointment_request(request), f"Failed to detect appointment: {request}"
            
            classification = classifier.classify_message(request)
            assert classification.message_type == MessageType.APPOINTMENT_REQUEST
            assert classification.processing_strategy == ProcessingStrategy.APPOINTMENT_WORKFLOW
    
    def test_thank_you_detection(self):
        """Test thank you message detection."""
        classifier = FastMessageClassifier()
        
        thank_you_messages = [
            "thank you",
            "thanks so much",
            "appreciate it",
            "that's helpful",
            "much appreciated"
        ]
        
        for message in thank_you_messages:
            classification = classifier.classify_message(message)
            assert classification.message_type == MessageType.THANK_YOU
            assert classification.processing_strategy == ProcessingStrategy.QUICK_RESPONSE
    
    def test_performance_target(self):
        """Test that classification meets performance targets (<100ms)."""
        classifier = FastMessageClassifier()
        test_messages = [
            "hello",
            "find me properties",
            "book an appointment", 
            "thank you",
            "what can you help me with"
        ]
        
        for message in test_messages:
            start_time = time.time()
            classification = classifier.classify_message(message)
            duration_ms = (time.time() - start_time) * 1000
            
            assert duration_ms < 100, f"Classification too slow: {duration_ms:.2f}ms for '{message}'"
            assert classification is not None
    
    def test_personalization_with_context(self):
        """Test personalization with user context."""
        classifier = FastMessageClassifier()
        
        context = ConversationalContext(
            user_id="test@example.com",
            user_name="John",
            previous_interactions=5,
            conversation_stage="returning"
        )
        
        classification = classifier.classify_message("hello", context)
        
        assert classification.message_type == MessageType.GREETING
        # Should have slightly faster estimated time for returning users
        assert classification.estimated_response_time < 1000.0


class TestConversationalResponseHandler:
    """Test conversational response handling."""
    
    def test_greeting_responses(self):
        """Test greeting response generation."""
        handler = ConversationalResponseHandler()
        
        greetings = ["hello", "hi", "good morning", "hey there"]
        
        for greeting in greetings:
            response = handler.handle_greeting(greeting)
            
            assert len(response) > 0
            assert "property" in response.lower()  # Should guide to properties
            assert any(emoji in response for emoji in ["👋", "🌅", "☀️", "🌤️", "🌆"])
    
    def test_thank_you_responses(self):
        """Test thank you response generation."""
        handler = ConversationalResponseHandler()
        
        response = handler.handle_thanks("thank you so much")
        
        assert len(response) > 0
        assert "welcome" in response.lower()
        assert "property" in response.lower()  # Should offer follow-up help
    
    def test_help_responses(self):
        """Test help request responses."""
        handler = ConversationalResponseHandler()
        
        response = handler.handle_help_request("what can you help me with")
        
        assert len(response) > 100  # Should be comprehensive
        assert "property" in response.lower()
        assert "search" in response.lower()
        assert "appointment" in response.lower()
        assert any(bullet in response for bullet in ["•", "✨", "🏠", "📅"])
    
    def test_personalization(self):
        """Test response personalization with user context."""
        handler = ConversationalResponseHandler()
        
        context = ConversationalContext(
            user_id="test@example.com",
            user_name="Alice",
            previous_interactions=3
        )
        
        response = handler.handle_greeting("hello", context)
        
        assert "Alice" in response or "back" in response.lower()
    
    @pytest.mark.asyncio
    async def test_user_context_retrieval(self):
        """Test user context retrieval from CRM."""
        handler = ConversationalResponseHandler()
        
        with patch('app.crm.get_user_profile') as mock_crm:
            mock_crm.return_value = {
                'name': 'Test User',
                'interaction_count': 5,
                'preferences': {'location': 'downtown'}
            }
            
            context = await handler.get_user_context("test@example.com")
            
            assert context is not None
            assert context.user_name == 'Test User'
            assert context.previous_interactions == 5
            assert context.conversation_stage == "returning"


class TestEnhancedIntentDetection:
    """Test enhanced intent detection with error handling."""
    
    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """Test fallback to fast classification when LLM fails."""
        service = EnhancedIntentDetectionService()
        
        with patch.object(service.original_intent_service, 'detect_recommendation_intent') as mock_intent:
            mock_intent.side_effect = Exception("LLM API error")
            
            classification = await service.detect_intent_with_fallback("hello there")
            
            # Should fall back to fast classification
            assert classification.message_type == MessageType.GREETING
            assert classification.confidence > 0.0
            assert "fast classification" in classification.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_json_parsing_recovery(self):
        """Test robust JSON parsing with malformed responses."""
        service = EnhancedIntentDetectionService()
        
        # Test various malformed JSON responses
        malformed_responses = [
            '{"category": "greeting", "confidence": 0.95}',  # Valid
            'category: "greeting", confidence: 0.95',        # Missing braces
            '{"category": "greeting", "confidence": 0.95, "extra": }',  # Malformed
            'The category is "greeting" with confidence 0.95',  # Text with extractable data
            'greeting, 0.95, looks like a greeting message'     # Minimal extractable data
        ]
        
        for response_text in malformed_responses:
            result = service._robust_json_parse(response_text, "hello")
            # Should either parse successfully or return None gracefully
            assert result is None or isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_combining_fast_and_llm_results(self):
        """Test intelligent combination of fast and LLM results."""
        service = EnhancedIntentDetectionService()
        
        # Mock fast classification
        fast_result = MessageClassification(
            message_type=MessageType.GREETING,
            confidence=0.9,
            processing_strategy=ProcessingStrategy.QUICK_RESPONSE,
            estimated_response_time=1000.0,
            requires_index=False,
            reasoning="Pattern match"
        )
        
        # Mock LLM result
        llm_result = MessageClassification(
            message_type=MessageType.GREETING,
            confidence=0.85,
            processing_strategy=ProcessingStrategy.QUICK_RESPONSE,
            estimated_response_time=1500.0,
            requires_index=False,
            reasoning="LLM classification"
        )
        
        # Test agreement case
        combined = service._combine_detection_results(fast_result, llm_result, "hello")
        assert combined.message_type == MessageType.GREETING
        assert combined.confidence == 0.9  # Should use higher confidence
        assert "agreement" in combined.reasoning.lower()


class TestIndexHealthValidator:
    """Test index health validation system."""
    
    @pytest.mark.asyncio
    async def test_healthy_index_validation(self):
        """Test validation of healthy index."""
        validator = IndexHealthValidator()
        
        with patch('app.rag.get_user_index') as mock_get_index:
            mock_index = Mock()
            mock_index.as_retriever.return_value = Mock()
            mock_index.docstore.docs = {'doc1': Mock(), 'doc2': Mock()}
            mock_get_index.return_value = mock_index
            
            with patch('app.rag.get_fusion_retriever') as mock_fusion:
                mock_fusion.return_value = Mock()
                
                result = await validator.validate_user_index_health("test@example.com")
                
                assert result.is_healthy
                assert result.index_exists
                assert result.retriever_functional
                assert result.document_count == 2
                assert len(result.issues_found) == 0
    
    @pytest.mark.asyncio
    async def test_unhealthy_index_validation(self):
        """Test validation of unhealthy index."""
        validator = IndexHealthValidator()
        
        with patch('app.rag.get_user_index') as mock_get_index:
            mock_get_index.return_value = None  # No index
            
            result = await validator.validate_user_index_health("test@example.com")
            
            assert not result.is_healthy
            assert not result.index_exists
            assert not result.retriever_functional
            assert len(result.issues_found) > 0
            assert "does not exist" in " ".join(result.issues_found)
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test validation result caching."""
        validator = IndexHealthValidator()
        
        with patch('app.rag.get_user_index') as mock_get_index:
            mock_index = Mock()
            mock_index.as_retriever.return_value = Mock()
            mock_index.docstore.docs = {}
            mock_get_index.return_value = mock_index
            
            # First call
            result1 = await validator.validate_user_index_health("test@example.com")
            # Second call (should use cache)
            result2 = await validator.validate_user_index_health("test@example.com")
            
            # Should only call get_user_index once due to caching
            assert mock_get_index.call_count == 1
            assert result1.user_id == result2.user_id
    
    @pytest.mark.asyncio
    async def test_auto_fix_functionality(self):
        """Test automatic fixing of common issues."""
        validator = IndexHealthValidator()
        
        with patch('app.rag.clear_user_index') as mock_clear:
            with patch('app.rag.build_user_index') as mock_build:
                with patch('os.path.exists') as mock_exists:
                    with patch('os.listdir') as mock_listdir:
                        mock_clear.return_value = True
                        mock_build.return_value = Mock()
                        mock_exists.return_value = True
                        mock_listdir.return_value = ['test.csv']
                        
                        fix_report = await validator.auto_fix_common_issues("test@example.com")
                        
                        assert "rebuild_index" in fix_report["fixes_attempted"]
                        assert fix_report["final_health"]["is_healthy"] or mock_build.called


class TestAsyncIndexManager:
    """Test asynchronous index management."""
    
    @pytest.mark.asyncio
    async def test_async_rebuild_initiation(self):
        """Test starting async index rebuild."""
        manager = AsyncIndexManager()
        
        with patch('os.path.exists') as mock_exists:
            with patch('os.listdir') as mock_listdir:
                mock_exists.return_value = True
                mock_listdir.return_value = ['test.csv']
                
                rebuild_op = await manager.rebuild_user_index_async("test@example.com")
                
                assert rebuild_op.user_id == "test@example.com"
                assert rebuild_op.status == RebuildStatus.PENDING
                assert manager.is_user_rebuilding("test@example.com")
    
    def test_fallback_responses(self):
        """Test fallback response generation."""
        manager = AsyncIndexManager()
        
        # Simulate active rebuild
        manager.rebuilding_users.add("test@example.com")
        
        response = manager.get_fallback_response("test@example.com")
        
        assert len(response) > 0
        assert "updating" in response.lower() or "refreshing" in response.lower()
        assert "property" in response.lower()
    
    @pytest.mark.asyncio
    async def test_rebuild_coordination(self):
        """Test that multiple rebuild requests for same user are coordinated."""
        manager = AsyncIndexManager()
        
        with patch('os.path.exists') as mock_exists:
            with patch('os.listdir') as mock_listdir:
                mock_exists.return_value = True
                mock_listdir.return_value = ['test.csv']
                
                # Start first rebuild
                rebuild_op1 = await manager.rebuild_user_index_async("test@example.com")
                
                # Try to start second rebuild for same user
                rebuild_op2 = await manager.rebuild_user_index_async("test@example.com")
                
                # Should return the same operation
                assert rebuild_op1.user_id == rebuild_op2.user_id
                assert rebuild_op1.started_at == rebuild_op2.started_at


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_normal_operation(self):
        """Test circuit breaker in normal operation."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1.0)
        breaker = CircuitBreaker("test", config)
        
        async def successful_operation():
            return "success"
        
        result = await breaker.call(successful_operation)
        
        assert result.success
        assert result.result == "success"
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=0.1)
        breaker = CircuitBreaker("test", config)
        
        async def slow_operation():
            await asyncio.sleep(0.5)  # Longer than timeout
            return "success"
        
        result = await breaker.call(slow_operation)
        
        assert not result.success
        assert result.was_timeout
        assert "timed out" in str(result.error)
    
    @pytest.mark.asyncio
    async def test_circuit_opening(self):
        """Test circuit opening after failures."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1.0)
        breaker = CircuitBreaker("test", config)
        
        async def failing_operation():
            raise Exception("Operation failed")
        
        # First failure
        result1 = await breaker.call(failing_operation)
        assert not result1.success
        assert breaker.state == CircuitState.CLOSED
        
        # Second failure - should open circuit
        result2 = await breaker.call(failing_operation)
        assert not result2.success
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self):
        """Test fallback mechanism when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=1.0)
        breaker = CircuitBreaker("test", config)
        
        async def failing_operation():
            raise Exception("Always fails")
        
        async def fallback_operation():
            return "fallback result"
        
        # Trigger circuit open
        await breaker.call(failing_operation)
        
        # Now circuit should be open, fallback should be used
        result = await breaker.call(failing_operation, fallback_operation)
        
        assert result.success
        assert result.result == "fallback result"


class TestPerformanceLogger:
    """Test performance logging functionality."""
    
    def test_slow_operation_logging(self):
        """Test logging of slow operations."""
        logger = PerformanceLogger()
        
        logger.log_slow_operation(
            operation="test_operation",
            duration_ms=3000.0,
            threshold_ms=1000.0,
            user_id="test@example.com",
            message_type=MessageType.GREETING
        )
        
        assert len(logger.log_entries) == 1
        entry = logger.log_entries[0]
        assert entry.operation == "test_operation"
        assert entry.duration_ms == 3000.0
        assert entry.level == LogLevel.ERROR  # 3x threshold
    
    def test_intent_detection_failure_logging(self):
        """Test logging of intent detection failures."""
        logger = PerformanceLogger()
        
        error = ValueError("JSON parsing failed")
        logger.log_intent_detection_failure(
            error=error,
            message="test message",
            user_id="test@example.com",
            fallback_used=True
        )
        
        assert len(logger.log_entries) == 1
        entry = logger.log_entries[0]
        assert entry.operation == "intent_detection"
        assert entry.error_type == "ValueError"
        assert entry.context["fallback_used"] is True
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        logger = PerformanceLogger()
        
        # Add some test log entries
        logger.log_slow_operation("op1", 2000.0, 1000.0)
        logger.log_slow_operation("op2", 1500.0, 1000.0)
        
        summary = logger.get_performance_summary(hours=1)
        
        assert summary["total_log_entries"] == 2
        assert "op1" in summary["operation_counts"]
        assert "op2" in summary["operation_counts"]
        assert len(summary["avg_durations_ms"]) == 2
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        logger = PerformanceLogger()
        
        # Add some errors
        logger.log_intent_detection_failure(ValueError("Error 1"), "msg1")
        logger.log_intent_detection_failure(ValueError("Error 2"), "msg2")
        logger.log_intent_detection_failure(TypeError("Error 3"), "msg3")
        
        analysis = logger.get_error_analysis()
        
        assert analysis["total_error_types"] > 0
        assert "ValueError" in str(analysis["most_common_errors"])
        assert len(analysis["recent_errors"]) == 3


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_complete_greeting_flow(self):
        """Test complete flow for greeting message."""
        # This would test the full flow from classification to response
        
        classifier = FastMessageClassifier()
        handler = ConversationalResponseHandler()
        
        # Classify message
        classification = classifier.classify_message("hello there")
        
        # Generate response
        response = handler.get_response_for_type(
            classification.message_type,
            "hello there"
        )
        
        assert classification.message_type == MessageType.GREETING
        assert classification.processing_strategy == ProcessingStrategy.QUICK_RESPONSE
        assert len(response) > 0
        assert "property" in response.lower()
    
    @pytest.mark.asyncio
    async def test_property_search_with_health_check(self):
        """Test property search flow with index health validation."""
        
        classifier = FastMessageClassifier()
        validator = IndexHealthValidator()
        
        # Classify as property search
        classification = classifier.classify_message("find me 2 bedroom apartments")
        assert classification.requires_index
        
        # Mock healthy index
        with patch('app.rag.get_user_index') as mock_get_index:
            mock_index = Mock()
            mock_index.as_retriever.return_value = Mock()
            mock_index.docstore.docs = {'doc1': Mock()}
            mock_get_index.return_value = mock_index
            
            with patch('app.rag.get_fusion_retriever') as mock_fusion:
                mock_fusion.return_value = Mock()
                
                # Validate index health
                health_result = await validator.validate_user_index_health("test@example.com")
                
                assert health_result.is_healthy
                # Would proceed with property search...
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across different components."""
        
        classifier = FastMessageClassifier()
        logger = PerformanceLogger()
        
        # Simulate multiple operations
        start_time = time.time()
        classification = classifier.classify_message("hello")
        duration_ms = (time.time() - start_time) * 1000
        
        # Log performance
        logger.log_slow_operation(
            operation="message_classification",
            duration_ms=duration_ms,
            threshold_ms=100.0,
            message_type=classification.message_type
        )
        
        # Check if logged correctly
        assert len(logger.log_entries) == 1 if duration_ms > 100.0 else 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 