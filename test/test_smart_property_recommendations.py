# /test/test_smart_property_recommendations.py
"""
Comprehensive Tests for Smart Property Recommendations

This module contains integration and unit tests for the Smart Property Recommendations
feature, covering intent detection, user context analysis, conversation management,
property recommendation generation, and workflow orchestration.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict
from datetime import datetime
from fastapi.testclient import TestClient

# Import the modules to test
from app.intent_detection import IntentDetectionService
from app.user_context_analyzer import UserContextAnalyzer
from app.conversation_state_manager import ConversationStateManager
from app.property_recommendation_engine import PropertyRecommendationEngine
from app.recommendation_workflow_manager import RecommendationWorkflowManager
from app.recommendation_error_handler import RecommendationErrorHandler, RecommendationErrorType, RecommendationErrorContext

from app.models import (
    RecommendationIntent, 
    UserContext, 
    ConversationSession, 
    PropertyRecommendation,
    RecommendationResult,
    ConversationState
)
from app.main import app


class TestIntentDetection:
    """Test the Intent Detection Service."""
    
    @pytest.fixture
    def intent_service(self):
        return IntentDetectionService()
    
    @pytest.mark.asyncio
    async def test_recommendation_intent_detection_positive(self, intent_service):
        """Test detection of clear recommendation requests."""
        test_messages = [
            "Suggest me a property",
            "Find me an apartment",
            "Any good listings for me?",
            "What properties do you have?",
            "Show me your best options",
            "I'm looking for a place to rent"
        ]
        
        for message in test_messages:
            intent = await intent_service.detect_recommendation_intent(message)
            assert intent.is_recommendation_request, f"Failed to detect recommendation intent in: '{message}'"
            assert intent.confidence > 0.5, f"Low confidence for obvious recommendation request: '{message}'"
    
    @pytest.mark.asyncio
    async def test_recommendation_intent_detection_negative(self, intent_service):
        """Test rejection of non-recommendation requests."""
        test_messages = [
            "Tell me about 123 Main St",  # Specific property inquiry
            "What is the rent for the 2-bedroom?",  # Specific question
            "How do I schedule a viewing?",  # Scheduling
            "Thank you",  # General chat
            "How are you?",  # General chat
            "What's the weather like?"  # Unrelated
        ]
        
        for message in test_messages:
            intent = await intent_service.detect_recommendation_intent(message)
            assert not intent.is_recommendation_request, f"False positive for: '{message}'"
    
    def test_preference_extraction(self, intent_service):
        """Test extraction of initial preferences from messages."""
        test_cases = [
            ("Find me a 2-bedroom apartment under $3000", ["size", "budget"]),
            ("Looking for a place in downtown with parking", ["location", "features"]),
            ("Suggest a property with a modern kitchen", ["features"]),
            ("Any apartments around $2500 per month?", ["budget"])
        ]
        
        for message, expected_categories in test_cases:
            preferences = intent_service.extract_initial_preferences(message)
            for category in expected_categories:
                assert category in preferences, f"Failed to extract {category} from: '{message}'"


class TestUserContextAnalyzer:
    """Test the User Context Analyzer."""
    
    @pytest.fixture
    def context_analyzer(self):
        return UserContextAnalyzer()
    
    @pytest.mark.asyncio
    async def test_user_context_analysis(self, context_analyzer):
        """Test user context analysis with mock data."""
        user_id = "test@example.com"
        
        # Mock the CRM and history functions
        with patch('app.user_context_analyzer.get_user_by_email') as mock_get_user, \
             patch('app.user_context_analyzer.get_user_history') as mock_get_history:
            
            # Mock user profile
            mock_user = Mock()
            mock_user.recommendation_preferences = {
                "budget": {"min": 2000, "max": 3000},
                "location": ["downtown", "midtown"]
            }
            mock_get_user.return_value = mock_user
            
            # Mock conversation history
            mock_get_history.return_value = [
                {"user_message": "I like properties with modern kitchens", "assistant_message": "Great!"},
                {"user_message": "Parking is important to me", "assistant_message": "Noted!"}
            ]
            
            # Analyze context
            user_context = await context_analyzer.analyze_user_context(user_id)
            
            assert user_context.user_id == user_id
            assert user_context.budget_range == (2000, 3000)
            assert "downtown" in user_context.preferred_locations
            assert "midtown" in user_context.preferred_locations
    
    @pytest.mark.asyncio
    async def test_missing_preferences_identification(self, context_analyzer):
        """Test identification of missing preferences."""
        # Create a user context with some missing preferences
        user_context = UserContext(
            user_id="test@example.com",
            historical_preferences={"budget": {"min": 2000, "max": 3000}},
            budget_range=(2000, 3000),
            preferred_locations=[],  # Missing
            required_features=[],   # Missing
        )
        
        missing = await context_analyzer.identify_missing_preferences(user_context)
        
        assert "location" in missing
        assert "features" in missing or "property_type" in missing
    
    @pytest.mark.asyncio
    async def test_preference_merging(self, context_analyzer):
        """Test merging of new preferences with existing ones."""
        user_id = "test@example.com"
        
        with patch('app.user_context_analyzer.get_user_by_email') as mock_get_user, \
             patch('app.user_context_analyzer.get_user_history') as mock_get_history, \
             patch('app.user_context_analyzer.create_or_update_user') as mock_update_user:
            
            # Mock existing user
            mock_user = Mock()
            mock_user.recommendation_preferences = {"budget": {"min": 2000, "max": 3000}}
            mock_get_user.return_value = mock_user
            mock_get_history.return_value = []
            
            # Mock update function
            mock_update_user.return_value = AsyncMock()
            
            # Merge new preferences
            new_prefs = {"location": ["downtown"], "features": ["parking"]}
            updated_context = await context_analyzer.merge_new_preferences(user_id, new_prefs)
            
            # Check that preferences were merged
            assert "budget" in updated_context.historical_preferences
            assert "location" in updated_context.historical_preferences
            assert "features" in updated_context.historical_preferences


class TestConversationStateManager:
    """Test the Conversation State Manager."""
    
    @pytest.fixture
    def state_manager(self):
        return ConversationStateManager()
    
    @pytest.mark.asyncio
    async def test_session_creation(self, state_manager):
        """Test creation of conversation sessions."""
        user_id = "test@example.com"
        user_context = UserContext(user_id=user_id)
        
        with patch.object(state_manager, '_store_session') as mock_store:
            session = await state_manager.create_session(user_id, user_context)
            
            assert session.user_id == user_id
            assert session.state == ConversationState.INITIATED
            assert session.session_id is not None
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_question_generation(self, state_manager):
        """Test generation of clarifying questions."""
        session = ConversationSession(
            session_id="test-session",
            user_id="test@example.com",
            state=ConversationState.GATHERING_PREFERENCES,
            collected_preferences={},
            questions_asked=[],
            responses_received=[]
        )
        
        with patch.object(state_manager, '_store_session'), \
             patch('app.conversation_state_manager.user_context_analyzer') as mock_analyzer:
            
            # Mock user context with missing preferences
            mock_context = UserContext(
                user_id="test@example.com",
                historical_preferences={},
                budget_range=None,
                preferred_locations=[],
                required_features=[]
            )
            mock_analyzer.analyze_user_context.return_value = mock_context
            
            question = await state_manager.get_next_question(session)
            
            assert question is not None
            assert len(question) > 0
            assert session.session_id in [q for q in session.questions_asked] or len(session.questions_asked) > 0


class TestPropertyRecommendationEngine:
    """Test the Property Recommendation Engine."""
    
    @pytest.fixture
    def recommendation_engine(self):
        return PropertyRecommendationEngine()
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, recommendation_engine):
        """Test property recommendation generation."""
        user_context = UserContext(
            user_id="test@example.com",
            historical_preferences={"property_type": "apartment"},
            budget_range=(2000, 3000),
            preferred_locations=["downtown"],
            required_features=["parking", "gym"]
        )
        
        # Mock the RAG retrieval
        mock_nodes = [Mock() for _ in range(5)]
        for i, node in enumerate(mock_nodes):
            node.get_content.return_value = f"property address: {i} Test St, monthly rent: {2000 + i*200}, size (sf): {800 + i*100}"
            node.metadata = {
                "property_address": f"{i} Test St",
                "monthly_rent": 2000 + i*200,
                "size_sf": 800 + i*100
            }
        
        with patch('app.property_recommendation_engine.rag_module') as mock_rag:
            mock_search_result = Mock()
            mock_search_result.nodes_found = mock_nodes
            mock_rag.retrieve_context_optimized.return_value = mock_search_result
            
            recommendations = await recommendation_engine.generate_recommendations(user_context, max_results=3)
            
            assert len(recommendations) <= 3
            for rec in recommendations:
                assert isinstance(rec, PropertyRecommendation)
                assert rec.property_id is not None
                assert rec.match_score >= 0.0
                assert len(rec.explanation) > 0
    
    def test_budget_matching(self, recommendation_engine):
        """Test budget matching logic."""
        user_context = UserContext(
            user_id="test@example.com",
            budget_range=(2000, 3000)
        )
        
        # Test perfect match
        property_data = {"monthly_rent": 2500}
        score = recommendation_engine._calculate_budget_match(property_data, user_context)
        assert score == 1.0
        
        # Test over budget
        property_data = {"monthly_rent": 3500}
        score = recommendation_engine._calculate_budget_match(property_data, user_context)
        assert score < 1.0
        
        # Test under budget
        property_data = {"monthly_rent": 1800}
        score = recommendation_engine._calculate_budget_match(property_data, user_context)
        assert score >= 0.7


class TestRecommendationWorkflowManager:
    """Test the Recommendation Workflow Manager."""
    
    @pytest.fixture
    def workflow_manager(self):
        return RecommendationWorkflowManager()
    
    @pytest.mark.asyncio
    async def test_workflow_initiation(self, workflow_manager):
        """Test initiation of recommendation workflow."""
        user_id = "test@example.com"
        initial_message = "Suggest me a property"
        
        with patch('app.recommendation_workflow_manager.intent_detection_service') as mock_intent, \
             patch('app.recommendation_workflow_manager.user_context_analyzer') as mock_context, \
             patch('app.recommendation_workflow_manager.conversation_state_manager') as mock_conv, \
             patch.object(workflow_manager, '_store_workflow_session') as mock_store:
            
            # Mock intent detection
            mock_intent.detect_recommendation_intent.return_value = RecommendationIntent(
                is_recommendation_request=True,
                confidence=0.9,
                initial_preferences={"budget": {"max": 3000}}
            )
            
            # Mock user context
            mock_context.analyze_user_context.return_value = UserContext(user_id=user_id)
            mock_context.merge_new_preferences.return_value = UserContext(user_id=user_id)
            
            # Mock conversation session
            mock_session = ConversationSession(
                session_id="conv-session",
                user_id=user_id,
                state=ConversationState.INITIATED
            )
            mock_conv.create_session.return_value = mock_session
            
            # Start workflow
            workflow_session = await workflow_manager.start_recommendation_workflow(user_id, initial_message)
            
            assert workflow_session.user_id == user_id
            assert workflow_session.session_id is not None
            assert "initial_message" in workflow_session.data
            mock_store.assert_called_once()


class TestRecommendationErrorHandler:
    """Test the Recommendation Error Handler."""
    
    @pytest.fixture
    def error_handler(self):
        return RecommendationErrorHandler()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, error_handler):
        """Test error handling with fallback strategies."""
        error_context = RecommendationErrorContext(
            error_type=RecommendationErrorType.INTENT_DETECTION_FAILED,
            user_id="test@example.com",
            error_message="Intent detection failed"
        )
        
        response = await error_handler.handle_recommendation_error(error_context)
        
        assert "message" in response
        assert "strategy" in response
        assert len(response["message"]) > 0
    
    def test_error_tracking(self, error_handler):
        """Test error tracking functionality."""
        error_context = RecommendationErrorContext(
            error_type=RecommendationErrorType.USER_CONTEXT_UNAVAILABLE,
            user_id="test@example.com",
            error_message="Context unavailable"
        )
        
        error_handler._track_error(error_context)
        
        stats = error_handler.get_error_statistics()
        assert "total_errors_by_type" in stats
        assert stats["users_with_errors"] == 1


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def test_client(self):
        return TestClient(app)
    
    def test_recommendation_intent_api(self, test_client):
        """Test the recommendation intent detection API endpoint."""
        response = test_client.post(
            "/api/debug/test-recommendation-intent",
            json="Suggest me a property"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_recommendation" in data
        assert "confidence" in data
    
    @pytest.mark.asyncio
    async def test_full_recommendation_workflow(self):
        """Test the complete recommendation workflow with mock data."""
        # This test would require more complex setup with temporary databases
        # and mock property data, but demonstrates the full workflow structure
        
        user_id = "test@example.com"
        message = "Find me an apartment under $3000"
        
        # Step 1: Intent detection
        intent_service = IntentDetectionService()
        intent = await intent_service.detect_recommendation_intent(message)
        assert intent.is_recommendation_request
        
        # Step 2: User context analysis (mocked)
        user_context = UserContext(
            user_id=user_id,
            budget_range=(0, 3000),
            preferred_locations=[],
            required_features=[]
        )
        
        # Step 3: Property recommendation (mocked)
        recommendation_engine = PropertyRecommendationEngine()
        
        # Mock the property retrieval
        with patch('app.property_recommendation_engine.rag_module') as mock_rag:
            mock_search_result = Mock()
            mock_search_result.nodes_found = []
            mock_rag.retrieve_context_optimized.return_value = mock_search_result
            
            recommendations = await recommendation_engine.generate_recommendations(user_context, max_results=3)
            
            # Even with no properties, the engine should handle gracefully
            assert isinstance(recommendations, list)


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 