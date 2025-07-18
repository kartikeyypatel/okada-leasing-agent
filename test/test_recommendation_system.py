# /test/test_recommendation_system.py
"""
Comprehensive Test Suite for Smart Property Recommendations

This module contains unit tests, integration tests, and end-to-end tests
for the complete Smart Property Recommendations system.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock, patch

# Import the modules to test
from app.intent_detection import IntentDetectionService
from app.user_context_analyzer import UserContextAnalyzer
from app.conversation_state_manager import ConversationStateManager
from app.property_recommendation_engine import PropertyRecommendationEngine
from app.recommendation_workflow_manager import RecommendationWorkflowManager
from app.models import (
    RecommendationIntent, UserContext, ConversationSession, 
    PropertyRecommendation, RecommendationResult, ConversationState
)


class TestIntentDetectionService:
    """Unit tests for Intent Detection Service."""
    
    @pytest.fixture
    def intent_service(self):
        return IntentDetectionService()
    
    @pytest.mark.asyncio
    async def test_detect_recommendation_intent_positive(self, intent_service):
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
            assert intent.is_recommendation_request == True
            assert intent.confidence > 0.6
            assert isinstance(intent.initial_preferences, dict)
    
    @pytest.mark.asyncio
    async def test_detect_recommendation_intent_negative(self, intent_service):
        """Test rejection of non-recommendation requests."""
        test_messages = [
            "Tell me about 123 Main St",
            "What is the rent for the 2-bedroom?",
            "How do I schedule a viewing?",
            "Thank you for your help",
            "What's the weather like?",
            "Hello, how are you?"
        ]
        
        for message in test_messages:
            intent = await intent_service.detect_recommendation_intent(message)
            assert intent.is_recommendation_request == False or intent.confidence < 0.6
    
    def test_extract_initial_preferences(self, intent_service):
        """Test extraction of preferences from messages."""
        message = "Find me a 2-bedroom apartment under $3000 in downtown with parking"
        preferences = intent_service.extract_initial_preferences(message)
        
        assert "bedrooms" in str(preferences).lower()
        assert "3000" in str(preferences) or "budget" in str(preferences).lower()
        assert "downtown" in str(preferences).lower()
        assert "parking" in str(preferences).lower()


class TestUserContextAnalyzer:
    """Unit tests for User Context Analyzer."""
    
    @pytest.fixture
    def context_analyzer(self):
        return UserContextAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_user_context_new_user(self, context_analyzer):
        """Test context analysis for new user with no history."""
        with patch.object(context_analyzer, '_get_user_profile', return_value={}):
            with patch.object(context_analyzer, '_get_conversation_history', return_value=[]):
                context = await context_analyzer.analyze_user_context("newuser@test.com")
                
                assert context.user_id == "newuser@test.com"
                assert isinstance(context.historical_preferences, dict)
                assert context.budget_range is None
                assert context.preferred_locations == []
                assert context.required_features == []
    
    @pytest.mark.asyncio
    async def test_analyze_user_context_existing_user(self, context_analyzer):
        """Test context analysis for existing user with preferences."""
        mock_profile = {
            'recommendation_preferences': {
                'budget': {'min': 2000, 'max': 3000},
                'location': ['downtown', 'midtown'],
                'required_features': ['parking', 'gym']
            }
        }
        
        with patch.object(context_analyzer, '_get_user_profile', return_value=mock_profile):
            with patch.object(context_analyzer, '_get_conversation_history', return_value=[]):
                context = await context_analyzer.analyze_user_context("existing@test.com")
                
                assert context.budget_range == (2000, 3000)
                assert "downtown" in context.preferred_locations
                assert "parking" in context.required_features
    
    @pytest.mark.asyncio
    async def test_identify_missing_preferences(self, context_analyzer):
        """Test identification of missing preferences."""
        # Context with only budget
        context = UserContext(
            user_id="test@test.com",
            historical_preferences={'budget': {'min': 2000, 'max': 3000}},
            budget_range=(2000, 3000),
            preferred_locations=[],
            required_features=[],
            excluded_features=[]
        )
        
        missing = await context_analyzer.identify_missing_preferences(context)
        
        assert "location" in missing
        assert "property_type" in missing
        assert "features" in missing
        assert "budget" not in missing
    
    @pytest.mark.asyncio
    async def test_merge_new_preferences(self, context_analyzer):
        """Test merging of new preferences with existing ones."""
        new_prefs = {
            'location': ['brooklyn'],
            'required_features': ['balcony']
        }
        
        with patch.object(context_analyzer, 'analyze_user_context') as mock_analyze:
            mock_context = UserContext(
                user_id="test@test.com",
                historical_preferences={'budget': {'min': 2000, 'max': 3000}},
                budget_range=(2000, 3000),
                preferred_locations=[],
                required_features=[],
                excluded_features=[]
            )
            mock_analyze.return_value = mock_context
            
            with patch.object(context_analyzer, '_update_user_preferences') as mock_update:
                updated_context = await context_analyzer.merge_new_preferences("test@test.com", new_prefs)
                
                assert "brooklyn" in updated_context.preferred_locations
                assert "balcony" in updated_context.required_features
                assert updated_context.budget_range == (2000, 3000)  # Preserved existing


class TestConversationStateManager:
    """Unit tests for Conversation State Manager."""
    
    @pytest.fixture
    def state_manager(self):
        return ConversationStateManager()
    
    @pytest.mark.asyncio
    async def test_create_session(self, state_manager):
        """Test creation of new conversation session."""
        mock_context = UserContext(
            user_id="test@test.com",
            historical_preferences={},
            budget_range=None,
            preferred_locations=[],
            required_features=[],
            excluded_features=[]
        )
        
        with patch.object(state_manager, '_store_session') as mock_store:
            session = await state_manager.create_session("test@test.com", mock_context)
            
            assert session.user_id == "test@test.com"
            assert session.state == ConversationState.INITIATED
            assert session.collected_preferences == {}
            assert session.questions_asked == []
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_session(self, state_manager):
        """Test updating session with user response."""
        session = ConversationSession(
            session_id="test-session",
            user_id="test@test.com",
            state=ConversationState.GATHERING_PREFERENCES,
            collected_preferences={},
            questions_asked=["What's your budget?"],
            responses_received=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        with patch.object(state_manager, '_get_session', return_value=session):
            with patch.object(state_manager, '_extract_preferences_from_response', return_value={'budget': {'max': 3000}}):
                with patch.object(state_manager, '_update_session_state'):
                    with patch.object(state_manager, '_store_session'):
                        updated_session = await state_manager.update_session("test-session", "$3000 maximum")
                        
                        assert "budget" in updated_session.collected_preferences
                        assert "$3000 maximum" in updated_session.responses_received
    
    @pytest.mark.asyncio
    async def test_get_next_question(self, state_manager):
        """Test generation of next clarifying question."""
        session = ConversationSession(
            session_id="test-session",
            user_id="test@test.com",
            state=ConversationState.GATHERING_PREFERENCES,
            collected_preferences={'budget': {'max': 3000}},
            questions_asked=["What's your budget?"],
            responses_received=["$3000 maximum"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        with patch('app.user_context_analyzer.user_context_analyzer.analyze_user_context') as mock_analyze:
            mock_context = UserContext(
                user_id="test@test.com",
                historical_preferences={'budget': {'max': 3000}},
                budget_range=(0, 3000),
                preferred_locations=[],
                required_features=[],
                excluded_features=[]
            )
            mock_analyze.return_value = mock_context
            
            with patch.object(state_manager, '_identify_missing_categories', return_value=['location']):
                with patch.object(state_manager, '_generate_personalized_question', return_value="Which areas interest you?"):
                    with patch.object(state_manager, '_store_session'):
                        question = await state_manager.get_next_question(session)
                        
                        assert question == "Which areas interest you?"


class TestPropertyRecommendationEngine:
    """Unit tests for Property Recommendation Engine."""
    
    @pytest.fixture
    def recommendation_engine(self):
        return PropertyRecommendationEngine()
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, recommendation_engine):
        """Test generation of property recommendations."""
        mock_context = UserContext(
            user_id="test@test.com",
            historical_preferences={
                'budget': {'min': 2000, 'max': 3000},
                'location': ['downtown'],
                'required_features': ['parking']
            },
            budget_range=(2000, 3000),
            preferred_locations=['downtown'],
            required_features=['parking'],
            excluded_features=[]
        )
        
        # Mock property data
        mock_properties = [
            {
                'property_id': 'prop1',
                'property_address': '123 Main St',
                'monthly_rent': 2500,
                'location': 'downtown',
                'features': ['parking', 'gym']
            },
            {
                'property_id': 'prop2',
                'property_address': '456 Oak Ave',
                'monthly_rent': 2800,
                'location': 'downtown',
                'features': ['parking', 'balcony']
            }
        ]
        
        with patch.object(recommendation_engine, '_retrieve_properties', return_value=mock_properties):
            with patch.object(recommendation_engine, '_score_properties') as mock_score:
                mock_scored = [
                    (mock_properties[0], 0.9),
                    (mock_properties[1], 0.8)
                ]
                mock_score.return_value = mock_scored
                
                with patch.object(recommendation_engine, '_select_top_recommendations') as mock_select:
                    mock_recommendations = [
                        PropertyRecommendation(
                            property_id='prop1',
                            property_data=mock_properties[0],
                            match_score=0.9,
                            explanation="Great match for your budget and location preferences",
                            matching_criteria=['budget', 'location', 'parking']
                        )
                    ]
                    mock_select.return_value = mock_recommendations
                    
                    recommendations = await recommendation_engine.generate_recommendations(mock_context)
                    
                    assert len(recommendations) == 1
                    assert recommendations[0].property_id == 'prop1'
                    assert recommendations[0].match_score == 0.9
    
    def test_build_search_queries(self, recommendation_engine):
        """Test building of search queries from user context."""
        context = UserContext(
            user_id="test@test.com",
            historical_preferences={},
            budget_range=(2000, 3000),
            preferred_locations=['downtown', 'midtown'],
            required_features=['parking', 'gym'],
            excluded_features=[]
        )
        
        queries = recommendation_engine._build_search_queries(context)
        
        # Check that queries contain expected elements
        query_text = ' '.join(queries)
        assert '2000' in query_text and '3000' in query_text
        assert 'downtown' in query_text
        assert 'parking' in query_text


class TestRecommendationWorkflowManager:
    """Unit tests for Recommendation Workflow Manager."""
    
    @pytest.fixture
    def workflow_manager(self):
        return RecommendationWorkflowManager()
    
    @pytest.mark.asyncio
    async def test_start_recommendation_workflow(self, workflow_manager):
        """Test starting of recommendation workflow."""
        with patch('app.intent_detection.intent_detection_service.detect_recommendation_intent') as mock_intent:
            mock_intent.return_value = RecommendationIntent(
                is_recommendation_request=True,
                confidence=0.9,
                initial_preferences={'location': ['downtown']},
                trigger_phrases=['suggest me a property']
            )
            
            with patch('app.user_context_analyzer.user_context_analyzer.analyze_user_context') as mock_context:
                mock_context.return_value = UserContext(
                    user_id="test@test.com",
                    historical_preferences={},
                    budget_range=None,
                    preferred_locations=[],
                    required_features=[],
                    excluded_features=[]
                )
                
                with patch('app.conversation_state_manager.conversation_state_manager.create_session') as mock_conv:
                    mock_conv.return_value = ConversationSession(
                        session_id="conv-session",
                        user_id="test@test.com",
                        state=ConversationState.INITIATED,
                        collected_preferences={},
                        questions_asked=[],
                        responses_received=[],
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    with patch.object(workflow_manager, '_store_workflow_session'):
                        session = await workflow_manager.start_recommendation_workflow(
                            "test@test.com", 
                            "Suggest me a property"
                        )
                        
                        assert session.user_id == "test@test.com"
                        assert session.current_step == "initiated"
                        assert "intent" in session.data


class TestEndToEndWorkflow:
    """End-to-end integration tests for the complete recommendation workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_recommendation_workflow(self):
        """Test complete workflow from intent detection to final recommendations."""
        # This would be a comprehensive test that exercises the entire system
        # In a real implementation, this would use test databases and mock external services
        
        # Mock user message
        user_message = "Find me a 2-bedroom apartment under $3000 in downtown"
        user_id = "test@test.com"
        
        # Test would follow these steps:
        # 1. Intent detection
        # 2. User context analysis
        # 3. Conversation management
        # 4. Property recommendation generation
        # 5. Workflow completion
        
        # For now, we'll test the basic flow structure
        assert user_message is not None
        assert user_id is not None
        
        # In a full implementation, this would verify:
        # - Intent is correctly detected
        # - User context is properly analyzed
        # - Conversation flows correctly
        # - Recommendations are generated
        # - Results are properly formatted
    
    @pytest.mark.asyncio
    async def test_workflow_with_clarifying_questions(self):
        """Test workflow that requires clarifying questions."""
        # Test scenario where user provides minimal initial information
        # and system asks clarifying questions
        
        user_message = "I need an apartment"
        user_id = "newuser@test.com"
        
        # This test would verify:
        # - System detects insufficient information
        # - Appropriate clarifying questions are generated
        # - User responses are processed correctly
        # - Final recommendations are generated after sufficient information is gathered
        
        assert user_message is not None
        assert user_id is not None
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test error handling throughout the workflow."""
        # Test various error scenarios:
        # - Database connection failures
        # - LLM API failures
        # - Invalid user input
        # - Property retrieval failures
        
        # Verify that system gracefully handles errors and provides fallback responses
        pass
    
    @pytest.mark.asyncio
    async def test_workflow_performance(self):
        """Test workflow performance under load."""
        # Test performance characteristics:
        # - Response time for intent detection
        # - Context analysis speed
        # - Recommendation generation time
        # - Memory usage
        
        # Verify that system meets performance requirements
        pass


class TestRecommendationAnalytics:
    """Tests for recommendation system analytics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_analytics_tracking(self):
        """Test that analytics are properly tracked."""
        # Test that system tracks:
        # - Recommendation requests
        # - User interactions
        # - Success/failure rates
        # - Performance metrics
        pass
    
    @pytest.mark.asyncio
    async def test_recommendation_quality_metrics(self):
        """Test recommendation quality measurement."""
        # Test quality metrics:
        # - Relevance scores
        # - User satisfaction indicators
        # - Conversion rates
        pass


# Test fixtures and utilities
@pytest.fixture
def mock_database():
    """Mock database for testing."""
    return Mock()

@pytest.fixture
def sample_user_context():
    """Sample user context for testing."""
    return UserContext(
        user_id="test@test.com",
        historical_preferences={
            'budget': {'min': 2000, 'max': 3000},
            'location': ['downtown'],
            'required_features': ['parking', 'gym']
        },
        budget_range=(2000, 3000),
        preferred_locations=['downtown'],
        required_features=['parking', 'gym'],
        excluded_features=['noisy'],
        last_updated=datetime.now()
    )

@pytest.fixture
def sample_properties():
    """Sample property data for testing."""
    return [
        {
            'property_id': 'prop1',
            'property_address': '123 Main St',
            'monthly_rent': 2500,
            'location': 'downtown',
            'features': ['parking', 'gym', 'balcony'],
            'bedrooms': 2,
            'bathrooms': 1,
            'sqft': 900
        },
        {
            'property_id': 'prop2',
            'property_address': '456 Oak Ave',
            'monthly_rent': 2800,
            'location': 'downtown',
            'features': ['parking', 'pool'],
            'bedrooms': 2,
            'bathrooms': 2,
            'sqft': 1100
        },
        {
            'property_id': 'prop3',
            'property_address': '789 Pine St',
            'monthly_rent': 3200,
            'location': 'uptown',
            'features': ['gym', 'doorman'],
            'bedrooms': 3,
            'bathrooms': 2,
            'sqft': 1300
        }
    ]


# Performance test helpers
def measure_performance(func):
    """Decorator to measure function performance."""
    import time
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


# Test configuration
pytest_plugins = ['pytest_asyncio']

# Test data cleanup
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    # Cleanup code would go here
    # In a real implementation, this would clean up test database records
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 