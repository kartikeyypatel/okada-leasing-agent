#!/usr/bin/env python3
"""
Test suite for strict response generation functionality.

This test suite verifies that the strict response generator:
1. Only uses retrieved context for responses
2. Validates context before response generation  
3. Implements clear "not found" responses when no relevant documents exist
4. Adds response quality validation to prevent hallucination

Requirements tested: 4.1, 4.2, 4.3, 4.4
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import List

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.strict_response_generator import (
    StrictResponseGenerator,
    ContextValidationResult,
    ResponseQualityResult,
    StrictResponseResult
)
from llama_index.core.schema import NodeWithScore, TextNode


class TestStrictResponseGenerator:
    """Test cases for the StrictResponseGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = StrictResponseGenerator()
        
        # Create mock nodes with property data
        self.mock_property_nodes = [
            self._create_mock_node(
                "address: 84 Mulberry St, monthly rent: $80,522, size: 9,567 SF, type: Commercial",
                score=0.9
            ),
            self._create_mock_node(
                "address: 123 Main St, monthly rent: $45,000, size: 5,000 SF, type: Office",
                score=0.8
            )
        ]
        
        self.empty_nodes = []
        
    def _create_mock_node(self, content: str, score: float = 0.5) -> NodeWithScore:
        """Create a mock NodeWithScore for testing."""
        text_node = TextNode(text=content)
        node_with_score = NodeWithScore(node=text_node, score=score)
        return node_with_score

    @pytest.mark.asyncio
    async def test_context_validation_with_valid_nodes(self):
        """Test that context validation works correctly with valid property nodes."""
        result = await self.generator._validate_context(
            "tell me about 84 Mulberry St",
            self.mock_property_nodes
        )
        
        assert result.is_valid
        assert result.property_count == 2
        assert result.has_specific_property
        assert "84 Mulberry St" in result.specific_property_address
        assert result.confidence_score > 0.5
        
    @pytest.mark.asyncio
    async def test_context_validation_with_empty_nodes(self):
        """Test that context validation fails correctly with no nodes."""
        result = await self.generator._validate_context(
            "tell me about 84 Mulberry St",
            self.empty_nodes
        )
        
        assert not result.is_valid
        assert result.property_count == 0
        assert not result.has_specific_property
        assert result.confidence_score == 0.0
        assert "No retrieved nodes available" in result.validation_issues

    @pytest.mark.asyncio
    async def test_context_validation_property_not_found(self):
        """Test context validation when specific property is not in the data."""
        result = await self.generator._validate_context(
            "tell me about 999 Nonexistent St",
            self.mock_property_nodes
        )
        
        # Should be invalid because the specific property isn't found
        assert not result.is_valid
        assert result.property_count == 2  # We have properties, just not the right one
        assert not result.has_specific_property
        assert result.specific_property_address == "999 Nonexistent St"

    @pytest.mark.asyncio
    async def test_address_extraction_from_query(self):
        """Test that addresses are correctly extracted from user queries."""
        test_cases = [
            ("tell me about 84 Mulberry St", "84 Mulberry St"),
            ("information about 123 Main Street", "123 Main Street"),
            ("show me 456 Oak Ave", "456 Oak Ave"),
            ("what is the rent for 789 Pine Rd?", "789 Pine Rd"),
            ("general question about properties", None)
        ]
        
        for query, expected_address in test_cases:
            result = self.generator._extract_address_from_query(query)
            assert result == expected_address, f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_addresses_match_function(self):
        """Test that address matching works correctly with various formats."""
        test_cases = [
            ("84 Mulberry St", "84 Mulberry Street", True),
            ("123 Main Ave", "123 Main Avenue", True),
            ("456 Oak Rd", "456 Oak Road", True),
            ("84 Mulberry St", "85 Mulberry St", False),
            ("123 Main St", "123 Oak St", False),
            ("", "123 Main St", False),
            ("123 Main St", "", False)
        ]
        
        for addr1, addr2, should_match in test_cases:
            result = self.generator._addresses_match(addr1, addr2)
            assert result == should_match, f"Failed for {addr1} vs {addr2}"

    @pytest.mark.asyncio
    async def test_format_context_for_prompt(self):
        """Test that context is properly formatted for LLM prompts."""
        formatted = self.generator._format_context_for_prompt(self.mock_property_nodes)
        
        assert "AVAILABLE PROPERTIES:" in formatted
        assert "Property 1" in formatted
        assert "Property 2" in formatted
        assert "84 Mulberry St" in formatted
        assert "123 Main St" in formatted
        assert "Relevance:" in formatted

    @pytest.mark.asyncio
    async def test_extract_search_target(self):
        """Test extraction of search targets for 'not found' messages."""
        test_cases = [
            ("tell me about 84 Mulberry St", '"84 Mulberry St"'),
            ("show me that property", "that property"),
            ("information about the apartment", "that apartment"),
            ("details on the building", "that building"),
            ("random question", "what you're looking for")
        ]
        
        for query, expected_target in test_cases:
            result = self.generator._extract_search_target(query)
            assert result == expected_target, f"Failed for query: {query}"

    @pytest.mark.asyncio
    @patch('app.strict_response_generator.Settings')
    async def test_generate_strict_response_with_valid_context(self, mock_settings):
        """Test complete strict response generation with valid context."""
        # Mock the LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.message.content = "Based on the property data, 84 Mulberry St is a commercial property with monthly rent of $80,522 and size of 9,567 SF."
        mock_llm.achat.return_value = mock_response
        mock_settings.llm = mock_llm
        
        # Mock the quality validation to return valid
        with patch.object(self.generator, '_validate_response_quality') as mock_quality:
            mock_quality.return_value = ResponseQualityResult(
                is_valid=True,
                uses_only_context=True,
                contains_hallucination=False,
                quality_issues=[],
                confidence_score=0.9
            )
            
            result = await self.generator.generate_strict_response(
                user_query="tell me about 84 Mulberry St",
                retrieved_nodes=self.mock_property_nodes,
                user_id="test@example.com"
            )
        
        assert result.generation_successful
        assert not result.fallback_used
        assert result.context_validation.is_valid
        assert result.quality_validation.is_valid
        assert "84 Mulberry St" in result.response_text

    @pytest.mark.asyncio
    async def test_generate_not_found_response(self):
        """Test generation of 'not found' responses."""
        context_validation = ContextValidationResult(
            is_valid=False,
            context_summary="No relevant context",
            property_count=0,
            has_specific_property=False,
            specific_property_address="999 Nonexistent St",
            validation_issues=["Property not found"],
            confidence_score=0.0
        )
        
        result = await self.generator._generate_not_found_response(
            "tell me about 999 Nonexistent St",
            context_validation,
            "test@example.com"
        )
        
        assert result.generation_successful
        assert result.fallback_used
        assert "don't have information" in result.response_text.lower()
        assert "999 Nonexistent St" in result.response_text
        assert result.quality_validation.is_valid  # Not found responses are always valid

    @pytest.mark.asyncio
    async def test_generate_error_response(self):
        """Test generation of error responses when something goes wrong."""
        context_validation = ContextValidationResult(
            is_valid=True,
            context_summary="Valid context",
            property_count=1,
            has_specific_property=True,
            specific_property_address="84 Mulberry St",
            validation_issues=[],
            confidence_score=0.8
        )
        
        result = await self.generator._generate_error_response(
            "tell me about 84 Mulberry St",
            context_validation,
            "Test error message",
            "test@example.com"
        )
        
        assert not result.generation_successful
        assert result.fallback_used
        assert "trouble processing" in result.response_text.lower()
        assert not result.quality_validation.is_valid

    @pytest.mark.asyncio
    @patch('app.strict_response_generator.Settings')
    async def test_response_quality_validation(self, mock_settings):
        """Test that response quality validation detects hallucination."""
        # Mock LLM response for quality validation
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.message.content = '{"uses_only_context": false, "contains_hallucination": true, "specific_issues": ["Response contains invented information"], "confidence_score": 0.2}'
        mock_llm.achat.return_value = mock_response
        mock_settings.llm = mock_llm
        
        result = await self.generator._validate_response_quality(
            "tell me about 84 Mulberry St",
            self.mock_property_nodes,
            "84 Mulberry St is a luxury building with a swimming pool and gym."  # Hallucinated info
        )
        
        assert not result.is_valid
        assert not result.uses_only_context
        assert result.contains_hallucination
        assert "invented information" in str(result.quality_issues)
        assert result.confidence_score < 0.5

    @pytest.mark.asyncio
    @patch('app.strict_response_generator.Settings')
    async def test_regenerate_with_stricter_prompt(self, mock_settings):
        """Test that responses can be regenerated with stricter prompts."""
        # Mock LLM response for regeneration
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.message.content = "I don't have information about amenities for 84 Mulberry St. The available data shows: address: 84 Mulberry St, monthly rent: $80,522, size: 9,567 SF, type: Commercial."
        mock_llm.achat.return_value = mock_response
        mock_settings.llm = mock_llm
        
        context_validation = ContextValidationResult(
            is_valid=True,
            context_summary="Valid context",
            property_count=1,
            has_specific_property=True,
            specific_property_address="84 Mulberry St",
            validation_issues=[],
            confidence_score=0.8
        )
        
        quality_validation = ResponseQualityResult(
            is_valid=False,
            uses_only_context=False,
            contains_hallucination=True,
            quality_issues=["Response contains invented amenity information"],
            confidence_score=0.3
        )
        
        result = await self.generator._regenerate_with_stricter_prompt(
            "tell me about amenities at 84 Mulberry St",
            self.mock_property_nodes,
            context_validation,
            quality_validation
        )
        
        assert "don't have information about amenities" in result
        assert "84 Mulberry St" in result
        assert "$80,522" in result

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self):
        """Test that confidence scores are calculated correctly."""
        # Test high confidence case
        high_confidence = self.generator._calculate_context_confidence(
            "tell me about 84 Mulberry St",
            self.mock_property_nodes,
            ["84 Mulberry St", "123 Main St"],
            has_specific_property=True
        )
        assert high_confidence > 0.7
        
        # Test low confidence case
        low_confidence = self.generator._calculate_context_confidence(
            "tell me about 999 Nonexistent St",
            [self._create_mock_node("unrelated content", 0.1)],
            [],
            has_specific_property=False
        )
        assert low_confidence < 0.3

    @pytest.mark.asyncio
    async def test_create_strict_prompt(self):
        """Test that strict prompts are created correctly."""
        formatted_context = "Property 1: address: 84 Mulberry St, rent: $80,522"
        context_validation = ContextValidationResult(
            is_valid=True,
            context_summary="Valid context",
            property_count=1,
            has_specific_property=True,
            specific_property_address="84 Mulberry St",
            validation_issues=[],
            confidence_score=0.8
        )
        
        prompt = self.generator._create_strict_prompt(
            "tell me about 84 Mulberry St",
            formatted_context,
            context_validation
        )
        
        assert "ONLY the property information provided" in prompt
        assert "Do NOT make up" in prompt
        assert "84 Mulberry St" in prompt
        assert formatted_context in prompt
        assert "CRITICAL INSTRUCTIONS" in prompt


class TestStrictResponseIntegration:
    """Integration tests for strict response generation with the main application."""
    
    @pytest.mark.asyncio
    async def test_integration_with_empty_nodes(self):
        """Test integration when no nodes are retrieved."""
        generator = StrictResponseGenerator()
        
        result = await generator.generate_strict_response(
            user_query="tell me about 999 Nonexistent St",
            retrieved_nodes=[],
            user_id="test@example.com"
        )
        
        assert result.generation_successful
        assert result.fallback_used
        assert "don't have information" in result.response_text.lower()
        assert not result.context_validation.is_valid
        assert result.quality_validation.is_valid  # Not found responses are valid

    @pytest.mark.asyncio
    @patch('app.strict_response_generator.Settings')
    async def test_integration_with_llm_error(self, mock_settings):
        """Test integration when LLM calls fail."""
        # Mock LLM to raise an exception
        mock_llm = AsyncMock()
        mock_llm.achat.side_effect = Exception("LLM service unavailable")
        mock_settings.llm = mock_llm
        
        generator = StrictResponseGenerator()
        
        # Create some mock nodes
        nodes = [
            generator._create_mock_node("address: 84 Mulberry St, rent: $80,522", 0.9)
        ]
        
        result = await generator.generate_strict_response(
            user_query="tell me about 84 Mulberry St",
            retrieved_nodes=nodes,
            user_id="test@example.com"
        )
        
        assert not result.generation_successful
        assert result.fallback_used
        assert "trouble processing" in result.response_text.lower()


def run_tests():
    """Run all tests."""
    print("🧪 Running strict response generation tests...")
    
    # Run the tests
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ All strict response generation tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)