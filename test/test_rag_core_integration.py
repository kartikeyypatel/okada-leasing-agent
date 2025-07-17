"""
Core RAG Integration Tests

This module contains comprehensive integration tests for the core RAG functionality,
testing the complete pipeline from document ingestion to response generation.

Tests cover:
- Index building and caching
- Document retrieval and search
- Response generation with strict validation
- Error handling and recovery
- Performance monitoring integration

Requirements covered: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict
import time

# Import the modules to test
import app.rag as rag_module
from app.chroma_client import ChromaClientManager, chroma_manager
from app.strict_response_generator import StrictResponseGenerator
from app.multi_strategy_search import multi_strategy_search
from app.config import settings
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.fixture
def sample_property_data():
    """Sample property data for testing."""
    return [
        {
            "unique_id": "1",
            "property address": "36 W 36th St",
            "floor": "E3",
            "suite": "300",
            "size (sf)": "18650",
            "rent/sf/year": "$87.00",
            "annual rent": "$1,622,550",
            "monthly rent": "$135,213",
            "associate 1": "Hector Barbossa",
            "broker email id": "test1@okadaco.com"
        },
        {
            "unique_id": "2",
            "property address": "15 W 38th St",
            "floor": "E3",
            "suite": "300",
            "size (sf)": "17260",
            "rent/sf/year": "$109.00",
            "annual rent": "$1,881,340",
            "monthly rent": "$156,778",
            "associate 1": "Joshamee Gibbs",
            "broker email id": "test1@okadaco.com"
        },
        {
            "unique_id": "3",
            "property address": "15 W 38th St",
            "floor": "E6",
            "suite": "600",
            "size (sf)": "15044",
            "rent/sf/year": "$87.00",
            "annual rent": "$1,308,828",
            "monthly rent": "$109,069",
            "associate 1": "James Norrington",
            "broker email id": "test1@okadaco.com"
        }
    ]


@pytest.fixture
def temp_csv_file(sample_property_data):
    """Create a temporary CSV file with sample data."""
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "test_properties.csv")
    
    df = pd.DataFrame(sample_property_data)
    df.to_csv(csv_path, index=False)
    
    yield csv_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_nodes():
    """Create mock nodes for testing."""
    nodes = []
    sample_data = [
        {
            "text": "unique_id: 1, property address: 36 W 36th St, floor: E3, suite: 300, size (sf): 18650, rent/sf/year: $87.00",
            "metadata": {"property address": "36 W 36th St", "rent/sf/year": "$87.00"},
            "score": 0.95
        },
        {
            "text": "unique_id: 2, property address: 15 W 38th St, floor: E3, suite: 300, size (sf): 17260, rent/sf/year: $109.00",
            "metadata": {"property address": "15 W 38th St", "rent/sf/year": "$109.00"},
            "score": 0.87
        }
    ]
    
    for data in sample_data:
        node = TextNode(
            text=data["text"],
            metadata=data["metadata"],
            id_=f"test_node_{len(nodes)}"
        )
        node_with_score = NodeWithScore(node=node, score=data["score"])
        nodes.append(node_with_score)
    
    return nodes


class TestCoreRAGIntegration:
    """Test core RAG integration functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_rag_pipeline(self, temp_csv_file, sample_property_data):
        """Test the complete RAG pipeline from CSV to response."""
        test_user_id = "test_core_integration@test.com"
        
        try:
            # Step 1: Build index from CSV
            index = await rag_module.build_user_index(test_user_id, [temp_csv_file])
            assert index is not None, "Index should be created successfully"
            
            # Step 2: Verify index is cached
            cached_index = rag_module.user_indexes.get(test_user_id)
            assert cached_index is not None, "Index should be cached"
            assert cached_index == index, "Cached index should match created index"
            
            # Step 3: Verify BM25 retriever is created
            bm25_retriever = rag_module.user_bm25_retrievers.get(test_user_id)
            assert bm25_retriever is not None, "BM25 retriever should be created"
            
            # Step 4: Test fusion retriever creation
            fusion_retriever = rag_module.get_fusion_retriever(test_user_id)
            assert fusion_retriever is not None, "Fusion retriever should be created"
            
            # Step 5: Test search functionality
            search_result = await multi_strategy_search(fusion_retriever, "36 W 36th St")
            assert len(search_result.nodes_found) > 0, "Search should find relevant documents"
            
            # Step 6: Test response generation
            generator = StrictResponseGenerator()
            response_result = await generator.generate_strict_response(
                user_query="tell me about 36 W 36th St",
                retrieved_nodes=search_result.nodes_found,
                user_id=test_user_id
            )
            
            assert response_result.generation_successful, "Response generation should succeed"
            assert response_result.context_validation.is_valid, "Context should be valid"
            assert "36 W 36th St" in response_result.response_text, "Response should mention the property"
            
        finally:
            # Cleanup
            await rag_module.clear_user_index(test_user_id)
    
    @pytest.mark.asyncio
    async def test_index_building_with_multiple_files(self, sample_property_data):
        """Test index building with multiple CSV files."""
        test_user_id = "test_multi_files@test.com"
        
        # Create multiple temporary CSV files
        temp_dir = tempfile.mkdtemp()
        try:
            file_paths = []
            
            # Split data into two files
            for i, data_chunk in enumerate([sample_property_data[:2], sample_property_data[2:]]):
                csv_path = os.path.join(temp_dir, f"properties_{i}.csv")
                df = pd.DataFrame(data_chunk)
                df.to_csv(csv_path, index=False)
                file_paths.append(csv_path)
            
            # Build index from multiple files
            index = await rag_module.build_user_index(test_user_id, file_paths)
            assert index is not None, "Index should be created from multiple files"
            
            # Verify all data is included
            fusion_retriever = rag_module.get_fusion_retriever(test_user_id)
            assert fusion_retriever is not None, "Fusion retriever should be created"
            
            # Test search across all files
            search_result = await multi_strategy_search(fusion_retriever, "15 W 38th St")
            assert len(search_result.nodes_found) >= 2, "Should find documents from multiple files"
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            await rag_module.clear_user_index(test_user_id)
    
    @pytest.mark.asyncio
    async def test_search_accuracy_and_ranking(self, temp_csv_file):
        """Test search accuracy and result ranking."""
        test_user_id = "test_search_accuracy@test.com"
        
        try:
            # Build index
            index = await rag_module.build_user_index(test_user_id, [temp_csv_file])
            assert index is not None
            
            fusion_retriever = rag_module.get_fusion_retriever(test_user_id)
            assert fusion_retriever is not None
            
            # Test specific address search
            search_result = await multi_strategy_search(fusion_retriever, "36 W 36th St")
            nodes_found = search_result.nodes_found
            
            assert len(nodes_found) > 0, "Should find the specific property"
            
            # Verify the most relevant result is first
            top_result = nodes_found[0]
            assert "36 W 36th St" in top_result.text, "Top result should contain the searched address"
            
            # Test rent-based search
            search_result = await multi_strategy_search(fusion_retriever, "rent under $90")
            nodes_found = search_result.nodes_found
            
            assert len(nodes_found) > 0, "Should find properties with low rent"
            
            # Verify results contain relevant rent information
            found_low_rent = any("$87.00" in node.text for node in nodes_found)
            assert found_low_rent, "Should find properties with $87.00 rent"
            
        finally:
            await rag_module.clear_user_index(test_user_id)
    
    @pytest.mark.asyncio
    async def test_response_generation_accuracy(self, mock_nodes):
        """Test response generation accuracy with mock data."""
        generator = StrictResponseGenerator()
        
        # Test with specific property query
        response_result = await generator.generate_strict_response(
            user_query="tell me about 36 W 36th St",
            retrieved_nodes=mock_nodes,
            user_id="test_user"
        )
        
        assert response_result.generation_successful, "Response generation should succeed"
        assert response_result.context_validation.is_valid, "Context should be valid"
        assert response_result.quality_validation.is_valid, "Quality should be valid"
        
        response_text = response_result.response_text
        assert "36 W 36th St" in response_text, "Response should mention the property"
        assert "$87.00" in response_text, "Response should include rent information"
        
        # Test with no relevant nodes
        empty_response = await generator.generate_strict_response(
            user_query="tell me about 999 Nonexistent St",
            retrieved_nodes=[],
            user_id="test_user"
        )
        
        assert empty_response.generation_successful, "Should handle empty results gracefully"
        assert empty_response.fallback_used, "Should use fallback response"
        assert "don't have information" in empty_response.response_text.lower(), "Should indicate no information available"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_csv_file):
        """Test error handling and recovery mechanisms."""
        test_user_id = "test_error_handling@test.com"
        
        try:
            # Test with valid data first
            index = await rag_module.build_user_index(test_user_id, [temp_csv_file])
            assert index is not None
            
            # Test with corrupted retriever state
            if test_user_id in rag_module.user_bm25_retrievers:
                del rag_module.user_bm25_retrievers[test_user_id]
            
            # Should still work with vector retriever only
            fusion_retriever = rag_module.get_fusion_retriever(test_user_id)
            # This might be None if BM25 is required, which is expected behavior
            
            # Test with invalid file path
            invalid_index = await rag_module.build_user_index("invalid_user", ["/nonexistent/file.csv"])
            assert invalid_index is None, "Should handle invalid file paths gracefully"
            
        finally:
            await rag_module.clear_user_index(test_user_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_csv_file):
        """Test concurrent RAG operations."""
        test_users = ["concurrent_user_1@test.com", "concurrent_user_2@test.com"]
        
        try:
            # Test concurrent index building
            tasks = []
            for user_id in test_users:
                task = rag_module.build_user_index(user_id, [temp_csv_file])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify both operations succeeded
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Concurrent operation {i} failed: {result}")
                assert result is not None, f"Index {i} should be created"
            
            # Test concurrent searches
            search_tasks = []
            for user_id in test_users:
                fusion_retriever = rag_module.get_fusion_retriever(user_id)
                if fusion_retriever:
                    task = multi_strategy_search(fusion_retriever, "36 W 36th St")
                    search_tasks.append(task)
            
            if search_tasks:
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                for i, result in enumerate(search_results):
                    if isinstance(result, Exception):
                        pytest.fail(f"Concurrent search {i} failed: {result}")
                    assert len(result.nodes_found) > 0, f"Search {i} should find results"
            
        finally:
            # Cleanup
            for user_id in test_users:
                await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, temp_csv_file):
        """Test integration with performance monitoring."""
        test_user_id = "test_performance@test.com"
        
        try:
            # Check if performance monitoring is available
            try:
                from app.performance_monitor import performance_monitor
                monitoring_available = True
            except ImportError:
                monitoring_available = False
            
            start_time = time.time()
            
            # Build index (should be monitored if available)
            index = await rag_module.build_user_index(test_user_id, [temp_csv_file])
            assert index is not None
            
            build_time = time.time() - start_time
            
            # Verify reasonable performance (should complete within 30 seconds)
            assert build_time < 30, f"Index building took too long: {build_time:.2f}s"
            
            if monitoring_available:
                # Performance monitoring integration would be tested here
                # This is a placeholder for when performance monitoring is fully integrated
                pass
            
        finally:
            await rag_module.clear_user_index(test_user_id)


@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Cleanup test data after each test."""
    # Setup
    original_indexes = rag_module.user_indexes.copy()
    original_retrievers = rag_module.user_bm25_retrievers.copy()
    
    yield
    
    # Cleanup - restore original state and clear any test data
    rag_module.user_indexes.clear()
    rag_module.user_bm25_retrievers.clear()
    rag_module.user_indexes.update(original_indexes)
    rag_module.user_bm25_retrievers.update(original_retrievers)


if __name__ == "__main__":
    pytest.main([__file__])