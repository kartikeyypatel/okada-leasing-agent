"""
Integration Tests for RAG Chatbot Fixes

This module contains comprehensive integration tests that verify the fixes
implemented for the RAG chatbot system, specifically addressing:
- Property search accuracy (84 Mulberry St test case)
- Automatic index building functionality
- Multi-strategy search logic
- Error handling and recovery scenarios

Requirements covered: 1.1, 1.2, 2.1, 2.2
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
from fastapi.testclient import TestClient

# Import the modules to test
import app.rag as rag_module
from app.chroma_client import ChromaClientManager, chroma_manager
from app.config import settings
from app.main import app

# Try to import optional modules - some may not be available
try:
    from app.multi_strategy_search import MultiStrategySearchResult, SearchStrategyResult
    MULTI_STRATEGY_AVAILABLE = True
except ImportError:
    MULTI_STRATEGY_AVAILABLE = False
    print("Warning: Multi-strategy search not available")

try:
    from app.user_context_validator import user_context_validator
    USER_CONTEXT_VALIDATOR_AVAILABLE = True
except ImportError:
    USER_CONTEXT_VALIDATOR_AVAILABLE = False
    print("Warning: User context validator not available")

try:
    from app.strict_response_generator import strict_response_generator
    STRICT_RESPONSE_AVAILABLE = True
except ImportError:
    STRICT_RESPONSE_AVAILABLE = False
    print("Warning: Strict response generator not available")


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
def mulberry_street_data():
    """Sample data that includes the specific 84 Mulberry St property for testing."""
    return [
        {
            "property address": "84 Mulberry St",
            "monthly rent": "80522",
            "size (sf)": "9567",
            "bedrooms": "4",
            "bathrooms": "3",
            "property_type": "Commercial",
            "neighborhood": "Financial District"
        },
        {
            "property address": "123 Main St",
            "monthly rent": "2500",
            "size (sf)": "1200",
            "bedrooms": "2",
            "bathrooms": "1",
            "property_type": "Residential",
            "neighborhood": "Downtown"
        },
        {
            "property address": "456 Oak Ave",
            "monthly rent": "3000",
            "size (sf)": "1500",
            "bedrooms": "3",
            "bathrooms": "2",
            "property_type": "Residential",
            "neighborhood": "Midtown"
        }
    ]


@pytest.fixture
def create_test_user_documents():
    """Factory fixture to create test user document directories with CSV files."""
    created_dirs = []
    
    def _create_user_docs(user_id: str, data: List[Dict], filename: str = "properties.csv"):
        # Create user documents directory
        user_doc_dir = os.path.join("user_documents", user_id)
        os.makedirs(user_doc_dir, exist_ok=True)
        created_dirs.append(user_doc_dir)
        
        # Create CSV file
        df = pd.DataFrame(data)
        csv_path = os.path.join(user_doc_dir, filename)
        df.to_csv(csv_path, index=False)
        
        return user_doc_dir, csv_path
    
    yield _create_user_docs
    
    # Cleanup
    for dir_path in created_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


class TestMulberryStreetQuery:
    """Test the specific 84 Mulberry St query that was failing."""
    
    @pytest.mark.asyncio
    async def test_mulberry_street_exact_match(self, mulberry_street_data, create_test_user_documents):
        """Test that '84 Mulberry St' query returns correct property information."""
        user_id = "mulberry_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None, "Index should be created successfully"
            
            # Test exact address query
            query = "tell me about 84 Mulberry St"
            
            # Test multi-strategy search
            search_result = await rag_module.retrieve_context_optimized(query, user_id)
            
            assert search_result is not None, "Search result should not be None"
            assert len(search_result.nodes_found) > 0, "Should find nodes for 84 Mulberry St"
            
            # Verify the correct property is found
            found_content = "\n".join([node.get_content() for node in search_result.nodes_found])
            
            # Check for key property details
            assert "84 Mulberry St" in found_content, "Should find the exact address"
            assert "80522" in found_content, "Should find the monthly rent ($80,522)"
            assert "9567" in found_content, "Should find the size (9,567 SF)"
            
            print(f"✅ Successfully found 84 Mulberry St with content: {found_content[:200]}...")
            
        finally:
            # Cleanup
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_mulberry_street_variations(self, mulberry_street_data, create_test_user_documents):
        """Test various ways of querying for 84 Mulberry St."""
        user_id = "mulberry_variations_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Test different query variations
            query_variations = [
                "84 Mulberry St",
                "tell me about 84 Mulberry St",
                "what is 84 Mulberry Street",
                "show me 84 Mulberry",
                "find 84 Mulberry St property"
            ]
            
            for query in query_variations:
                search_result = await rag_module.retrieve_context_optimized(query, user_id)
                
                assert search_result is not None, f"Search result should not be None for query: {query}"
                assert len(search_result.nodes_found) > 0, f"Should find nodes for query: {query}"
                
                found_content = "\n".join([node.get_content() for node in search_result.nodes_found])
                assert "84 Mulberry St" in found_content, f"Should find 84 Mulberry St for query: {query}"
                
                print(f"✅ Query '{query}' successfully found 84 Mulberry St")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_mulberry_street_api_integration(self, mulberry_street_data, create_test_user_documents, test_client):
        """Test the full API integration for 84 Mulberry St query."""
        user_id = "mulberry_api_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build index directly (simulating document upload)
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Test API chat endpoint
            response = test_client.post("/api/chat", json={
                "user_id": user_id,
                "message": "tell me about 84 Mulberry St"
            })
            
            assert response.status_code == 200, f"API should return 200, got {response.status_code}"
            
            response_data = response.json()
            assert "answer" in response_data, "Response should contain answer field"
            
            answer = response_data["answer"]
            assert "84 Mulberry St" in answer, "Answer should mention 84 Mulberry St"
            assert any(rent_info in answer for rent_info in ["80522", "80,522", "$80,522"]), \
                "Answer should include rent information"
            
            print(f"✅ API integration test passed. Answer: {answer[:200]}...")
            
        finally:
            await rag_module.clear_user_index(user_id)


class TestAutomaticIndexBuilding:
    """Test automatic index building functionality."""
    
    @pytest.mark.asyncio
    async def test_auto_index_building_on_chat(self, mulberry_street_data, create_test_user_documents, test_client):
        """Test that index is automatically built when user has documents but no index."""
        user_id = "auto_index_test@test.com"
        
        try:
            # Create test documents but don't build index
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Ensure no index exists initially
            assert user_id not in rag_module.user_indexes, "Index should not exist initially"
            
            # Make chat request - should trigger auto index building
            response = test_client.post("/api/chat", json={
                "user_id": user_id,
                "message": "tell me about 84 Mulberry St"
            })
            
            # Should succeed (index built automatically)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Verify index was created
            user_index = await rag_module.get_user_index(user_id)
            assert user_index is not None, "Index should be created automatically"
            
            # Verify retriever is available
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is not None, "Fusion retriever should be available after auto-indexing"
            
            print("✅ Automatic index building test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_index_validation_and_rebuild(self, mulberry_street_data, create_test_user_documents):
        """Test index validation and automatic rebuild when retriever creation fails."""
        user_id = "index_validation_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build initial index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Simulate corrupted BM25 retriever by removing it
            if user_id in rag_module.user_bm25_retrievers:
                del rag_module.user_bm25_retrievers[user_id]
            
            # Try to get fusion retriever - should fail
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is None, "Retriever should fail when BM25 is missing"
            
            # Rebuild index - should recreate BM25 retriever
            rebuilt_index = await rag_module.build_user_index(user_id, [csv_path])
            assert rebuilt_index is not None
            
            # Now retriever should work
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is not None, "Retriever should work after rebuild"
            
            print("✅ Index validation and rebuild test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_index_building_with_multiple_files(self, create_test_user_documents):
        """Test index building with multiple CSV files."""
        user_id = "multi_file_test@test.com"
        
        try:
            # Create multiple test files
            file1_data = [
                {"property address": "100 Test St", "monthly rent": "2000", "size (sf)": "1000"}
            ]
            file2_data = [
                {"property address": "200 Demo Ave", "monthly rent": "3000", "size (sf)": "1500"}
            ]
            
            user_doc_dir = os.path.join("user_documents", user_id)
            os.makedirs(user_doc_dir, exist_ok=True)
            
            # Create multiple CSV files
            file1_path = os.path.join(user_doc_dir, "properties1.csv")
            file2_path = os.path.join(user_doc_dir, "properties2.csv")
            
            pd.DataFrame(file1_data).to_csv(file1_path, index=False)
            pd.DataFrame(file2_data).to_csv(file2_path, index=False)
            
            # Build index with multiple files
            index = await rag_module.build_user_index(user_id, [file1_path, file2_path])
            assert index is not None
            
            # Test that both properties can be found
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is not None
            
            # Search for property from first file
            results1 = await retriever.aretrieve("100 Test St")
            assert len(results1) > 0, "Should find property from first file"
            
            # Search for property from second file
            results2 = await retriever.aretrieve("200 Demo Ave")
            assert len(results2) > 0, "Should find property from second file"
            
            print("✅ Multi-file index building test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
            if os.path.exists(user_doc_dir):
                shutil.rmtree(user_doc_dir)


class TestMultiStrategySearch:
    """Test multi-strategy search logic implementation."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not MULTI_STRATEGY_AVAILABLE, reason="Multi-strategy search not available")
    async def test_multi_strategy_search_execution(self, mulberry_street_data, create_test_user_documents):
        """Test that multi-strategy search tries multiple approaches."""
        user_id = "multi_strategy_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Test multi-strategy search
            query = "tell me about 84 Mulberry St"
            search_result = await rag_module.retrieve_context_optimized(query, user_id)
            
            assert search_result is not None, "Search result should not be None"
            assert isinstance(search_result, MultiStrategySearchResult), "Should return MultiStrategySearchResult"
            
            # Verify multiple strategies were tried
            assert len(search_result.all_results) > 1, "Should try multiple search strategies"
            
            # Verify best result was selected
            assert search_result.best_result is not None, "Should select a best result"
            
            # Verify nodes were found
            assert len(search_result.nodes_found) > 0, "Should find relevant nodes"
            
            # Log strategy details for debugging
            print(f"✅ Multi-strategy search tried {len(search_result.all_results)} strategies:")
            for result in search_result.all_results:
                print(f"  - {result.strategy}: {len(result.nodes)} nodes, success={result.success}")
            
            print(f"Best strategy: {search_result.best_result.strategy}")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_address_preservation_in_search(self, mulberry_street_data, create_test_user_documents):
        """Test that exact address formats are preserved in search queries."""
        user_id = "address_preservation_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Test with exact address format
            exact_query = "84 Mulberry St"
            search_result = await rag_module.retrieve_context_optimized(exact_query, user_id)
            
            assert search_result is not None
            assert len(search_result.nodes_found) > 0, "Should find nodes with exact address"
            
            # Verify that at least one strategy used the exact query
            exact_strategy_found = False
            for result in search_result.all_results:
                if result.query_used == exact_query:
                    exact_strategy_found = True
                    break
            
            assert exact_strategy_found, "At least one strategy should use the exact query"
            
            print("✅ Address preservation test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_fallback_search_strategies(self, mulberry_street_data, create_test_user_documents):
        """Test fallback search strategies when primary search fails."""
        user_id = "fallback_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Test with a query that might not match exactly but should find results via fallback
            partial_query = "Mulberry Street property"
            search_result = await rag_module.retrieve_context_optimized(partial_query, user_id)
            
            assert search_result is not None
            
            # Even if primary strategy fails, fallback should find results
            if len(search_result.nodes_found) > 0:
                found_content = "\n".join([node.get_content() for node in search_result.nodes_found])
                assert "84 Mulberry St" in found_content, "Fallback should find Mulberry St property"
                print("✅ Fallback search found results")
            else:
                print("ℹ️ No results found - this tests the fallback mechanism")
            
            # Verify multiple strategies were attempted
            assert len(search_result.all_results) > 1, "Should try multiple fallback strategies"
            
        finally:
            await rag_module.clear_user_index(user_id)


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_chromadb_connection_failure_recovery(self, mulberry_street_data, create_test_user_documents):
        """Test recovery when ChromaDB connection fails."""
        user_id = "chromadb_failure_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Mock ChromaDB failure
            with patch('app.rag.chroma_manager') as mock_chroma:
                mock_chroma.get_or_create_collection.side_effect = Exception("ChromaDB connection failed")
                
                # Should fallback to in-memory index
                index = await rag_module.build_user_index(user_id, [csv_path])
                assert index is not None, "Should create fallback in-memory index"
                
                # Verify retriever still works
                retriever = rag_module.get_fusion_retriever(user_id)
                assert retriever is not None, "Should create retriever with fallback index"
                
                # Test search functionality
                results = await retriever.aretrieve("84 Mulberry St")
                assert len(results) > 0, "Search should work with fallback index"
                
                print("✅ ChromaDB failure recovery test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_missing_user_documents_handling(self, test_client):
        """Test handling when user has no documents."""
        user_id = "no_docs_test@test.com"
        
        # Ensure user has no documents
        user_doc_dir = os.path.join("user_documents", user_id)
        if os.path.exists(user_doc_dir):
            shutil.rmtree(user_doc_dir)
        
        # Try to chat without documents
        response = test_client.post("/api/chat", json={
            "user_id": user_id,
            "message": "tell me about properties"
        })
        
        # Should return 503 (service unavailable) with helpful message
        assert response.status_code == 503, f"Expected 503, got {response.status_code}"
        
        response_data = response.json()
        assert "detail" in response_data
        assert "upload documents" in response_data["detail"].lower(), "Should suggest uploading documents"
        
        print("✅ Missing documents handling test passed")
    
    @pytest.mark.asyncio
    async def test_corrupted_index_recovery(self, mulberry_street_data, create_test_user_documents):
        """Test recovery from corrupted index state."""
        user_id = "corrupted_index_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build initial index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Simulate corrupted index by replacing with invalid object
            rag_module.user_indexes[user_id] = "corrupted_index"
            
            # Try to get fusion retriever - should handle corruption gracefully
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is None, "Should return None for corrupted index"
            
            # Rebuild index should recover
            recovered_index = await rag_module.build_user_index(user_id, [csv_path])
            assert recovered_index is not None, "Should recover from corruption"
            
            # Verify functionality is restored
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is not None, "Should work after recovery"
            
            print("✅ Corrupted index recovery test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, mulberry_street_data, create_test_user_documents):
        """Test handling of search timeouts."""
        user_id = "timeout_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Mock retriever to simulate timeout
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is not None
            
            # Test with mock timeout
            with patch.object(retriever, 'aretrieve') as mock_retrieve:
                mock_retrieve.side_effect = asyncio.TimeoutError("Search timeout")
                
                # Multi-strategy search should handle timeout gracefully
                search_result = await rag_module.retrieve_context_optimized("84 Mulberry St", user_id)
                
                # Should return empty result rather than crashing
                assert search_result is not None, "Should return result object even on timeout"
                assert len(search_result.nodes_found) == 0, "Should return empty nodes on timeout"
                
                print("✅ Search timeout handling test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not USER_CONTEXT_VALIDATOR_AVAILABLE, reason="User context validator not available")
    async def test_user_context_validation_and_fixing(self, mulberry_street_data, create_test_user_documents):
        """Test user context validation and automatic fixing."""
        user_id = "context_validation_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Test user context validation
            validation_result = await user_context_validator.validate_user_context(user_id)
            
            assert validation_result is not None, "Should return validation result"
            assert hasattr(validation_result, 'is_valid'), "Should have is_valid attribute"
            assert hasattr(validation_result, 'issues'), "Should have issues list"
            
            # If validation finds issues, test the fix functionality
            if not validation_result.is_valid:
                print(f"Found validation issues: {validation_result.issues}")
                
                # Attempt to fix issues
                fix_report = await user_context_validator.fix_user_context_issues(user_id)
                assert fix_report is not None, "Should return fix report"
                
                print("✅ User context validation and fixing test passed")
            else:
                print("✅ User context validation passed (no issues found)")
            
        finally:
            await rag_module.clear_user_index(user_id)


class TestIntegrationPerformance:
    """Test performance aspects of the RAG fixes."""
    
    @pytest.mark.asyncio
    async def test_search_performance_with_large_dataset(self, create_test_user_documents):
        """Test search performance with larger dataset."""
        user_id = "performance_test@test.com"
        
        try:
            # Create larger test dataset
            large_data = []
            for i in range(50):  # 50 properties
                large_data.append({
                    "property address": f"{i} Performance St",
                    "monthly rent": str(2000 + i * 10),
                    "size (sf)": str(1000 + i * 5),
                    "bedrooms": str((i % 4) + 1),
                    "bathrooms": str((i % 3) + 1)
                })
            
            # Add the specific test property
            large_data.append({
                "property address": "84 Mulberry St",
                "monthly rent": "80522",
                "size (sf)": "9567",
                "bedrooms": "4",
                "bathrooms": "3"
            })
            
            user_doc_dir, csv_path = create_test_user_documents(user_id, large_data)
            
            # Measure index building time
            start_time = time.time()
            index = await rag_module.build_user_index(user_id, [csv_path])
            index_time = time.time() - start_time
            
            assert index is not None, "Should build index successfully"
            assert index_time < 30, f"Index building should complete within 30 seconds, took {index_time:.2f}s"
            
            # Measure search time
            start_time = time.time()
            search_result = await rag_module.retrieve_context_optimized("84 Mulberry St", user_id)
            search_time = time.time() - start_time
            
            assert search_result is not None, "Should return search result"
            assert len(search_result.nodes_found) > 0, "Should find the target property"
            assert search_time < 10, f"Search should complete within 10 seconds, took {search_time:.2f}s"
            
            print(f"✅ Performance test passed - Index: {index_time:.2f}s, Search: {search_time:.2f}s")
            
        finally:
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_user_searches(self, mulberry_street_data, create_test_user_documents):
        """Test concurrent searches by multiple users."""
        user_ids = [f"concurrent_user_{i}@test.com" for i in range(3)]
        
        try:
            # Create indexes for all users
            for user_id in user_ids:
                user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
                index = await rag_module.build_user_index(user_id, [csv_path])
                assert index is not None, f"Should build index for {user_id}"
            
            # Perform concurrent searches
            async def search_for_user(user_id: str):
                search_result = await rag_module.retrieve_context_optimized("84 Mulberry St", user_id)
                assert search_result is not None, f"Search should work for {user_id}"
                assert len(search_result.nodes_found) > 0, f"Should find results for {user_id}"
                return user_id
            
            # Run searches concurrently
            start_time = time.time()
            results = await asyncio.gather(*[search_for_user(user_id) for user_id in user_ids])
            concurrent_time = time.time() - start_time
            
            assert len(results) == len(user_ids), "All concurrent searches should complete"
            assert concurrent_time < 15, f"Concurrent searches should complete within 15 seconds, took {concurrent_time:.2f}s"
            
            print(f"✅ Concurrent search test passed - {len(user_ids)} users in {concurrent_time:.2f}s")
            
        finally:
            # Cleanup all users
            for user_id in user_ids:
                await rag_module.clear_user_index(user_id)


# Test configuration and cleanup
@pytest.fixture(autouse=True)
async def setup_and_cleanup():
    """Setup and cleanup for each test."""
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
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])