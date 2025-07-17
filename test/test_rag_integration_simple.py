"""
Simple Integration Tests for RAG Chatbot Fixes

This module contains focused integration tests that verify the core fixes
implemented for the RAG chatbot system, specifically addressing:
- Property search accuracy (84 Mulberry St test case)
- Automatic index building functionality
- Basic error handling scenarios

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
    async def test_mulberry_street_basic_search(self, mulberry_street_data, create_test_user_documents):
        """Test that '84 Mulberry St' query returns correct property information using basic search."""
        user_id = "mulberry_basic_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Build user index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None, "Index should be created successfully"
            
            # Test basic retriever functionality
            retriever = rag_module.get_fusion_retriever(user_id)
            if retriever is not None:
                # Test direct retrieval
                results = await retriever.aretrieve("84 Mulberry St")
                assert len(results) > 0, "Should find nodes for 84 Mulberry St"
                
                # Verify the correct property is found
                found_content = "\n".join([node.get_content() for node in results])
                
                # Check for key property details
                assert "84 Mulberry St" in found_content, "Should find the exact address"
                assert "80522" in found_content, "Should find the monthly rent ($80,522)"
                assert "9567" in found_content, "Should find the size (9,567 SF)"
                
                print(f"✅ Successfully found 84 Mulberry St with basic search")
            else:
                # Test fallback to basic context retrieval
                context = await rag_module.retrieve_context("84 Mulberry St", user_id)
                assert context is not None, "Should get context even without fusion retriever"
                assert "84 Mulberry St" in context, "Context should contain the address"
                print(f"✅ Found 84 Mulberry St using fallback context retrieval")
            
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
                "Mulberry Street"
            ]
            
            retriever = rag_module.get_fusion_retriever(user_id)
            
            for query in query_variations:
                if retriever is not None:
                    results = await retriever.aretrieve(query)
                    if len(results) > 0:
                        found_content = "\n".join([node.get_content() for node in results])
                        if "84 Mulberry St" in found_content:
                            print(f"✅ Query '{query}' successfully found 84 Mulberry St")
                        else:
                            print(f"ℹ️ Query '{query}' found results but not specific address")
                    else:
                        print(f"ℹ️ Query '{query}' found no results")
                else:
                    # Fallback test
                    context = await rag_module.retrieve_context(query, user_id)
                    if context and "84 Mulberry St" in context:
                        print(f"✅ Query '{query}' found address via fallback")
            
        finally:
            await rag_module.clear_user_index(user_id)


class TestAutomaticIndexBuilding:
    """Test automatic index building functionality."""
    
    @pytest.mark.asyncio
    async def test_index_building_basic(self, mulberry_street_data, create_test_user_documents):
        """Test basic index building functionality."""
        user_id = "index_building_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Ensure no index exists initially
            assert user_id not in rag_module.user_indexes, "Index should not exist initially"
            
            # Build index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None, "Index should be created successfully"
            
            # Verify index was cached
            assert user_id in rag_module.user_indexes, "Index should be cached"
            
            # Test that we can get the index again
            retrieved_index = await rag_module.get_user_index(user_id)
            assert retrieved_index is not None, "Should retrieve cached index"
            
            print("✅ Basic index building test passed")
            
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
            
            # Test basic search functionality
            retriever = rag_module.get_fusion_retriever(user_id)
            if retriever is not None:
                # Search for property from first file
                results1 = await retriever.aretrieve("100 Test St")
                assert len(results1) > 0, "Should find property from first file"
                
                # Search for property from second file
                results2 = await retriever.aretrieve("200 Demo Ave")
                assert len(results2) > 0, "Should find property from second file"
                
                print("✅ Multi-file index building test passed")
            else:
                print("ℹ️ Multi-file test completed (fusion retriever not available)")
            
        finally:
            await rag_module.clear_user_index(user_id)
            if os.path.exists(user_doc_dir):
                shutil.rmtree(user_doc_dir)


class TestErrorHandlingAndRecovery:
    """Test basic error handling and recovery scenarios."""
    
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
    async def test_index_clearing(self, mulberry_street_data, create_test_user_documents):
        """Test index clearing functionality."""
        user_id = "index_clearing_test@test.com"
        
        try:
            # Create test documents and build index
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Verify index exists
            assert user_id in rag_module.user_indexes
            
            # Clear index
            success = await rag_module.clear_user_index(user_id)
            assert success, "Index clearing should succeed"
            
            # Verify index is cleared
            assert user_id not in rag_module.user_indexes, "Index should be cleared from cache"
            
            print("✅ Index clearing test passed")
            
        finally:
            # Ensure cleanup
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_chromadb_fallback(self, mulberry_street_data, create_test_user_documents):
        """Test fallback to in-memory index when ChromaDB fails."""
        user_id = "chromadb_fallback_test@test.com"
        
        try:
            # Create test documents
            user_doc_dir, csv_path = create_test_user_documents(user_id, mulberry_street_data)
            
            # Mock ChromaDB failure during index building
            with patch('app.rag.chroma_manager') as mock_chroma:
                mock_chroma.get_or_create_collection.side_effect = Exception("ChromaDB connection failed")
                
                # Should fallback to in-memory index
                index = await rag_module.build_user_index(user_id, [csv_path])
                assert index is not None, "Should create fallback in-memory index"
                
                # Test basic functionality with fallback
                context = await rag_module.retrieve_context("84 Mulberry St", user_id)
                assert context is not None, "Should get context with fallback index"
                
                print("✅ ChromaDB fallback test passed")
            
        finally:
            await rag_module.clear_user_index(user_id)


class TestBasicPerformance:
    """Test basic performance aspects."""
    
    @pytest.mark.asyncio
    async def test_index_building_performance(self, create_test_user_documents):
        """Test index building performance with moderate dataset."""
        user_id = "performance_test@test.com"
        
        try:
            # Create moderate test dataset
            test_data = []
            for i in range(20):  # 20 properties
                test_data.append({
                    "property address": f"{i} Performance St",
                    "monthly rent": str(2000 + i * 10),
                    "size (sf)": str(1000 + i * 5),
                    "bedrooms": str((i % 4) + 1),
                    "bathrooms": str((i % 3) + 1)
                })
            
            # Add the specific test property
            test_data.append({
                "property address": "84 Mulberry St",
                "monthly rent": "80522",
                "size (sf)": "9567",
                "bedrooms": "4",
                "bathrooms": "3"
            })
            
            user_doc_dir, csv_path = create_test_user_documents(user_id, test_data)
            
            # Measure index building time
            start_time = time.time()
            index = await rag_module.build_user_index(user_id, [csv_path])
            index_time = time.time() - start_time
            
            assert index is not None, "Should build index successfully"
            assert index_time < 30, f"Index building should complete within 30 seconds, took {index_time:.2f}s"
            
            # Test basic search
            retriever = rag_module.get_fusion_retriever(user_id)
            if retriever is not None:
                start_time = time.time()
                results = await retriever.aretrieve("84 Mulberry St")
                search_time = time.time() - start_time
                
                assert len(results) > 0, "Should find the target property"
                assert search_time < 5, f"Search should complete within 5 seconds, took {search_time:.2f}s"
                
                print(f"✅ Performance test passed - Index: {index_time:.2f}s, Search: {search_time:.2f}s")
            else:
                print(f"✅ Performance test passed - Index: {index_time:.2f}s (retriever not available)")
            
        finally:
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