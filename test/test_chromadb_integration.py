"""
Comprehensive Tests for ChromaDB Integration

This module contains unit tests, integration tests, and performance tests
for the ChromaDB integration in the Okada Leasing Agent.
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
from app.config import settings
from app.migration import DataMigrationManager
from app.main import app
from fastapi.testclient import TestClient

# Test fixtures
@pytest.fixture
def temp_user_docs_dir():
    """Create a temporary directory with test user documents."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test user directories and CSV files
    test_users = {
        "user1@test.com": [
            {"property address": "123 Main St", "monthly rent": "2500", "size (sf)": "1200"},
            {"property address": "456 Oak Ave", "monthly rent": "3000", "size (sf)": "1500"}
        ],
        "user2@test.com": [
            {"property address": "789 Pine Rd", "monthly rent": "2000", "size (sf)": "1000"}
        ]
    }
    
    for user_id, properties in test_users.items():
        user_dir = os.path.join(temp_dir, user_id)
        os.makedirs(user_dir)
        
        # Create CSV file
        df = pd.DataFrame(properties)
        csv_path = os.path.join(user_dir, "properties.csv")
        df.to_csv(csv_path, index=False)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing."""
    mock_client = Mock()
    mock_collection = Mock()
    
    # Configure mock collection
    mock_collection.get.return_value = {
        "documents": ["test document"],
        "metadatas": [{"user_id": "test@test.com"}],
        "ids": ["test_id"]
    }
    
    # Configure mock client
    mock_client.get_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = []
    
    return mock_client


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


# Unit Tests for ChromaDB Client Manager
class TestChromaClientManager:
    """Test ChromaDB client manager functionality."""
    
    def test_client_manager_initialization(self):
        """Test ChromaClientManager initialization."""
        manager = ChromaClientManager()
        assert manager._client is None
    
    @patch('app.chroma_client.chromadb.PersistentClient')
    def test_get_client_local(self, mock_persistent_client):
        """Test getting local ChromaDB client."""
        mock_client = Mock()
        mock_persistent_client.return_value = mock_client
        
        manager = ChromaClientManager()
        
        # Test with local configuration
        with patch.object(settings, 'CHROMA_HOST', None):
            with patch.object(settings, 'CHROMA_PORT', None):
                client = manager.get_client()
                
                assert client == mock_client
                mock_persistent_client.assert_called_once_with(
                    path=settings.CHROMA_PERSIST_DIRECTORY
                )
    
    @patch('app.chroma_client.chromadb.HttpClient')
    def test_get_client_remote(self, mock_http_client):
        """Test getting remote ChromaDB client."""
        mock_client = Mock()
        mock_http_client.return_value = mock_client
        
        manager = ChromaClientManager()
        
        # Test with remote configuration
        with patch.object(settings, 'CHROMA_HOST', 'localhost'):
            with patch.object(settings, 'CHROMA_PORT', 8000):
                client = manager.get_client()
                
                assert client == mock_client
                mock_http_client.assert_called_once_with(
                    host='localhost',
                    port=8000
                )
    
    @patch('app.chroma_client.chromadb.PersistentClient')
    def test_get_or_create_collection(self, mock_persistent_client):
        """Test getting or creating a collection."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        
        # Test getting existing collection
        mock_client.get_collection.return_value = mock_collection
        
        manager = ChromaClientManager()
        collection = manager.get_or_create_collection("test@test.com")
        
        assert collection == mock_collection
        mock_client.get_collection.assert_called_once()
    
    @patch('app.chroma_client.chromadb.PersistentClient')
    def test_get_or_create_collection_new(self, mock_persistent_client):
        """Test creating a new collection."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_persistent_client.return_value = mock_client
        
        # Simulate collection doesn't exist
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        
        manager = ChromaClientManager()
        collection = manager.get_or_create_collection("test@test.com")
        
        assert collection == mock_collection
        mock_client.create_collection.assert_called_once()
    
    @patch('app.chroma_client.chromadb.PersistentClient')
    def test_delete_user_collection(self, mock_persistent_client):
        """Test deleting a user collection."""
        mock_client = Mock()
        mock_persistent_client.return_value = mock_client
        
        manager = ChromaClientManager()
        result = manager.delete_user_collection("test@test.com")
        
        assert result is True
        mock_client.delete_collection.assert_called_once()


# Unit Tests for RAG Module
class TestRAGModule:
    """Test RAG module ChromaDB integration."""
    
    @pytest.mark.asyncio
    async def test_get_user_index_cached(self):
        """Test getting cached user index."""
        # Setup cached index
        mock_index = Mock()
        rag_module.user_indexes["test@test.com"] = mock_index
        
        result = await rag_module.get_user_index("test@test.com")
        assert result == mock_index
        
        # Cleanup
        del rag_module.user_indexes["test@test.com"]
    
    @pytest.mark.asyncio
    @patch('app.rag.chroma_manager')
    @patch('app.rag.ChromaVectorStore')
    @patch('app.rag.VectorStoreIndex')
    async def test_get_user_index_new(self, mock_index_class, mock_vector_store_class, mock_chroma_manager):
        """Test creating new user index."""
        # Setup mocks
        mock_collection = Mock()
        mock_vector_store = Mock()
        mock_index = Mock()
        
        mock_chroma_manager.get_or_create_collection.return_value = mock_collection
        mock_vector_store_class.return_value = mock_vector_store
        mock_index_class.from_vector_store.return_value = mock_index
        
        # Test
        result = await rag_module.get_user_index("test@test.com")
        
        assert result == mock_index
        assert "test@test.com" in rag_module.user_indexes
        
        # Cleanup
        if "test@test.com" in rag_module.user_indexes:
            del rag_module.user_indexes["test@test.com"]
    
    @pytest.mark.asyncio
    @patch('app.rag.chroma_manager')
    async def test_get_user_index_fallback(self, mock_chroma_manager):
        """Test fallback to in-memory index on ChromaDB failure."""
        # Setup mock to fail
        mock_chroma_manager.get_or_create_collection.side_effect = Exception("ChromaDB failed")
        
        # Test
        result = await rag_module.get_user_index("test@test.com")
        
        # Should return fallback index
        assert result is not None
        assert "test@test.com" in rag_module.user_indexes
        
        # Cleanup
        if "test@test.com" in rag_module.user_indexes:
            del rag_module.user_indexes["test@test.com"]
    
    @pytest.mark.asyncio
    async def test_clear_user_index(self):
        """Test clearing user index."""
        # Setup test data
        rag_module.user_indexes["test@test.com"] = Mock()
        rag_module.user_bm25_retrievers["test@test.com"] = Mock()
        
        # Test
        with patch('app.rag.chroma_manager') as mock_chroma_manager:
            mock_chroma_manager.delete_user_collection.return_value = True
            
            result = await rag_module.clear_user_index("test@test.com")
            
            assert result is True
            assert "test@test.com" not in rag_module.user_indexes
            assert "test@test.com" not in rag_module.user_bm25_retrievers


# Integration Tests
class TestChromaDBIntegration:
    """Integration tests for ChromaDB functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_workflow(self, temp_user_docs_dir):
        """Test complete document upload and search workflow."""
        user_id = "integration_test@test.com"
        
        # Create test CSV file
        test_data = [
            {"property address": "123 Integration St", "monthly rent": "2500", "size (sf)": "1200"},
            {"property address": "456 Test Ave", "monthly rent": "3000", "size (sf)": "1500"}
        ]
        
        df = pd.DataFrame(test_data)
        csv_path = os.path.join(temp_user_docs_dir, "test.csv")
        df.to_csv(csv_path, index=False)
        
        try:
            # Test building index
            index = await rag_module.build_user_index(user_id, [csv_path])
            assert index is not None
            
            # Test getting index
            retrieved_index = await rag_module.get_user_index(user_id)
            assert retrieved_index is not None
            
            # Test search functionality
            retriever = rag_module.get_fusion_retriever(user_id)
            assert retriever is not None
            
            # Test search
            results = await retriever.aretrieve("Integration St")
            assert len(results) > 0
            
        finally:
            # Cleanup
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, temp_user_docs_dir):
        """Test that users have isolated collections."""
        user1_id = "user1@isolation.test"
        user2_id = "user2@isolation.test"
        
        # Create test data for each user
        user1_data = [{"property address": "User1 Property", "monthly rent": "2000"}]
        user2_data = [{"property address": "User2 Property", "monthly rent": "3000"}]
        
        user1_csv = os.path.join(temp_user_docs_dir, "user1.csv")
        user2_csv = os.path.join(temp_user_docs_dir, "user2.csv")
        
        pd.DataFrame(user1_data).to_csv(user1_csv, index=False)
        pd.DataFrame(user2_data).to_csv(user2_csv, index=False)
        
        try:
            # Build indexes for both users
            index1 = await rag_module.build_user_index(user1_id, [user1_csv])
            index2 = await rag_module.build_user_index(user2_id, [user2_csv])
            
            assert index1 is not None
            assert index2 is not None
            
            # Test that each user gets their own data
            retriever1 = rag_module.get_fusion_retriever(user1_id)
            retriever2 = rag_module.get_fusion_retriever(user2_id)
            
            results1 = await retriever1.aretrieve("User1")
            results2 = await retriever2.aretrieve("User2")
            
            # User1 should find their property
            assert len(results1) > 0
            assert "User1 Property" in results1[0].text
            
            # User2 should find their property
            assert len(results2) > 0
            assert "User2 Property" in results2[0].text
            
        finally:
            # Cleanup
            await rag_module.clear_user_index(user1_id)
            await rag_module.clear_user_index(user2_id)


# Performance Tests
class TestPerformance:
    """Performance tests for ChromaDB integration."""
    
    @pytest.mark.asyncio
    async def test_indexing_performance(self, temp_user_docs_dir):
        """Test indexing performance with larger datasets."""
        user_id = "perf_test@test.com"
        
        # Create larger test dataset
        large_data = []
        for i in range(100):  # 100 properties
            large_data.append({
                "property address": f"{i} Performance St",
                "monthly rent": str(2000 + i * 10),
                "size (sf)": str(1000 + i * 5)
            })
        
        df = pd.DataFrame(large_data)
        csv_path = os.path.join(temp_user_docs_dir, "large_test.csv")
        df.to_csv(csv_path, index=False)
        
        try:
            # Measure indexing time
            start_time = time.time()
            index = await rag_module.build_user_index(user_id, [csv_path])
            indexing_time = time.time() - start_time
            
            assert index is not None
            assert indexing_time < 30  # Should complete within 30 seconds
            
            # Measure search time
            retriever = rag_module.get_fusion_retriever(user_id)
            
            start_time = time.time()
            results = await retriever.aretrieve("Performance St")
            search_time = time.time() - start_time
            
            assert len(results) > 0
            assert search_time < 5  # Should complete within 5 seconds
            
        finally:
            # Cleanup
            await rag_module.clear_user_index(user_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, temp_user_docs_dir):
        """Test concurrent access to ChromaDB."""
        user_ids = [f"concurrent_user_{i}@test.com" for i in range(5)]
        
        # Create test data for each user
        csv_files = []
        for i, user_id in enumerate(user_ids):
            data = [{"property address": f"Concurrent Property {i}", "monthly rent": "2000"}]
            csv_path = os.path.join(temp_user_docs_dir, f"concurrent_{i}.csv")
            pd.DataFrame(data).to_csv(csv_path, index=False)
            csv_files.append(csv_path)
        
        try:
            # Test concurrent indexing
            tasks = []
            for i, user_id in enumerate(user_ids):
                task = rag_module.build_user_index(user_id, [csv_files[i]])
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_time = time.time() - start_time
            
            # All should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result is not None
            
            # Should complete within reasonable time
            assert concurrent_time < 60
            
        finally:
            # Cleanup
            for user_id in user_ids:
                await rag_module.clear_user_index(user_id)


# API Integration Tests
class TestAPIIntegration:
    """Test API endpoints with ChromaDB integration."""
    
    def test_health_check_endpoint(self, test_client):
        """Test ChromaDB health check endpoint."""
        response = test_client.get("/api/health/chromadb")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
    
    def test_chat_endpoint_with_chromadb(self, test_client):
        """Test chat endpoint with ChromaDB integration."""
        # This would require setting up test data and mocking
        # For now, just test that the endpoint exists
        response = test_client.post("/api/chat", json={
            "user_id": "test@test.com",
            "message": "hello"
        })
        # Should return 503 if no documents are indexed
        assert response.status_code in [200, 503]


# Migration Tests
class TestDataMigration:
    """Test data migration utilities."""
    
    @pytest.mark.asyncio
    async def test_discover_existing_documents(self, temp_user_docs_dir):
        """Test discovering existing user documents."""
        # Setup test environment
        original_user_docs = "user_documents"
        
        # Temporarily replace user_documents path
        with patch('app.migration.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.iterdir.return_value = [
                type('MockPath', (), {'name': 'user1@test.com', 'is_dir': lambda: True, 'glob': lambda pattern: [
                    type('MockFile', (), {'name': 'test.csv'})()
                ]})()
            ]
            
            manager = DataMigrationManager()
            result = await manager.discover_existing_documents()
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_validate_chromadb_connection(self):
        """Test ChromaDB connection validation."""
        manager = DataMigrationManager()
        
        # Test with mocked successful connection
        with patch('app.migration.chroma_manager') as mock_manager:
            mock_client = Mock()
            mock_manager.get_client.return_value = mock_client
            mock_client.list_collections.return_value = []
            
            result = await manager.validate_chromadb_connection()
            assert result is True


# Test Configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Clear any existing indexes
    rag_module.user_indexes.clear()
    rag_module.user_bm25_retrievers.clear()
    
    yield
    
    # Cleanup after each test
    rag_module.user_indexes.clear()
    rag_module.user_bm25_retrievers.clear()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 