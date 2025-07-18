import chromadb
import hashlib
import logging
import time
import asyncio
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from app.config import settings
from app.error_handler import handle_sync_error, ErrorCategory

logger = logging.getLogger(__name__)

class ChromaClientManager:
    """
    Optimized ChromaDB client manager with enhanced connection management,
    collection isolation, health monitoring, and efficient document storage patterns.
    """
    
    def __init__(self):
        self._client: Optional[chromadb.Client] = None  # type: ignore
        self._connection_pool: Dict[str, chromadb.Client] = {}  # type: ignore
        self._collection_cache: Dict[str, Tuple[chromadb.Collection, datetime]] = {}  # type: ignore
        self._health_stats: Dict[str, Any] = {
            "total_connections": 0,
            "failed_connections": 0,
            "collection_operations": 0,
            "last_health_check": None,
            "connection_errors": []
        }
        self._cache_ttl = timedelta(minutes=30)  # Collection cache TTL
        self._max_retries = 3
        self._retry_delay = 1.0
    
    def get_client(self, connection_key: str = "default") -> chromadb.Client:  # type: ignore
        """
        Get or create ChromaDB client with enhanced connection management and retry logic.
        
        Args:
            connection_key: Key for connection pooling (default: "default")
            
        Returns:
            ChromaDB client instance
        """
        # Check connection pool first
        if connection_key in self._connection_pool:
            try:
                # Test connection health
                client = self._connection_pool[connection_key]
                # Simple health check - try to list collections
                client.list_collections()
                return client
            except Exception as e:
                logger.warning(f"Cached connection {connection_key} failed health check: {e}")
                # Remove failed connection from pool
                del self._connection_pool[connection_key]
        
        # Create new connection with retry logic
        for attempt in range(self._max_retries):
            try:
                self._health_stats["total_connections"] += 1
                
                if settings.CHROMA_HOST and settings.CHROMA_PORT:
                    # Remote ChromaDB server with enhanced configuration
                    client = chromadb.HttpClient(
                        host=settings.CHROMA_HOST,
                        port=settings.CHROMA_PORT
                    )
                    logger.info(f"Connected to remote ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
                else:
                    # Local persistent ChromaDB with optimizations
                    client = chromadb.PersistentClient(
                        path=settings.CHROMA_PERSIST_DIRECTORY
                    )
                    logger.info(f"Connected to local ChromaDB at {settings.CHROMA_PERSIST_DIRECTORY}")
                
                # Test the connection
                client.list_collections()
                
                # Cache the successful connection
                self._connection_pool[connection_key] = client
                if connection_key == "default":
                    self._client = client
                
                logger.debug(f"ChromaDB connection {connection_key} established successfully")
                return client
                
            except Exception as e:
                self._health_stats["failed_connections"] += 1
                self._health_stats["connection_errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt + 1,
                    "error": str(e)
                })
                
                if attempt < self._max_retries - 1:
                    wait_time = self._retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"ChromaDB connection attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    logger.error(f"All ChromaDB connection attempts failed. Last error: {e}")
                    
                    # Handle ChromaDB connection errors with error handler
                    recovery_successful, user_message = handle_sync_error(
                        e, "chromadb_connection", 
                        additional_data={
                            "host": settings.CHROMA_HOST, 
                            "port": settings.CHROMA_PORT,
                            "attempts": self._max_retries
                        }
                    )
                    
                    if user_message:
                        logger.error(f"User message: {user_message}")
                    raise
    
    def get_or_create_collection(self, user_id: str, use_cache: bool = True) -> chromadb.Collection:  # type: ignore
        """
        Get or create a user-specific ChromaDB collection with enhanced validation and caching.
        
        Args:
            user_id: User identifier
            use_cache: Whether to use collection caching (default: True)
            
        Returns:
            ChromaDB collection instance
        """
        # Enhanced validation and logging for user context debugging
        if not user_id or not isinstance(user_id, str):
            raise ValueError(f"Invalid user_id provided: {user_id}")
        
        # Validate user_id doesn't contain problematic characters for collection naming
        problematic_chars = ['\x00', '\n', '\r', '\t']  # ChromaDB specific problematic chars
        for char in problematic_chars:
            if char in user_id:
                raise ValueError(f"User ID contains invalid character: {repr(char)}")
        
        # Generate collection name
        user_id_hash = hashlib.md5(user_id.encode('utf-8')).hexdigest()
        collection_name = f"{settings.CHROMA_COLLECTION_PREFIX}{user_id_hash}"
        
        # Check cache first if enabled
        if use_cache and collection_name in self._collection_cache:
            cached_collection, cache_time = self._collection_cache[collection_name]
            if datetime.now() - cache_time < self._cache_ttl:
                try:
                    # Validate cached collection is still healthy
                    cached_collection.count()
                    logger.debug(f"Using cached collection: {collection_name} for user: {user_id}")
                    return cached_collection
                except Exception as e:
                    logger.warning(f"Cached collection {collection_name} failed health check: {e}")
                    # Remove from cache and continue with fresh retrieval
                    del self._collection_cache[collection_name]
        
        logger.info(f"Getting or creating ChromaDB collection for user: {user_id}")
        self._health_stats["collection_operations"] += 1
        
        client = self.get_client()
        
        logger.debug(f"Generated collection name: {collection_name} for user: {user_id} (hash: {user_id_hash})")
        
        # Validate collection name length (ChromaDB requirement)
        if len(collection_name) > 63:
            logger.error(f"Collection name too long ({len(collection_name)} chars): {collection_name}")
            raise ValueError(f"Collection name exceeds 63 character limit")
        
        # Additional validation for ChromaDB collection name requirements
        if not collection_name.replace('_', '').replace('-', '').isalnum():
            logger.warning(f"Collection name contains non-alphanumeric characters: {collection_name}")
        
        collection = None
        
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection: {collection_name} for user: {user_id}")
            
            # Enhanced collection validation and stats for debugging
            try:
                doc_count = collection.count()
                logger.info(f"Collection {collection_name} contains {doc_count} documents")
                
                # Additional validation: check if collection is accessible
                if doc_count > 0:
                    # Try to peek at documents to ensure collection is healthy
                    try:
                        sample = collection.peek(limit=1)
                        logger.debug(f"Collection {collection_name} is healthy - sample retrieved successfully")
                    except Exception as peek_error:
                        logger.warning(f"Collection {collection_name} may be corrupted - peek failed: {peek_error}")
                        
            except Exception as count_error:
                logger.warning(f"Could not count documents in collection {collection_name}: {count_error}")
                # Collection might be corrupted, but we'll still return it and let the caller handle it
                
        except Exception as get_error:
            # Collection doesn't exist, create it
            logger.info(f"Collection {collection_name} not found (error: {get_error}), creating new collection for user: {user_id}")
            try:
                # Create collection with enhanced metadata for debugging and management
                collection_metadata = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "created_by": "chroma_client_manager",
                    "version": "2.0",
                    "user_id_hash": user_id_hash,
                    "collection_purpose": "rag_document_storage"
                }
                
                collection = client.create_collection(
                    name=collection_name,
                    metadata=collection_metadata
                )
                logger.info(f"Successfully created new collection: {collection_name} for user: {user_id}")
                
                # Verify the collection was created successfully
                try:
                    verify_count = collection.count()
                    logger.debug(f"New collection {collection_name} verified with {verify_count} documents")
                except Exception as verify_error:
                    logger.warning(f"Could not verify new collection {collection_name}: {verify_error}")
                    
            except Exception as create_error:
                logger.error(f"Failed to create collection {collection_name} for user {user_id}: {create_error}")
                
                # Enhanced error handling - try to provide more specific error information
                if "already exists" in str(create_error).lower():
                    logger.info(f"Collection {collection_name} was created by another process, attempting to retrieve it")
                    try:
                        collection = client.get_collection(name=collection_name)
                        logger.info(f"Successfully retrieved collection {collection_name} after creation race condition")
                    except Exception as retry_error:
                        logger.error(f"Failed to retrieve collection after creation race condition: {retry_error}")
                        raise create_error
                else:
                    raise create_error
        
        # Final validation before returning
        if not collection:
            raise RuntimeError(f"Collection creation/retrieval failed for user {user_id}")
        
        # Cache the collection if caching is enabled
        if use_cache:
            self._collection_cache[collection_name] = (collection, datetime.now())
            logger.debug(f"Cached collection {collection_name} for future use")
        
        # Log final collection info for debugging
        try:
            final_count = collection.count()
            logger.debug(f"Returning collection {collection_name} for user {user_id} with {final_count} documents")
        except Exception:
            logger.debug(f"Returning collection {collection_name} for user {user_id} (count unavailable)")
        
        return collection
    
    def delete_user_collection(self, user_id: str, clear_cache: bool = True) -> bool:
        """
        Delete a user's ChromaDB collection with enhanced cleanup.
        
        Args:
            user_id: User identifier
            clear_cache: Whether to clear the collection from cache (default: True)
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            client = self.get_client()
            user_id_hash = hashlib.md5(user_id.encode()).hexdigest()
            collection_name = f"{settings.CHROMA_COLLECTION_PREFIX}{user_id_hash}"
            
            # Remove from cache first if requested
            if clear_cache and collection_name in self._collection_cache:
                del self._collection_cache[collection_name]
                logger.debug(f"Removed collection {collection_name} from cache")
            
            # Delete the collection
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name} for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection for user {user_id}: {e}")
            return False
    
    def get_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a user's collection.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing collection statistics
        """
        try:
            collection = self.get_or_create_collection(user_id, use_cache=False)
            
            stats = {
                "user_id": user_id,
                "collection_name": collection.name,
                "document_count": collection.count(),
                "metadata": collection.metadata,
                "health_status": "healthy",
                "last_checked": datetime.now().isoformat()
            }
            
            # Try to get sample documents for additional validation
            try:
                if stats["document_count"] > 0:
                    sample = collection.peek(limit=3)
                    documents = sample.get("documents", [])
                    embeddings = sample.get("embeddings", [])
                    metadatas = sample.get("metadatas", [])
                    
                    stats["sample_document_count"] = len(documents) if documents else 0
                    stats["has_embeddings"] = len(embeddings) > 0 if embeddings else False
                    stats["has_metadata"] = len(metadatas) > 0 if metadatas else False
                else:
                    stats["sample_document_count"] = 0
                    stats["has_embeddings"] = False
                    stats["has_metadata"] = False
                    
            except Exception as sample_error:
                stats["health_status"] = "degraded"
                stats["sample_error"] = str(sample_error)
                logger.warning(f"Could not get sample documents for {user_id}: {sample_error}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "health_status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def optimize_collection(self, user_id: str) -> Dict[str, Any]:
        """
        Optimize a user's collection for better performance.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing optimization results
        """
        optimization_result = {
            "user_id": user_id,
            "optimizations_applied": [],
            "performance_improvements": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            collection = self.get_or_create_collection(user_id, use_cache=False)
            
            # Clear cache to force fresh retrieval
            user_id_hash = hashlib.md5(user_id.encode()).hexdigest()
            collection_name = f"{settings.CHROMA_COLLECTION_PREFIX}{user_id_hash}"
            
            if collection_name in self._collection_cache:
                del self._collection_cache[collection_name]
                optimization_result["optimizations_applied"].append("cache_cleared")
            
            # Test collection performance
            start_time = time.time()
            doc_count = collection.count()
            count_time = time.time() - start_time
            
            optimization_result["performance_improvements"]["count_time_ms"] = count_time * 1000
            optimization_result["performance_improvements"]["document_count"] = doc_count
            
            # Test query performance if documents exist
            if doc_count > 0:
                try:
                    start_time = time.time()
                    sample = collection.peek(limit=1)
                    peek_time = time.time() - start_time
                    
                    optimization_result["performance_improvements"]["peek_time_ms"] = peek_time * 1000
                    optimization_result["optimizations_applied"].append("performance_tested")
                    
                except Exception as peek_error:
                    optimization_result["warnings"] = [f"Peek test failed: {peek_error}"]
            
            logger.info(f"Collection optimization completed for user {user_id}")
            
        except Exception as e:
            optimization_result["error"] = str(e)
            logger.error(f"Collection optimization failed for user {user_id}: {e}")
        
        return optimization_result
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the ChromaDB client manager.
        
        Returns:
            Dictionary containing health status information
        """
        self._health_stats["last_health_check"] = datetime.now().isoformat()
        
        # Calculate success rates
        total_connections = self._health_stats["total_connections"]
        failed_connections = self._health_stats["failed_connections"]
        
        if total_connections > 0:
            success_rate = (total_connections - failed_connections) / total_connections
        else:
            success_rate = 1.0
        
        health_status = {
            "overall_status": "healthy" if success_rate >= 0.9 else "degraded" if success_rate >= 0.7 else "unhealthy",
            "connection_success_rate": success_rate,
            "active_connections": len(self._connection_pool),
            "cached_collections": len(self._collection_cache),
            "statistics": self._health_stats.copy()
        }
        
        # Add connection pool health
        healthy_connections = 0
        for conn_key, client in self._connection_pool.items():
            try:
                client.list_collections()
                healthy_connections += 1
            except Exception:
                pass
        
        health_status["healthy_connections"] = healthy_connections
        health_status["connection_pool_health"] = healthy_connections / len(self._connection_pool) if self._connection_pool else 1.0
        
        # Clean up old error records (keep only last 100)
        if len(self._health_stats["connection_errors"]) > 100:
            self._health_stats["connection_errors"] = self._health_stats["connection_errors"][-100:]
        
        return health_status
    
    def cleanup_cache(self, max_age_minutes: int = 60) -> int:
        """
        Clean up expired entries from the collection cache.
        
        Args:
            max_age_minutes: Maximum age in minutes for cache entries (default: 60)
            
        Returns:
            Number of entries removed from cache
        """
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        expired_keys = []
        
        for collection_name, (collection, cache_time) in self._collection_cache.items():
            if cache_time < cutoff_time:
                expired_keys.append(collection_name)
        
        for key in expired_keys:
            del self._collection_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def batch_operation_context(self, user_id: str):
        """
        Context manager for batch operations on a user's collection.
        
        Args:
            user_id: User identifier
            
        Returns:
            Context manager that provides optimized collection access
        """
        return BatchOperationContext(self, user_id)
    
    def close_client(self):
        """Close all ChromaDB client connections and clear caches."""
        # Close all connections in the pool
        for conn_key in list(self._connection_pool.keys()):
            try:
                # ChromaDB client doesn't have an explicit close method
                # Just remove the reference
                del self._connection_pool[conn_key]
            except Exception as e:
                logger.warning(f"Error closing connection {conn_key}: {e}")
        
        # Clear the main client reference
        if self._client is not None:
            self._client = None
        
        # Clear all caches
        self._collection_cache.clear()
        
        logger.info("All ChromaDB client connections closed and caches cleared")


class BatchOperationContext:
    """Context manager for optimized batch operations on ChromaDB collections."""
    
    def __init__(self, manager: ChromaClientManager, user_id: str):
        self.manager = manager
        self.user_id = user_id
        self.collection = None
        self.start_time = None
    
    def __enter__(self):
        """Enter the batch operation context."""
        self.start_time = time.time()
        # Get collection without caching to ensure fresh connection
        self.collection = self.manager.get_or_create_collection(self.user_id, use_cache=False)
        logger.debug(f"Started batch operation context for user: {self.user_id}")
        return self.collection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the batch operation context."""
        if self.start_time:
            duration = time.time() - self.start_time
            logger.debug(f"Batch operation completed for user {self.user_id} in {duration:.3f}s")
        
        # Force cache refresh after batch operations
        user_id_hash = hashlib.md5(self.user_id.encode()).hexdigest()
        collection_name = f"{settings.CHROMA_COLLECTION_PREFIX}{user_id_hash}"
        
        if collection_name in self.manager._collection_cache:
            # Update cache with fresh timestamp
            if self.collection is not None:
                self.manager._collection_cache[collection_name] = (self.collection, datetime.now())


# Enhanced async context manager for async operations
@asynccontextmanager
async def async_batch_operation_context(manager: ChromaClientManager, user_id: str):
    """Async context manager for batch operations."""
    start_time = time.time()
    collection = None
    
    try:
        # Get collection in thread pool to avoid blocking
        collection = await asyncio.to_thread(manager.get_or_create_collection, user_id, False)
        logger.debug(f"Started async batch operation context for user: {user_id}")
        yield collection
        
    finally:
        if start_time:
            duration = time.time() - start_time
            logger.debug(f"Async batch operation completed for user {user_id} in {duration:.3f}s")
        
        # Update cache if collection was retrieved
        if collection:
            user_id_hash = hashlib.md5(user_id.encode()).hexdigest()
            collection_name = f"{settings.CHROMA_COLLECTION_PREFIX}{user_id_hash}"
            manager._collection_cache[collection_name] = (collection, datetime.now())

# Global instance
chroma_manager = ChromaClientManager() 