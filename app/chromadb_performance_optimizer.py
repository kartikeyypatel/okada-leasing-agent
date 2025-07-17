# /app/chromadb_performance_optimizer.py
"""
ChromaDB Performance Optimization System

This module provides comprehensive performance optimization for ChromaDB collections,
embedding validation, and vector search performance enhancement for the RAG system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import numpy as np

from app.chroma_client import chroma_manager
from app.models import (
    ValidationResult, StructureOptimizationResult, BenchmarkResult, CleanupResult,
    PerformanceMetrics, VectorStoreStatus
)

logger = logging.getLogger(__name__)

class ChromaDBPerformanceOptimizer:
    """
    ChromaDB performance optimization and health management.
    
    Provides collection health validation, performance benchmarking,
    embedding integrity checks, and automated cleanup operations.
    """
    
    def __init__(self):
        self.benchmark_history: Dict[str, List[BenchmarkResult]] = {}
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
    async def validate_collection_health(self, user_id: str) -> ValidationResult:
        """
        Validate ChromaDB collection health and integrity.
        
        Args:
            user_id: User identifier for collection validation
            
        Returns:
            ValidationResult with health status and recommendations
        """
        start_time = time.time()
        issues_found = []
        recommendations = []
        
        try:
            # Check cache first
            cache_key = f"health_{user_id}"
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                cache_age = (datetime.now() - cached_result.timestamp).total_seconds()
                if cache_age < self.cache_ttl_seconds:
                    return cached_result
            
            # Get collection
            collection = await asyncio.to_thread(
                chroma_manager.get_or_create_collection, user_id
            )
            
            if not collection:
                issues_found.append("Collection not found or could not be created")
                recommendations.append("Initialize user collection with document upload")
                
                result = ValidationResult(
                    is_valid=False,
                    issues_found=issues_found,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    recommendations=recommendations,
                    timestamp=datetime.now()
                )
                self.validation_cache[cache_key] = result
                return result
            
            # Check collection document count
            doc_count = await asyncio.to_thread(collection.count)
            if doc_count == 0:
                issues_found.append("Collection is empty")
                recommendations.append("Upload documents to populate the collection")
            elif doc_count < 10:
                recommendations.append("Low document count - consider adding more documents for better search results")
            
            # Validate collection structure
            try:
                # Get a sample of documents to validate structure
                sample_results = await asyncio.to_thread(collection.peek, limit=5)
                
                if not sample_results or not sample_results.get("documents"):
                    issues_found.append("No document content found in collection")
                    recommendations.append("Rebuild collection with proper document content")
                else:
                    # Check for empty documents
                    empty_docs = sum(1 for doc in sample_results.get("documents", []) 
                                   if not doc or not doc.strip())
                    if empty_docs > 0:
                        issues_found.append(f"Found {empty_docs} empty documents in sample")
                        recommendations.append("Clean up empty documents from collection")
                
                # Check metadata consistency
                metadatas = sample_results.get("metadatas", [])
                if metadatas:
                    # Check for required metadata fields
                    required_fields = ["source", "file_name"]
                    for i, metadata in enumerate(metadatas):
                        if metadata:
                            missing_fields = [field for field in required_fields 
                                            if field not in metadata]
                            if missing_fields:
                                issues_found.append(
                                    f"Document {i} missing metadata fields: {missing_fields}"
                                )
                                recommendations.append("Ensure all documents have required metadata")
                                break
                
            except Exception as e:
                issues_found.append(f"Error validating collection structure: {str(e)}")
                recommendations.append("Check collection integrity and consider rebuilding")
            
            # Check for embedding consistency
            try:
                # Perform a simple query to test retrieval
                test_query = "test query for validation"
                query_results = await asyncio.to_thread(
                    collection.query, 
                    query_texts=[test_query], 
                    n_results=min(3, doc_count)
                )
                
                if not query_results or not query_results.get("documents"):
                    issues_found.append("Query test failed - embeddings may be corrupted")
                    recommendations.append("Rebuild embeddings for the collection")
                
            except Exception as e:
                issues_found.append(f"Embedding validation failed: {str(e)}")
                recommendations.append("Rebuild collection embeddings")
            
            # Generate overall assessment
            is_valid = len(issues_found) == 0
            if not recommendations and is_valid:
                recommendations.append("Collection is healthy and optimized")
            
            validation_time_ms = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                is_valid=is_valid,
                issues_found=issues_found,
                validation_time_ms=validation_time_ms,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Cache the result
            self.validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error validating collection health for {user_id}: {e}")
            
            result = ValidationResult(
                is_valid=False,
                issues_found=[f"Validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000,
                recommendations=["Check ChromaDB connection and collection access"],
                timestamp=datetime.now()
            )
            return result
    
    async def benchmark_retrieval_performance(self, user_id: str) -> BenchmarkResult:
        """
        Benchmark vector search performance for a user's collection.
        
        Args:
            user_id: User identifier for performance benchmarking
            
        Returns:
            BenchmarkResult with performance metrics
        """
        benchmark_start = time.time()
        
        try:
            collection = await asyncio.to_thread(
                chroma_manager.get_or_create_collection, user_id
            )
            
            if not collection:
                return BenchmarkResult(
                    operation_type="vector_search",
                    avg_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    throughput_ops_per_sec=0.0,
                    success_rate=0.0,
                    benchmark_duration_sec=0.0,
                    timestamp=datetime.now()
                )
            
            # Check if collection has documents
            doc_count = await asyncio.to_thread(collection.count)
            if doc_count == 0:
                return BenchmarkResult(
                    operation_type="vector_search",
                    avg_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    throughput_ops_per_sec=0.0,
                    success_rate=0.0,
                    benchmark_duration_sec=0.0,
                    timestamp=datetime.now()
                )
            
            # Perform benchmark queries
            test_queries = [
                "apartment rental information",
                "property details and pricing",
                "location and amenities",
                "square footage and bedrooms",
                "lease terms and availability"
            ]
            
            latencies = []
            successful_queries = 0
            
            for query in test_queries:
                try:
                    query_start = time.time()
                    
                    # Perform the query
                    results = await asyncio.to_thread(
                        collection.query,
                        query_texts=[query],
                        n_results=min(5, doc_count)
                    )
                    
                    query_latency = (time.time() - query_start) * 1000
                    latencies.append(query_latency)
                    
                    if results and results.get("documents"):
                        successful_queries += 1
                    
                except Exception as e:
                    logger.warning(f"Benchmark query failed: {e}")
                    continue
            
            # Calculate performance metrics
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = np.percentile(latencies, 95) if len(latencies) > 1 else avg_latency
                p99_latency = np.percentile(latencies, 99) if len(latencies) > 1 else avg_latency
            else:
                avg_latency = p95_latency = p99_latency = 0.0
            
            total_duration = time.time() - benchmark_start
            throughput = len(latencies) / total_duration if total_duration > 0 else 0.0
            success_rate = successful_queries / len(test_queries) if test_queries else 0.0
            
            benchmark_result = BenchmarkResult(
                operation_type="vector_search",
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                throughput_ops_per_sec=throughput,
                success_rate=success_rate,
                benchmark_duration_sec=total_duration,
                timestamp=datetime.now()
            )
            
            # Store benchmark history
            if user_id not in self.benchmark_history:
                self.benchmark_history[user_id] = []
            self.benchmark_history[user_id].append(benchmark_result)
            
            # Keep only recent benchmarks (last 10)
            if len(self.benchmark_history[user_id]) > 10:
                self.benchmark_history[user_id] = self.benchmark_history[user_id][-10:]
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Error benchmarking retrieval performance for {user_id}: {e}")
            return BenchmarkResult(
                operation_type="vector_search",
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                throughput_ops_per_sec=0.0,
                success_rate=0.0,
                benchmark_duration_sec=time.time() - benchmark_start,
                timestamp=datetime.now()
            )
    
    async def validate_embeddings_integrity(self, user_id: str) -> ValidationResult:
        """
        Validate embedding integrity and consistency for a user's collection.
        
        Args:
            user_id: User identifier for embedding validation
            
        Returns:
            ValidationResult with integrity assessment
        """
        start_time = time.time()
        issues_found = []
        recommendations = []
        
        try:
            collection = await asyncio.to_thread(
                chroma_manager.get_or_create_collection, user_id
            )
            
            if not collection:
                issues_found.append("Collection not accessible for integrity validation")
                return ValidationResult(
                    is_valid=False,
                    issues_found=issues_found,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    recommendations=["Initialize collection before validation"],
                    timestamp=datetime.now()
                )
            
            # Check document count
            doc_count = await asyncio.to_thread(collection.count)
            if doc_count == 0:
                issues_found.append("No documents found for integrity validation")
                recommendations.append("Upload documents to validate embeddings")
                
                return ValidationResult(
                    is_valid=False,
                    issues_found=issues_found,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    recommendations=recommendations,
                    timestamp=datetime.now()
                )
            
            # Test embedding consistency with sample queries
            test_samples = min(10, doc_count)
            try:
                # Get sample documents
                sample_results = await asyncio.to_thread(
                    collection.peek, limit=test_samples
                )
                
                if sample_results and sample_results.get("documents"):
                    documents = sample_results.get("documents", [])
                    ids = sample_results.get("ids", [])
                    
                    # Check for duplicate content
                    doc_set = set()
                    duplicates = 0
                    for doc in documents:
                        if doc in doc_set:
                            duplicates += 1
                        else:
                            doc_set.add(doc)
                    
                    if duplicates > 0:
                        issues_found.append(f"Found {duplicates} duplicate documents")
                        recommendations.append("Remove duplicate documents to improve search quality")
                    
                    # Test query consistency
                    if len(documents) > 0:
                        # Use first document as test query
                        test_doc = documents[0][:100]  # First 100 chars
                        
                        query_results = await asyncio.to_thread(
                            collection.query,
                            query_texts=[test_doc],
                            n_results=min(3, doc_count)
                        )
                        
                        if not query_results or not query_results.get("documents"):
                            issues_found.append("Embeddings may be corrupted - query test failed")
                            recommendations.append("Rebuild embeddings for better search performance")
                        else:
                            # Check if original document is in top results (it should be)
                            result_docs = query_results.get("documents", [[]])[0]
                            if test_doc not in str(result_docs):
                                issues_found.append("Embedding similarity test failed")
                                recommendations.append("Consider regenerating embeddings")
                
            except Exception as e:
                issues_found.append(f"Embedding test failed: {str(e)}")
                recommendations.append("Check embedding model and regenerate if necessary")
            
            # Generate overall integrity assessment
            is_valid = len(issues_found) == 0
            if is_valid and not recommendations:
                recommendations.append("Embeddings integrity is good")
            
            return ValidationResult(
                is_valid=is_valid,
                issues_found=issues_found,
                validation_time_ms=(time.time() - start_time) * 1000,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error validating embeddings integrity for {user_id}: {e}")
            return ValidationResult(
                is_valid=False,
                issues_found=[f"Integrity validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000,
                recommendations=["Check ChromaDB connection and try again"],
                timestamp=datetime.now()
            )
    
    async def optimize_collection_structure(self, user_id: str) -> StructureOptimizationResult:
        """
        Optimize collection structure and performance.
        
        Args:
            user_id: User identifier for collection optimization
            
        Returns:
            StructureOptimizationResult with optimization details
        """
        try:
            collection = await asyncio.to_thread(
                chroma_manager.get_or_create_collection, user_id
            )
            
            if not collection:
                return StructureOptimizationResult(
                    collection_name=f"user_{user_id}",
                    optimizations_applied=[],
                    before_performance={},
                    after_performance={},
                    recommendations=["Collection not found - initialize first"],
                    timestamp=datetime.now()
                )
            
            # Get before performance metrics
            before_benchmark = await self.benchmark_retrieval_performance(user_id)
            before_performance = {
                "avg_latency_ms": before_benchmark.avg_latency_ms,
                "success_rate": before_benchmark.success_rate,
                "throughput": before_benchmark.throughput_ops_per_sec
            }
            
            optimizations_applied = []
            recommendations = []
            
            # Check document count for optimization decisions
            doc_count = await asyncio.to_thread(collection.count)
            
            if doc_count > 1000:
                recommendations.append("Consider implementing document chunking for large collections")
                optimizations_applied.append("Recommended document chunking strategy")
            
            # Check for metadata optimization opportunities
            try:
                sample_results = await asyncio.to_thread(collection.peek, limit=10)
                if sample_results and sample_results.get("metadatas"):
                    metadatas = sample_results.get("metadatas", [])
                    
                    # Analyze metadata structure
                    all_keys = set()
                    for metadata in metadatas:
                        if metadata:
                            all_keys.update(metadata.keys())
                    
                    if len(all_keys) > 10:
                        recommendations.append("Consider simplifying metadata structure")
                        optimizations_applied.append("Metadata structure analysis completed")
                    
                    # Check for consistent metadata
                    if metadatas:
                        first_keys = set(metadatas[0].keys()) if metadatas[0] else set()
                        inconsistent = any(
                            set(m.keys()) != first_keys for m in metadatas[1:] if m
                        )
                        if inconsistent:
                            recommendations.append("Standardize metadata fields across documents")
                            optimizations_applied.append("Identified metadata inconsistencies")
            
            except Exception as e:
                logger.warning(f"Error analyzing metadata structure: {e}")
            
            # Get after performance (same as before for now, as we haven't made changes)
            after_performance = before_performance.copy()
            
            if not optimizations_applied:
                optimizations_applied.append("Collection structure analysis completed")
            
            if not recommendations:
                recommendations.append("Collection structure is well-optimized")
            
            return StructureOptimizationResult(
                collection_name=f"user_{user_id}",
                optimizations_applied=optimizations_applied,
                before_performance=before_performance,
                after_performance=after_performance,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing collection structure for {user_id}: {e}")
            return StructureOptimizationResult(
                collection_name=f"user_{user_id}",
                optimizations_applied=[],
                before_performance={},
                after_performance={},
                recommendations=[f"Optimization error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def cleanup_stale_embeddings(self, user_id: str) -> CleanupResult:
        """
        Clean up stale or corrupted embeddings from a user's collection.
        
        Args:
            user_id: User identifier for cleanup operation
            
        Returns:
            CleanupResult with cleanup statistics
        """
        cleanup_start = time.time()
        
        try:
            collection = await asyncio.to_thread(
                chroma_manager.get_or_create_collection, user_id
            )
            
            if not collection:
                return CleanupResult(
                    collections_cleaned=0,
                    documents_removed=0,
                    storage_freed_mb=0.0,
                    cleanup_duration_sec=time.time() - cleanup_start,
                    errors_encountered=["Collection not found"],
                    timestamp=datetime.now()
                )
            
            # Get initial document count
            initial_count = await asyncio.to_thread(collection.count)
            
            if initial_count == 0:
                return CleanupResult(
                    collections_cleaned=1,
                    documents_removed=0,
                    storage_freed_mb=0.0,
                    cleanup_duration_sec=time.time() - cleanup_start,
                    errors_encountered=[],
                    timestamp=datetime.now()
                )
            
            errors_encountered = []
            documents_removed = 0
            
            try:
                # Get all documents for analysis
                all_results = await asyncio.to_thread(
                    collection.get,
                    include=["documents", "metadatas", "ids"]
                )
                
                if all_results:
                    documents = all_results.get("documents", [])
                    metadatas = all_results.get("metadatas", [])
                    ids = all_results.get("ids", [])
                    
                    # Identify documents to remove
                    ids_to_remove = []
                    
                    for i, (doc, metadata, doc_id) in enumerate(zip(documents, metadatas, ids)):
                        # Remove empty documents
                        if not doc or not doc.strip():
                            ids_to_remove.append(doc_id)
                            continue
                        
                        # Remove documents with corrupted metadata
                        if metadata is None or (isinstance(metadata, dict) and not metadata):
                            # Only remove if it's clearly corrupted (None), not just empty dict
                            if metadata is None:
                                ids_to_remove.append(doc_id)
                                continue
                    
                    # Remove identified documents
                    if ids_to_remove:
                        try:
                            await asyncio.to_thread(collection.delete, ids=ids_to_remove)
                            documents_removed = len(ids_to_remove)
                            logger.info(f"Removed {documents_removed} stale documents from {user_id}")
                        except Exception as e:
                            errors_encountered.append(f"Error removing documents: {str(e)}")
                
            except Exception as e:
                errors_encountered.append(f"Error during cleanup analysis: {str(e)}")
            
            # Estimate storage freed (rough estimate)
            storage_freed_mb = documents_removed * 0.001  # Rough estimate: 1KB per document
            
            cleanup_duration = time.time() - cleanup_start
            
            return CleanupResult(
                collections_cleaned=1,
                documents_removed=documents_removed,
                storage_freed_mb=storage_freed_mb,
                cleanup_duration_sec=cleanup_duration,
                errors_encountered=errors_encountered,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up stale embeddings for {user_id}: {e}")
            return CleanupResult(
                collections_cleaned=0,
                documents_removed=0,
                storage_freed_mb=0.0,
                cleanup_duration_sec=time.time() - cleanup_start,
                errors_encountered=[f"Cleanup error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def get_collection_status(self, user_id: str) -> VectorStoreStatus:
        """
        Get comprehensive status of a user's ChromaDB collection.
        
        Args:
            user_id: User identifier
            
        Returns:
            VectorStoreStatus with collection health and metrics
        """
        try:
            collection = await asyncio.to_thread(
                chroma_manager.get_or_create_collection, user_id
            )
            
            if not collection:
                return VectorStoreStatus(
                    collections_healthy=False,
                    total_collections=0,
                    storage_size_mb=0.0,
                    last_check=datetime.now(),
                    error_message="Collection not accessible"
                )
            
            # Get collection metrics
            doc_count = await asyncio.to_thread(collection.count)
            
            # Estimate storage size (rough calculation)
            storage_size_mb = doc_count * 0.001  # Rough estimate
            
            # Check if collection is healthy
            validation_result = await self.validate_collection_health(user_id)
            
            return VectorStoreStatus(
                collections_healthy=validation_result.is_valid,
                total_collections=1 if doc_count > 0 else 0,
                storage_size_mb=storage_size_mb,
                last_check=datetime.now(),
                error_message="; ".join(validation_result.issues_found) if validation_result.issues_found else None
            )
            
        except Exception as e:
            logger.error(f"Error getting collection status for {user_id}: {e}")
            return VectorStoreStatus(
                collections_healthy=False,
                total_collections=0,
                storage_size_mb=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )

# Global instance
chromadb_performance_optimizer = ChromaDBPerformanceOptimizer() 