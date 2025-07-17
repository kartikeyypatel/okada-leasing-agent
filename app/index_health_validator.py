# /app/index_health_validator.py
"""
Index Health Validation System

This service validates index health before expensive operations and provides
caching mechanisms to avoid repeated validation checks.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import app.rag as rag_module

logger = logging.getLogger(__name__)


@dataclass
class IndexHealthResult:
    """Result of index health validation."""
    is_healthy: bool
    user_id: str
    validation_time: float
    issues_found: List[str]
    recommendations: List[str]
    index_exists: bool
    retriever_functional: bool
    document_count: Optional[int] = None
    last_updated: Optional[datetime] = None


@dataclass
class ValidationCache:
    """Cached validation results."""
    result: IndexHealthResult
    cached_at: datetime
    ttl_minutes: int = 5  # Cache for 5 minutes


class IndexHealthValidator:
    """
    Service for validating index health and providing caching.
    
    Prevents expensive operations by checking index availability and health
    before triggering expensive RAG workflows.
    """
    
    def __init__(self):
        # Cache validation results to avoid repeated checks
        self.validation_cache: Dict[str, ValidationCache] = {}
        
        # Performance tracking
        self.validation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Health thresholds
        self.MIN_DOCUMENTS_THRESHOLD = 1
        self.MAX_VALIDATION_TIME_MS = 2000  # 2 seconds
        self.CACHE_TTL_MINUTES = 5
    
    async def validate_user_index_health(self, user_id: str, force_refresh: bool = False) -> IndexHealthResult:
        """
        Validate the health of a user's index with caching.
        
        Args:
            user_id: User identifier
            force_refresh: Force validation even if cached result exists
            
        Returns:
            IndexHealthResult with validation details
        """
        start_time = time.time()
        
        try:
            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_result = self._get_cached_result(user_id)
                if cached_result:
                    self.cache_hits += 1
                    logger.debug(f"Using cached health result for user {user_id}")
                    return cached_result.result
            
            self.cache_misses += 1
            
            # Perform comprehensive health validation
            logger.info(f"Validating index health for user: {user_id}")
            
            validation_result = await self._perform_health_validation(user_id)
            
            # Cache the result
            self._cache_result(user_id, validation_result)
            
            # Track performance
            validation_time = (time.time() - start_time) * 1000
            self.validation_times.append(validation_time)
            validation_result.validation_time = validation_time
            
            logger.info(f"Index health validation completed for {user_id} in {validation_time:.2f}ms: {'healthy' if validation_result.is_healthy else 'unhealthy'}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating index health for {user_id}: {e}")
            return IndexHealthResult(
                is_healthy=False,
                user_id=user_id,
                validation_time=(time.time() - start_time) * 1000,
                issues_found=[f"Validation error: {e}"],
                recommendations=["Contact support for index validation issues"],
                index_exists=False,
                retriever_functional=False
            )
    
    async def _perform_health_validation(self, user_id: str) -> IndexHealthResult:
        """Perform comprehensive index health validation."""
        
        issues_found = []
        recommendations = []
        index_exists = False
        retriever_functional = False
        document_count = None
        last_updated = None
        
        # Step 1: Check if user index exists
        try:
            user_index = await rag_module.get_user_index(user_id)
            if user_index:
                index_exists = True
                logger.debug(f"Index exists for user {user_id}")
                
                # Step 2: Validate index document count
                try:
                    if hasattr(user_index, 'docstore') and hasattr(user_index.docstore, 'docs'):
                        document_count = len(user_index.docstore.docs)
                        logger.debug(f"Index contains {document_count} documents")
                        
                        if document_count < self.MIN_DOCUMENTS_THRESHOLD:
                            issues_found.append(f"Index has only {document_count} documents (minimum: {self.MIN_DOCUMENTS_THRESHOLD})")
                            recommendations.append("Upload more documents or rebuild index from user documents")
                    else:
                        logger.warning(f"Cannot determine document count for user {user_id}")
                        issues_found.append("Unable to determine document count")
                        
                except Exception as e:
                    issues_found.append(f"Error checking document count: {e}")
                
                # Step 3: Test retriever functionality
                try:
                    retriever = user_index.as_retriever(similarity_top_k=1)
                    if retriever:
                        # Test with a simple query
                        test_query = "test query"
                        test_nodes = await asyncio.wait_for(
                            asyncio.to_thread(retriever.retrieve, test_query),
                            timeout=5.0
                        )
                        retriever_functional = True
                        logger.debug(f"Retriever functional for user {user_id} (returned {len(test_nodes)} nodes)")
                    else:
                        issues_found.append("Index exists but retriever creation failed")
                        recommendations.append("Rebuild index - retriever creation is failing")
                        
                except asyncio.TimeoutError:
                    issues_found.append("Retriever test timed out after 5 seconds")
                    recommendations.append("Index may be corrupted - consider rebuilding")
                except Exception as e:
                    issues_found.append(f"Retriever test failed: {e}")
                    recommendations.append("Index may be corrupted - rebuild recommended")
                
                # Step 4: Check fusion retriever
                try:
                    fusion_retriever = rag_module.get_fusion_retriever(user_id)
                    if not fusion_retriever:
                        issues_found.append("Fusion retriever creation failed")
                        recommendations.append("BM25 retriever may be missing - rebuild index")
                except Exception as e:
                    issues_found.append(f"Fusion retriever check failed: {e}")
                    
            else:
                issues_found.append("User index does not exist")
                recommendations.append("Build index from user documents")
                
        except Exception as e:
            issues_found.append(f"Error accessing user index: {e}")
            recommendations.append("Index access failed - may need to clear and rebuild")
        
        # Step 5: Check if user has documents available for indexing
        import os
        user_doc_dir = os.path.join("user_documents", user_id)
        if os.path.exists(user_doc_dir):
            csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
            if csv_files and not index_exists:
                recommendations.append(f"Found {len(csv_files)} CSV files - can rebuild index")
            elif csv_files and document_count == 0:
                recommendations.append("Documents available but index is empty - rebuild needed")
        else:
            if not index_exists:
                issues_found.append("No user documents found and no existing index")
                recommendations.append("Upload documents before using property search features")
        
        # Determine overall health
        is_healthy = (
            index_exists and 
            retriever_functional and 
            (document_count is None or document_count >= self.MIN_DOCUMENTS_THRESHOLD) and
            len(issues_found) == 0
        )
        
        return IndexHealthResult(
            is_healthy=is_healthy,
            user_id=user_id,
            validation_time=0.0,  # Will be set by caller
            issues_found=issues_found,
            recommendations=recommendations,
            index_exists=index_exists,
            retriever_functional=retriever_functional,
            document_count=document_count,
            last_updated=datetime.now()
        )
    
    def _get_cached_result(self, user_id: str) -> Optional[ValidationCache]:
        """Get cached validation result if still valid."""
        
        cached = self.validation_cache.get(user_id)
        if not cached:
            return None
        
        # Check if cache is still valid
        age = datetime.now() - cached.cached_at
        if age.total_seconds() / 60 > cached.ttl_minutes:
            # Cache expired
            del self.validation_cache[user_id]
            return None
        
        return cached
    
    def _cache_result(self, user_id: str, result: IndexHealthResult):
        """Cache validation result."""
        
        self.validation_cache[user_id] = ValidationCache(
            result=result,
            cached_at=datetime.now(),
            ttl_minutes=self.CACHE_TTL_MINUTES
        )
        
        # Clean up old cache entries (keep last 100 users)
        if len(self.validation_cache) > 100:
            oldest_user = min(self.validation_cache.keys(), 
                            key=lambda k: self.validation_cache[k].cached_at)
            del self.validation_cache[oldest_user]
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear validation cache for specific user or all users."""
        
        if user_id:
            self.validation_cache.pop(user_id, None)
            logger.info(f"Cleared validation cache for user {user_id}")
        else:
            self.validation_cache.clear()
            logger.info("Cleared all validation cache")
    
    async def batch_validate_users(self, user_ids: List[str]) -> Dict[str, IndexHealthResult]:
        """Validate multiple users' indices in parallel."""
        
        logger.info(f"Batch validating {len(user_ids)} user indices")
        
        # Run validations in parallel
        tasks = [self.validate_user_index_health(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        validation_results = {}
        for user_id, result in zip(user_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Batch validation failed for {user_id}: {result}")
                validation_results[user_id] = IndexHealthResult(
                    is_healthy=False,
                    user_id=user_id,
                    validation_time=0.0,
                    issues_found=[f"Validation error: {result}"],
                    recommendations=["Contact support"],
                    index_exists=False,
                    retriever_functional=False
                )
            else:
                validation_results[user_id] = result
        
        healthy_count = sum(1 for r in validation_results.values() if r.is_healthy)
        logger.info(f"Batch validation complete: {healthy_count}/{len(user_ids)} indices healthy")
        
        return validation_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for validation."""
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        avg_validation_time = (
            sum(self.validation_times) / len(self.validation_times) 
            if self.validation_times else 0
        )
        
        return {
            "total_validations": len(self.validation_times),
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cached_users": len(self.validation_cache),
            "avg_validation_time_ms": f"{avg_validation_time:.2f}",
            "target_met": sum(1 for t in self.validation_times if t < self.MAX_VALIDATION_TIME_MS) / len(self.validation_times) * 100 if self.validation_times else 0
        }
    
    async def auto_fix_common_issues(self, user_id: str) -> Dict[str, Any]:
        """Attempt to automatically fix common index issues."""
        
        fix_report = {
            "user_id": user_id,
            "fixes_attempted": [],
            "fixes_successful": [],
            "fixes_failed": [],
            "final_health": None
        }
        
        # Get current health status
        health_result = await self.validate_user_index_health(user_id, force_refresh=True)
        
        if health_result.is_healthy:
            fix_report["message"] = "Index is already healthy"
            return fix_report
        
        # Fix 1: Rebuild index if documents exist but index is missing/corrupted
        if not health_result.index_exists or not health_result.retriever_functional:
            fix_report["fixes_attempted"].append("rebuild_index")
            
            try:
                import os
                user_doc_dir = os.path.join("user_documents", user_id)
                if os.path.exists(user_doc_dir):
                    csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
                    if csv_files:
                        file_paths = [os.path.join(user_doc_dir, f) for f in csv_files]
                        
                        # Clear existing index
                        await rag_module.clear_user_index(user_id)
                        
                        # Rebuild
                        new_index = await rag_module.build_user_index(user_id, file_paths)
                        
                        if new_index:
                            fix_report["fixes_successful"].append("rebuild_index")
                            # Clear our cache so next validation is fresh
                            self.clear_cache(user_id)
                        else:
                            fix_report["fixes_failed"].append("rebuild_index: returned None")
                    else:
                        fix_report["fixes_failed"].append("rebuild_index: no documents found")
                else:
                    fix_report["fixes_failed"].append("rebuild_index: no document directory")
                    
            except Exception as e:
                fix_report["fixes_failed"].append(f"rebuild_index: {e}")
        
        # Get final health status
        final_health = await self.validate_user_index_health(user_id, force_refresh=True)
        fix_report["final_health"] = {
            "is_healthy": final_health.is_healthy,
            "issues_remaining": len(final_health.issues_found),
            "issues": final_health.issues_found
        }
        
        return fix_report 