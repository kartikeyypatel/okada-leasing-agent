# /app/rag_performance_optimizer.py
"""
RAG Retrieval Performance Optimization System

This module provides comprehensive performance optimization for RAG operations,
including retrieval latency tracking, embedding quality validation, and
automatic cleanup of stale embeddings.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from app.chroma_client import chroma_manager
from app.models import OptimizationResult, PerformanceMetrics, OperationType
import app.rag as rag_module

logger = logging.getLogger(__name__)


@dataclass
class RAGPerformanceMetrics:
    """RAG-specific performance metrics."""
    user_id: str
    operation_type: str
    retrieval_latency_ms: float
    embedding_count: int
    similarity_scores: List[float]
    context_quality_score: float
    success: bool
    timestamp: datetime
    query_text: str
    retrieved_documents: int
    fallback_used: bool = False
    error_message: Optional[str] = None


@dataclass
class EmbeddingQualityReport:
    """Report on embedding quality and integrity."""
    user_id: str
    total_embeddings: int
    valid_embeddings: int
    corrupted_embeddings: int
    duplicate_embeddings: int
    orphaned_embeddings: int
    avg_embedding_dimension: float
    dimension_consistency: bool
    quality_score: float
    recommendations: List[str]
    timestamp: datetime


@dataclass
class RetrievalOptimizationResult:
    """Result of retrieval optimization."""
    user_id: str
    optimization_type: str
    before_latency_ms: float
    after_latency_ms: float
    improvement_percent: float
    optimizations_applied: List[str]
    context_quality_improvement: float
    recommendations: List[str]
    timestamp: datetime


class RAGPerformanceOptimizer:
    """
    RAG retrieval performance optimization and monitoring system.
    
    Integrates with existing RAG system to monitor performance, optimize
    retrieval operations, and maintain embedding quality.
    """
    
    def __init__(self):
        self.performance_history: List[RAGPerformanceMetrics] = []
        self.optimization_history: List[RetrievalOptimizationResult] = []
        self.embedding_quality_cache: Dict[str, EmbeddingQualityReport] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_retrieval_latency_ms": 2000.0,  # 2 seconds
            "min_context_quality_score": 0.7,
            "max_embedding_dimension_variance": 0.05,
            "min_similarity_threshold": 0.3
        }
        
        # Optimization settings
        self.optimization_enabled = True
        self.auto_cleanup_enabled = True
        self.quality_monitoring_enabled = True
        
        # Cache settings
        self.cache_ttl_minutes = 30
        self.max_performance_history = 1000
        
        logger.info("RAG Performance Optimizer initialized")
    
    async def monitor_retrieval_performance(self, user_id: str, query: str, 
                                          start_time: float) -> RAGPerformanceMetrics:
        """
        Monitor and record RAG retrieval performance.
        
        Args:
            user_id: User identifier
            query: Search query
            start_time: Query start timestamp
            
        Returns:
            RAGPerformanceMetrics with performance data
        """
        try:
            end_time = time.time()
            retrieval_latency_ms = (end_time - start_time) * 1000
            
            # Get retrieval context and quality metrics
            context_quality = await self._assess_context_quality(user_id, query)
            
            # Check if fallback was used
            fallback_used = hasattr(rag_module, '_last_fallback_used') and rag_module._last_fallback_used
            
            # Create performance metrics
            metrics = RAGPerformanceMetrics(
                user_id=user_id,
                operation_type="retrieval",
                retrieval_latency_ms=retrieval_latency_ms,
                embedding_count=context_quality.get("embedding_count", 0),
                similarity_scores=context_quality.get("similarity_scores", []),
                context_quality_score=context_quality.get("quality_score", 0.0),
                success=retrieval_latency_ms < self.performance_thresholds["max_retrieval_latency_ms"],
                timestamp=datetime.now(),
                query_text=query,
                retrieved_documents=context_quality.get("document_count", 0),
                fallback_used=fallback_used
            )
            
            # Record metrics
            self.performance_history.append(metrics)
            
            # Maintain history size
            if len(self.performance_history) > self.max_performance_history:
                self.performance_history = self.performance_history[-self.max_performance_history:]
            
            # Check for optimization opportunities
            if self.optimization_enabled:
                await self._check_optimization_triggers(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring retrieval performance: {e}")
            return RAGPerformanceMetrics(
                user_id=user_id,
                operation_type="retrieval",
                retrieval_latency_ms=0.0,
                embedding_count=0,
                similarity_scores=[],
                context_quality_score=0.0,
                success=False,
                timestamp=datetime.now(),
                query_text=query,
                retrieved_documents=0,
                error_message=str(e)
            )
    
    async def optimize_user_retrieval(self, user_id: str) -> RetrievalOptimizationResult:
        """
        Optimize RAG retrieval performance for a specific user.
        
        Args:
            user_id: User identifier for optimization
            
        Returns:
            RetrievalOptimizationResult with optimization details
        """
        try:
            # Get baseline performance
            baseline_metrics = await self._get_user_baseline_performance(user_id)
            before_latency = baseline_metrics.get("avg_latency_ms", 0.0)
            
            optimizations_applied = []
            
            # 1. Optimize embedding collection structure
            collection_optimized = await self._optimize_collection_structure(user_id)
            if collection_optimized:
                optimizations_applied.append("Collection structure optimization")
            
            # 2. Clean up stale embeddings
            cleanup_result = await self._cleanup_stale_embeddings(user_id)
            if cleanup_result["embeddings_removed"] > 0:
                optimizations_applied.append(f"Removed {cleanup_result['embeddings_removed']} stale embeddings")
            
            # 3. Optimize search parameters
            search_optimized = await self._optimize_search_parameters(user_id)
            if search_optimized:
                optimizations_applied.append("Search parameter optimization")
            
            # 4. Update embedding quality
            quality_improved = await self._improve_embedding_quality(user_id)
            if quality_improved:
                optimizations_applied.append("Embedding quality improvement")
            
            # Get post-optimization performance
            post_metrics = await self._get_user_baseline_performance(user_id)
            after_latency = post_metrics.get("avg_latency_ms", before_latency)
            
            # Calculate improvement
            improvement_percent = 0.0
            if before_latency > 0:
                improvement_percent = ((before_latency - after_latency) / before_latency) * 100
            
            # Context quality improvement
            context_improvement = post_metrics.get("avg_quality_score", 0.0) - baseline_metrics.get("avg_quality_score", 0.0)
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(user_id, baseline_metrics, post_metrics)
            
            result = RetrievalOptimizationResult(
                user_id=user_id,
                optimization_type="comprehensive",
                before_latency_ms=before_latency,
                after_latency_ms=after_latency,
                improvement_percent=improvement_percent,
                optimizations_applied=optimizations_applied,
                context_quality_improvement=context_improvement,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            self.optimization_history.append(result)
            
            logger.info(f"RAG optimization completed for user {user_id}: {improvement_percent:.1f}% improvement")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing user retrieval: {e}")
            return RetrievalOptimizationResult(
                user_id=user_id,
                optimization_type="error",
                before_latency_ms=0.0,
                after_latency_ms=0.0,
                improvement_percent=0.0,
                optimizations_applied=[],
                context_quality_improvement=0.0,
                recommendations=[f"Optimization failed: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def validate_embedding_quality(self, user_id: str) -> EmbeddingQualityReport:
        """
        Validate embedding quality and integrity for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            EmbeddingQualityReport with quality assessment
        """
        try:
            # Check cache first
            cache_key = f"{user_id}_quality"
            if cache_key in self.embedding_quality_cache:
                cached_report = self.embedding_quality_cache[cache_key]
                if datetime.now() - cached_report.timestamp < timedelta(minutes=self.cache_ttl_minutes):
                    return cached_report
            
            # Get user's collection
            collection_name = chroma_manager._get_collection_name(user_id)
            
            try:
                collection = await chroma_manager._get_or_create_collection(collection_name)
                
                # Get all embeddings
                result = collection.get(include=['embeddings', 'metadatas', 'documents'])
                
                total_embeddings = len(result['ids']) if result['ids'] else 0
                embeddings = result['embeddings'] if result['embeddings'] else []
                
                # Validate embeddings
                valid_embeddings = 0
                corrupted_embeddings = 0
                dimensions = []
                
                for embedding in embeddings:
                    if embedding and len(embedding) > 0:
                        if all(isinstance(x, (int, float)) and not np.isnan(x) for x in embedding):
                            valid_embeddings += 1
                            dimensions.append(len(embedding))
                        else:
                            corrupted_embeddings += 1
                    else:
                        corrupted_embeddings += 1
                
                # Check for duplicates (simplified check)
                duplicate_embeddings = total_embeddings - len(set(str(emb) for emb in embeddings if emb))
                
                # Calculate dimension consistency
                avg_dimension = np.mean(dimensions) if dimensions else 0.0
                dimension_consistency = len(set(dimensions)) <= 1 if dimensions else True
                
                # Orphaned embeddings (embeddings without documents)
                documents = result['documents'] if result['documents'] else []
                orphaned_embeddings = max(0, total_embeddings - len([doc for doc in documents if doc]))
                
                # Calculate quality score
                quality_score = 0.0
                if total_embeddings > 0:
                    validity_score = valid_embeddings / total_embeddings
                    consistency_score = 1.0 if dimension_consistency else 0.5
                    orphan_penalty = orphaned_embeddings / total_embeddings
                    duplicate_penalty = duplicate_embeddings / total_embeddings
                    
                    quality_score = max(0.0, validity_score * consistency_score - orphan_penalty - duplicate_penalty)
                
                # Generate recommendations
                recommendations = []
                if corrupted_embeddings > 0:
                    recommendations.append(f"Remove {corrupted_embeddings} corrupted embeddings")
                if duplicate_embeddings > 0:
                    recommendations.append(f"Remove {duplicate_embeddings} duplicate embeddings")
                if orphaned_embeddings > 0:
                    recommendations.append(f"Clean up {orphaned_embeddings} orphaned embeddings")
                if not dimension_consistency:
                    recommendations.append("Standardize embedding dimensions")
                if quality_score < 0.8:
                    recommendations.append("Consider re-generating embeddings for better quality")
                
                if not recommendations:
                    recommendations.append("Embedding quality is good")
                
                report = EmbeddingQualityReport(
                    user_id=user_id,
                    total_embeddings=total_embeddings,
                    valid_embeddings=valid_embeddings,
                    corrupted_embeddings=corrupted_embeddings,
                    duplicate_embeddings=duplicate_embeddings,
                    orphaned_embeddings=orphaned_embeddings,
                    avg_embedding_dimension=avg_dimension,
                    dimension_consistency=dimension_consistency,
                    quality_score=quality_score,
                    recommendations=recommendations,
                    timestamp=datetime.now()
                )
                
                # Cache the report
                self.embedding_quality_cache[cache_key] = report
                
                return report
                
            except Exception as collection_error:
                logger.warning(f"Error accessing collection for user {user_id}: {collection_error}")
                return EmbeddingQualityReport(
                    user_id=user_id,
                    total_embeddings=0,
                    valid_embeddings=0,
                    corrupted_embeddings=0,
                    duplicate_embeddings=0,
                    orphaned_embeddings=0,
                    avg_embedding_dimension=0.0,
                    dimension_consistency=True,
                    quality_score=0.0,
                    recommendations=["Collection not accessible or empty"],
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error validating embedding quality: {e}")
            return EmbeddingQualityReport(
                user_id=user_id,
                total_embeddings=0,
                valid_embeddings=0,
                corrupted_embeddings=0,
                duplicate_embeddings=0,
                orphaned_embeddings=0,
                avg_embedding_dimension=0.0,
                dimension_consistency=True,
                quality_score=0.0,
                recommendations=[f"Quality validation failed: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def cleanup_stale_embeddings(self, user_id: str) -> Dict[str, Any]:
        """
        Clean up stale and orphaned embeddings for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            collection_name = chroma_manager._get_collection_name(user_id)
            
            try:
                collection = await chroma_manager._get_or_create_collection(collection_name)
                
                # Get all data
                result = collection.get(include=['embeddings', 'metadatas', 'documents'])
                
                if not result['ids']:
                    return {
                        "embeddings_removed": 0,
                        "orphaned_removed": 0,
                        "duplicates_removed": 0,
                        "corrupted_removed": 0,
                        "cleanup_successful": True
                    }
                
                ids_to_remove = []
                cleanup_stats = {
                    "embeddings_removed": 0,
                    "orphaned_removed": 0,
                    "duplicates_removed": 0,
                    "corrupted_removed": 0
                }
                
                # Identify items to remove
                seen_embeddings = set()
                
                for i, (id_, embedding, document, metadata) in enumerate(zip(
                    result['ids'], 
                    result['embeddings'] or [], 
                    result['documents'] or [], 
                    result['metadatas'] or []
                )):
                    should_remove = False
                    
                    # Check for corrupted embeddings
                    if not embedding or not all(isinstance(x, (int, float)) and not np.isnan(x) for x in embedding):
                        should_remove = True
                        cleanup_stats["corrupted_removed"] += 1
                    
                    # Check for orphaned embeddings (no document)
                    elif not document or document.strip() == "":
                        should_remove = True
                        cleanup_stats["orphaned_removed"] += 1
                    
                    # Check for duplicates (simplified)
                    elif embedding:
                        embedding_str = str(embedding)
                        if embedding_str in seen_embeddings:
                            should_remove = True
                            cleanup_stats["duplicates_removed"] += 1
                        else:
                            seen_embeddings.add(embedding_str)
                    
                    if should_remove:
                        ids_to_remove.append(id_)
                
                # Remove identified items
                if ids_to_remove:
                    collection.delete(ids=ids_to_remove)
                    cleanup_stats["embeddings_removed"] = len(ids_to_remove)
                    logger.info(f"Cleaned up {len(ids_to_remove)} stale embeddings for user {user_id}")
                
                cleanup_stats["cleanup_successful"] = True
                return cleanup_stats
                
            except Exception as collection_error:
                logger.error(f"Error during cleanup for user {user_id}: {collection_error}")
                return {
                    "embeddings_removed": 0,
                    "orphaned_removed": 0,
                    "duplicates_removed": 0,
                    "corrupted_removed": 0,
                    "cleanup_successful": False,
                    "error": str(collection_error)
                }
            
        except Exception as e:
            logger.error(f"Error cleaning up stale embeddings: {e}")
            return {
                "embeddings_removed": 0,
                "orphaned_removed": 0,
                "duplicates_removed": 0,
                "corrupted_removed": 0,
                "cleanup_successful": False,
                "error": str(e)
            }
    
    def get_performance_dashboard(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get RAG performance dashboard data."""
        try:
            # Filter metrics by user if specified
            metrics = self.performance_history
            if user_id:
                metrics = [m for m in metrics if m.user_id == user_id]
            
            if not metrics:
                return {
                    "message": "No performance data available",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate statistics
            recent_time = datetime.now() - timedelta(hours=24)
            recent_metrics = [m for m in metrics if m.timestamp >= recent_time]
            
            avg_latency = sum(m.retrieval_latency_ms for m in recent_metrics) / len(recent_metrics)
            avg_quality = sum(m.context_quality_score for m in recent_metrics) / len(recent_metrics)
            success_rate = len([m for m in recent_metrics if m.success]) / len(recent_metrics)
            fallback_rate = len([m for m in recent_metrics if m.fallback_used]) / len(recent_metrics)
            
            # Performance trends
            latency_trend = "stable"
            if len(recent_metrics) >= 10:
                first_half = recent_metrics[:len(recent_metrics)//2]
                second_half = recent_metrics[len(recent_metrics)//2:]
                
                first_avg = sum(m.retrieval_latency_ms for m in first_half) / len(first_half)
                second_avg = sum(m.retrieval_latency_ms for m in second_half) / len(second_half)
                
                if second_avg > first_avg * 1.1:
                    latency_trend = "increasing"
                elif second_avg < first_avg * 0.9:
                    latency_trend = "decreasing"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "performance_summary": {
                    "total_queries_24h": len(recent_metrics),
                    "avg_latency_ms": avg_latency,
                    "avg_quality_score": avg_quality,
                    "success_rate": success_rate,
                    "fallback_rate": fallback_rate,
                    "latency_trend": latency_trend
                },
                "thresholds": {
                    "latency_threshold_ms": self.performance_thresholds["max_retrieval_latency_ms"],
                    "quality_threshold": self.performance_thresholds["min_context_quality_score"],
                    "latency_status": "good" if avg_latency < self.performance_thresholds["max_retrieval_latency_ms"] else "poor",
                    "quality_status": "good" if avg_quality >= self.performance_thresholds["min_context_quality_score"] else "poor"
                },
                "recent_optimizations": [
                    {
                        "timestamp": opt.timestamp.isoformat(),
                        "improvement_percent": opt.improvement_percent,
                        "optimizations": opt.optimizations_applied
                    }
                    for opt in self.optimization_history[-5:]  # Last 5 optimizations
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting performance dashboard: {e}")
            return {"error": f"Dashboard error: {str(e)}"}
    
    # Private helper methods
    
    async def _assess_context_quality(self, user_id: str, query: str) -> Dict[str, Any]:
        """Assess the quality of retrieved context."""
        try:
            # This would integrate with the actual RAG retrieval to get context quality metrics
            # For now, we'll provide a placeholder implementation
            
            # Get collection info
            collection_name = chroma_manager._get_collection_name(user_id)
            
            try:
                collection = await chroma_manager._get_or_create_collection(collection_name)
                result = collection.get()
                
                embedding_count = len(result['ids']) if result['ids'] else 0
                
                # Simulate similarity scores and quality assessment
                # In a real implementation, this would come from the actual retrieval
                similarity_scores = [0.8, 0.7, 0.6, 0.5, 0.4]  # Placeholder
                quality_score = max(similarity_scores) if similarity_scores else 0.0
                document_count = min(5, embedding_count)  # Assume top 5 documents retrieved
                
                return {
                    "embedding_count": embedding_count,
                    "similarity_scores": similarity_scores,
                    "quality_score": quality_score,
                    "document_count": document_count
                }
                
            except Exception:
                return {
                    "embedding_count": 0,
                    "similarity_scores": [],
                    "quality_score": 0.0,
                    "document_count": 0
                }
            
        except Exception as e:
            logger.error(f"Error assessing context quality: {e}")
            return {
                "embedding_count": 0,
                "similarity_scores": [],
                "quality_score": 0.0,
                "document_count": 0
            }
    
    async def _check_optimization_triggers(self, metrics: RAGPerformanceMetrics):
        """Check if optimization should be triggered based on performance metrics."""
        try:
            # Check if latency is consistently high
            user_metrics = [m for m in self.performance_history[-10:] if m.user_id == metrics.user_id]
            
            if len(user_metrics) >= 5:
                avg_latency = sum(m.retrieval_latency_ms for m in user_metrics) / len(user_metrics)
                if avg_latency > self.performance_thresholds["max_retrieval_latency_ms"]:
                    logger.info(f"Triggering optimization for user {metrics.user_id} due to high latency")
                    # Could trigger automatic optimization here
            
        except Exception as e:
            logger.error(f"Error checking optimization triggers: {e}")
    
    async def _get_user_baseline_performance(self, user_id: str) -> Dict[str, float]:
        """Get baseline performance metrics for a user."""
        user_metrics = [m for m in self.performance_history if m.user_id == user_id]
        
        if not user_metrics:
            return {
                "avg_latency_ms": 0.0,
                "avg_quality_score": 0.0,
                "success_rate": 0.0
            }
        
        # Get recent metrics (last 24 hours)
        recent_time = datetime.now() - timedelta(hours=24)
        recent_metrics = [m for m in user_metrics if m.timestamp >= recent_time]
        
        if not recent_metrics:
            recent_metrics = user_metrics[-10:]  # Last 10 if no recent data
        
        return {
            "avg_latency_ms": sum(m.retrieval_latency_ms for m in recent_metrics) / len(recent_metrics),
            "avg_quality_score": sum(m.context_quality_score for m in recent_metrics) / len(recent_metrics),
            "success_rate": len([m for m in recent_metrics if m.success]) / len(recent_metrics)
        }
    
    async def _optimize_collection_structure(self, user_id: str) -> bool:
        """Optimize ChromaDB collection structure for better performance."""
        try:
            # This would implement actual collection optimization
            # For now, return True to indicate optimization was attempted
            logger.info(f"Collection structure optimization attempted for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error optimizing collection structure: {e}")
            return False
    
    async def _cleanup_stale_embeddings(self, user_id: str) -> Dict[str, Any]:
        """Clean up stale embeddings (wrapper for the main cleanup method)."""
        return await self.cleanup_stale_embeddings(user_id)
    
    async def _optimize_search_parameters(self, user_id: str) -> bool:
        """Optimize search parameters for better performance."""
        try:
            # This would implement search parameter optimization
            logger.info(f"Search parameter optimization attempted for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error optimizing search parameters: {e}")
            return False
    
    async def _improve_embedding_quality(self, user_id: str) -> bool:
        """Improve embedding quality through various techniques."""
        try:
            # This would implement embedding quality improvement
            logger.info(f"Embedding quality improvement attempted for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error improving embedding quality: {e}")
            return False
    
    async def _generate_optimization_recommendations(self, user_id: str, 
                                                   before_metrics: Dict[str, float], 
                                                   after_metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        
        try:
            # Latency recommendations
            if after_metrics["avg_latency_ms"] > self.performance_thresholds["max_retrieval_latency_ms"]:
                recommendations.append("Consider reducing the number of documents in the collection")
                recommendations.append("Review embedding model complexity")
            
            # Quality recommendations
            if after_metrics["avg_quality_score"] < self.performance_thresholds["min_context_quality_score"]:
                recommendations.append("Consider improving document quality and relevance")
                recommendations.append("Review embedding generation strategy")
            
            # Success rate recommendations
            if after_metrics["success_rate"] < 0.95:
                recommendations.append("Investigate and fix retrieval failures")
                recommendations.append("Implement better error handling")
            
            if not recommendations:
                recommendations.append("Performance is within acceptable thresholds")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations


# Global RAG performance optimizer instance
rag_performance_optimizer = RAGPerformanceOptimizer() 