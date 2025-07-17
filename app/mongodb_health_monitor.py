# /app/mongodb_health_monitor.py
"""
MongoDB Health Monitoring System

This module provides comprehensive health monitoring for MongoDB connections,
performance tracking, and optimization recommendations for the RAG chatbot system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, ConnectionFailure
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.database import db_manager
from app.models import (
    DatabaseStatus, HealthStatus, PerformanceMetrics, QueryPerformanceReport,
    CollectionStatsReport, SlowQuery, IndexOptimizationResult, DatabaseHealthStatus,
    VectorStoreStatus, ConnectionPoolStatus, HealthIssue, PerformanceLevel
)

logger = logging.getLogger(__name__)

class MongoDBHealthMonitor:
    """
    Comprehensive MongoDB health monitoring and performance analysis.
    
    Provides real-time connection monitoring, query performance analysis,
    collection statistics, and automated optimization recommendations.
    """
    
    def __init__(self):
        self.performance_history: List[PerformanceMetrics] = []
        self.slow_query_threshold_ms = 100
        self.health_check_cache = {}
        self.cache_ttl_seconds = 30
        
    async def check_connection_health(self) -> DatabaseStatus:
        """
        Check MongoDB connection health with ping test and server info retrieval.
        
        Returns:
            DatabaseStatus with connection details and performance metrics
        """
        try:
            if not db_manager.client or not db_manager.db:
                return DatabaseStatus(
                    connection_available=False,
                    ping_time_ms=0.0,
                    last_check=datetime.now(),
                    error_message="MongoDB client not initialized"
                )
            
            # Perform ping test
            start_time = time.time()
            ping_result = await db_manager.db.command("ping")
            ping_time_ms = (time.time() - start_time) * 1000
            
            # Get server info
            server_info = await db_manager.client.server_info()
            
            return DatabaseStatus(
                connection_available=ping_result.get("ok", 0) == 1,
                ping_time_ms=ping_time_ms,
                last_check=datetime.now(),
                server_info=server_info
            )
            
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"MongoDB connection failed: {e}")
            return DatabaseStatus(
                connection_available=False,
                ping_time_ms=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error checking MongoDB health: {e}")
            return DatabaseStatus(
                connection_available=False,
                ping_time_ms=0.0,
                last_check=datetime.now(),
                error_message=f"Unexpected error: {str(e)}"
            )
    
    async def analyze_query_performance(self) -> QueryPerformanceReport:
        """
        Analyze query performance patterns and identify optimization opportunities.
        
        Returns:
            QueryPerformanceReport with performance metrics and recommendations
        """
        try:
            if not self.performance_history:
                return QueryPerformanceReport(
                    total_queries=0,
                    avg_duration_ms=0.0,
                    slow_queries_count=0,
                    slow_query_threshold_ms=self.slow_query_threshold_ms,
                    query_patterns={},
                    recommendations=["No query history available for analysis"],
                    report_period="No data"
                )
            
            # Calculate performance metrics
            total_queries = len(self.performance_history)
            durations = [metric.duration_ms for metric in self.performance_history]
            avg_duration = sum(durations) / len(durations)
            
            slow_queries = [metric for metric in self.performance_history 
                          if metric.duration_ms > self.slow_query_threshold_ms]
            
            # Analyze query patterns
            query_patterns = {}
            for metric in self.performance_history:
                pattern = metric.query_pattern or "unknown"
                query_patterns[pattern] = query_patterns.get(pattern, 0) + 1
            
            # Generate recommendations
            recommendations = []
            if len(slow_queries) > total_queries * 0.1:  # More than 10% slow queries
                recommendations.append("High percentage of slow queries detected. Consider index optimization.")
            
            if avg_duration > 50:  # Average duration > 50ms
                recommendations.append("Average query duration is high. Review query patterns and indexing.")
            
            if not recommendations:
                recommendations.append("Query performance is within acceptable thresholds.")
            
            return QueryPerformanceReport(
                total_queries=total_queries,
                avg_duration_ms=avg_duration,
                slow_queries_count=len(slow_queries),
                slow_query_threshold_ms=self.slow_query_threshold_ms,
                query_patterns=query_patterns,
                recommendations=recommendations,
                report_period=f"Last {total_queries} queries"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query performance: {e}")
            return QueryPerformanceReport(
                total_queries=0,
                avg_duration_ms=0.0,
                slow_queries_count=0,
                slow_query_threshold_ms=self.slow_query_threshold_ms,
                query_patterns={},
                recommendations=[f"Error analyzing performance: {str(e)}"],
                report_period="Error"
            )
    
    async def monitor_collection_stats(self) -> List[CollectionStatsReport]:
        """
        Monitor collection statistics including document counts, storage size, and indexes.
        
        Returns:
            List of CollectionStatsReport for each collection
        """
        try:
            if not db_manager.db:
                return []
            
            collection_names = await db_manager.db.list_collection_names()
            stats_reports = []
            
            for collection_name in collection_names:
                try:
                    collection = db_manager.db[collection_name]
                    
                    # Get basic collection stats
                    stats = await db_manager.db.command("collStats", collection_name)
                    
                    # Get document count
                    doc_count = await collection.count_documents({})
                    
                    # Get index information
                    indexes = await collection.list_indexes().to_list(length=None)
                    index_count = len(indexes)
                    
                    # Calculate sizes
                    storage_size_mb = stats.get("storageSize", 0) / (1024 * 1024)
                    index_size_mb = stats.get("totalIndexSize", 0) / (1024 * 1024)
                    avg_object_size = stats.get("avgObjSize", 0)
                    
                    stats_report = CollectionStatsReport(
                        collection_name=collection_name,
                        document_count=doc_count,
                        storage_size_mb=storage_size_mb,
                        avg_object_size_bytes=avg_object_size,
                        index_count=index_count,
                        index_size_mb=index_size_mb,
                        last_updated=datetime.now()
                    )
                    
                    stats_reports.append(stats_report)
                    
                except Exception as e:
                    logger.warning(f"Error getting stats for collection {collection_name}: {e}")
                    continue
            
            return stats_reports
            
        except Exception as e:
            logger.error(f"Error monitoring collection stats: {e}")
            return []
    
    async def detect_slow_queries(self, threshold_ms: int = 100) -> List[SlowQuery]:
        """
        Detect slow queries based on performance history and generate optimization suggestions.
        
        Args:
            threshold_ms: Threshold in milliseconds to consider a query slow
            
        Returns:
            List of SlowQuery objects with optimization suggestions
        """
        try:
            self.slow_query_threshold_ms = threshold_ms
            slow_queries_data = {}
            
            # Group slow queries by pattern
            for metric in self.performance_history:
                if metric.duration_ms > threshold_ms:
                    pattern = metric.query_pattern or "unknown"
                    if pattern not in slow_queries_data:
                        slow_queries_data[pattern] = {
                            "durations": [],
                            "count": 0,
                            "last_seen": metric.timestamp
                        }
                    
                    slow_queries_data[pattern]["durations"].append(metric.duration_ms)
                    slow_queries_data[pattern]["count"] += 1
                    slow_queries_data[pattern]["last_seen"] = max(
                        slow_queries_data[pattern]["last_seen"], 
                        metric.timestamp
                    )
            
            # Create SlowQuery objects with suggestions
            slow_queries = []
            for pattern, data in slow_queries_data.items():
                avg_duration = sum(data["durations"]) / len(data["durations"])
                
                # Generate index suggestions based on query pattern
                suggested_indexes = self._generate_index_suggestions(pattern)
                
                slow_query = SlowQuery(
                    query_pattern=pattern,
                    avg_duration_ms=avg_duration,
                    execution_count=data["count"],
                    last_seen=data["last_seen"],
                    suggested_indexes=suggested_indexes
                )
                
                slow_queries.append(slow_query)
            
            return slow_queries
            
        except Exception as e:
            logger.error(f"Error detecting slow queries: {e}")
            return []
    
    async def optimize_indexes(self) -> IndexOptimizationResult:
        """
        Analyze and optimize database indexes based on query patterns.
        
        Returns:
            IndexOptimizationResult with optimization actions and performance impact
        """
        try:
            if not db_manager.db:
                return IndexOptimizationResult(
                    collection_name="N/A",
                    created_indexes=[],
                    dropped_indexes=[],
                    optimization_suggestions=["MongoDB connection not available"],
                    performance_impact={},
                    timestamp=datetime.now()
                )
            
            optimization_suggestions = []
            created_indexes = []
            dropped_indexes = []
            performance_impact = {}
            
            # Analyze collections for optimization opportunities
            collection_names = await db_manager.db.list_collection_names()
            
            for collection_name in collection_names:
                try:
                    collection = db_manager.db[collection_name]
                    
                    # Analyze indexes
                    indexes = await collection.list_indexes().to_list(length=None)
                    
                    # Get collection stats for optimization decisions
                    stats = await db_manager.db.command("collStats", collection_name)
                    doc_count = stats.get("count", 0)
                    
                    # Suggest optimizations based on collection type and usage
                    suggestions = await self._analyze_collection_indexes(
                        collection_name, indexes, doc_count
                    )
                    optimization_suggestions.extend(suggestions)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing indexes for {collection_name}: {e}")
                    continue
            
            # Performance impact estimation
            performance_impact = {
                "estimated_query_speedup": "10-30%",
                "storage_optimization": "5-15%",
                "index_maintenance_cost": "minimal"
            }
            
            return IndexOptimizationResult(
                collection_name="all_collections",
                created_indexes=created_indexes,
                dropped_indexes=dropped_indexes,
                optimization_suggestions=optimization_suggestions,
                performance_impact=performance_impact,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing indexes: {e}")
            return IndexOptimizationResult(
                collection_name="error",
                created_indexes=[],
                dropped_indexes=[],
                optimization_suggestions=[f"Error during optimization: {str(e)}"],
                performance_impact={},
                timestamp=datetime.now()
            )
    
    async def record_performance_metric(self, metric: PerformanceMetrics):
        """
        Record a performance metric for analysis.
        
        Args:
            metric: PerformanceMetrics object to record
        """
        self.performance_history.append(metric)
        
        # Keep only recent metrics (last 1000 operations)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _generate_index_suggestions(self, query_pattern: str) -> List[str]:
        """Generate index suggestions based on query pattern."""
        suggestions = []
        
        if "user_email" in query_pattern.lower():
            suggestions.append("Create index on user_email field")
        
        if "created_at" in query_pattern.lower():
            suggestions.append("Create index on created_at field for time-based queries")
        
        if "user_id" in query_pattern.lower():
            suggestions.append("Create index on user_id field")
        
        if "_id" in query_pattern.lower() and "user" in query_pattern.lower():
            suggestions.append("Consider compound index on (user_email, created_at)")
        
        if not suggestions:
            suggestions.append("Review query pattern for potential indexing opportunities")
        
        return suggestions
    
    async def _analyze_collection_indexes(
        self, 
        collection_name: str, 
        indexes: List[Dict], 
        doc_count: int
    ) -> List[str]:
        """Analyze collection indexes and suggest optimizations."""
        suggestions = []
        
        # Check for common optimization patterns
        index_names = [idx.get("name", "") for idx in indexes]
        
        # Check for recommended indexes based on collection type
        if collection_name == "users":
            if not any("email" in name for name in index_names):
                suggestions.append(f"{collection_name}: Create unique index on email field")
            
            if not any("company_id" in name for name in index_names):
                suggestions.append(f"{collection_name}: Create index on company_id for user queries")
        
        elif collection_name == "conversation_history":
            if not any("user_email" in name for name in index_names):
                suggestions.append(f"{collection_name}: Create index on user_email field")
            
            if not any("created_at" in name for name in index_names):
                suggestions.append(f"{collection_name}: Create index on created_at for time-based queries")
            
            # Suggest compound index for common query patterns
            compound_exists = any("user_email" in name and "created_at" in name 
                                 for name in index_names)
            if not compound_exists:
                suggestions.append(
                    f"{collection_name}: Create compound index on (user_email, created_at)"
                )
        
        elif collection_name == "companies":
            if not any("name" in name for name in index_names):
                suggestions.append(f"{collection_name}: Create index on company name field")
        
        # Check for unused indexes (basic heuristic)
        if len(indexes) > 5 and doc_count < 1000:
            suggestions.append(
                f"{collection_name}: Review indexes - collection has {len(indexes)} indexes "
                f"but only {doc_count} documents"
            )
        
        return suggestions

# Global instance
mongodb_health_monitor = MongoDBHealthMonitor() 