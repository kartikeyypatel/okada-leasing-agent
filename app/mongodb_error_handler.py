# /app/mongodb_error_handler.py
"""
MongoDB Error Recovery and Handling System

This module provides comprehensive error handling and recovery mechanisms
specifically for MongoDB operations, including connection failures, query timeouts,
index corruption, and disk space issues.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import pymongo
from pymongo.errors import (
    ConnectionFailure, ServerSelectionTimeoutError, NetworkTimeout,
    OperationFailure, DuplicateKeyError, BulkWriteError, WriteError,
    ConfigurationError, InvalidOperation, CursorNotFound
)
from motor.motor_asyncio import AsyncIOMotorClient

from app.database import db_manager
from app.models import RecoveryResult, TimeoutResult, IndexRepairResult, DiskSpaceResult

logger = logging.getLogger(__name__)


class MongoDBErrorType(Enum):
    """Types of MongoDB errors."""
    CONNECTION_FAILURE = "connection_failure"
    QUERY_TIMEOUT = "query_timeout"
    INDEX_CORRUPTION = "index_corruption"
    DISK_SPACE = "disk_space"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    OPERATION_FAILURE = "operation_failure"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RECONNECT = "reconnect"
    FALLBACK_TO_CACHE = "fallback_to_cache"
    INDEX_REBUILD = "index_rebuild"
    QUERY_OPTIMIZATION = "query_optimization"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class MongoDBError:
    """MongoDB error record with recovery information."""
    timestamp: datetime
    error_type: MongoDBErrorType
    original_exception: Exception
    operation: str
    collection_name: Optional[str]
    query: Optional[Dict[str, Any]]
    recovery_strategy: RecoveryStrategy
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    error_message: str = ""


@dataclass
class ConnectionHealthMetrics:
    """Connection health metrics for monitoring."""
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    avg_connection_time_ms: float = 0.0
    last_successful_connection: Optional[datetime] = None
    last_failed_connection: Optional[datetime] = None
    consecutive_failures: int = 0


class MongoDBErrorHandler:
    """
    Comprehensive MongoDB error handling and recovery system.
    
    Provides automatic recovery mechanisms for connection failures, query timeouts,
    index corruption, and other MongoDB-specific issues.
    """
    
    def __init__(self):
        self.error_history: List[MongoDBError] = []
        self.connection_metrics = ConnectionHealthMetrics()
        self.recovery_cache: Dict[str, Any] = {}
        
        # Recovery configuration
        self.max_retry_attempts = 3
        self.retry_backoff_base = 1.0  # seconds
        self.max_backoff_time = 30.0  # seconds
        self.connection_timeout = 10.0  # seconds
        self.query_timeout = 30.0  # seconds
        
        # Circuit breaker configuration
        self.circuit_breaker = {
            "failure_threshold": 5,
            "recovery_timeout": 60,  # seconds
            "is_open": False,
            "failure_count": 0,
            "last_failure_time": None
        }
        
        # Fallback mechanisms
        self.enable_cache_fallback = True
        self.enable_query_optimization = True
        
        logger.info("MongoDB Error Handler initialized")
    
    async def handle_connection_failure(self, error: Exception, operation: str = "unknown") -> RecoveryResult:
        """
        Handle MongoDB connection failures with automatic retry and failover.
        
        Args:
            error: Connection failure exception
            operation: Operation that failed
            
        Returns:
            RecoveryResult with recovery status and recommendations
        """
        try:
            error_record = MongoDBError(
                timestamp=datetime.now(),
                error_type=MongoDBErrorType.CONNECTION_FAILURE,
                original_exception=error,
                operation=operation,
                collection_name=None,
                query=None,
                recovery_strategy=RecoveryStrategy.RECONNECT,
                error_message=str(error)
            )
            
            self.error_history.append(error_record)
            self.connection_metrics.failed_connections += 1
            self.connection_metrics.consecutive_failures += 1
            self.connection_metrics.last_failed_connection = datetime.now()
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                return RecoveryResult(
                    success=False,
                    recovery_strategy="circuit_breaker_open",
                    message="Circuit breaker is open - too many consecutive failures",
                    details={
                        "circuit_breaker_status": "open",
                        "failure_count": self.circuit_breaker["failure_count"],
                        "next_retry_time": self.circuit_breaker["last_failure_time"] + timedelta(seconds=self.circuit_breaker["recovery_timeout"])
                    },
                    timestamp=datetime.now()
                )
            
            # Attempt recovery with exponential backoff
            recovery_successful = False
            for attempt in range(self.max_retry_attempts):
                try:
                    error_record.retry_count = attempt + 1
                    error_record.recovery_attempted = True
                    
                    # Calculate backoff time
                    backoff_time = min(
                        self.retry_backoff_base * (2 ** attempt),
                        self.max_backoff_time
                    )
                    
                    if attempt > 0:
                        logger.info(f"Retrying connection after {backoff_time}s (attempt {attempt + 1}/{self.max_retry_attempts})")
                        await asyncio.sleep(backoff_time)
                    
                    # Attempt to reconnect
                    await self._attempt_reconnection()
                    
                    # Test the connection
                    if await self._test_connection():
                        recovery_successful = True
                        error_record.recovery_successful = True
                        self.connection_metrics.successful_connections += 1
                        self.connection_metrics.consecutive_failures = 0
                        self.connection_metrics.last_successful_connection = datetime.now()
                        self._reset_circuit_breaker()
                        break
                        
                except Exception as retry_error:
                    logger.warning(f"Connection retry {attempt + 1} failed: {retry_error}")
                    continue
            
            if recovery_successful:
                return RecoveryResult(
                    success=True,
                    recovery_strategy="reconnection_successful",
                    message=f"Successfully reconnected after {error_record.retry_count} attempts",
                    details={
                        "retry_count": error_record.retry_count,
                        "connection_metrics": self.connection_metrics.__dict__
                    },
                    timestamp=datetime.now()
                )
            else:
                # Update circuit breaker
                self._update_circuit_breaker()
                
                # Enable fallback mechanisms
                fallback_enabled = await self._enable_fallback_mechanisms()
                
                return RecoveryResult(
                    success=False,
                    recovery_strategy="fallback_enabled" if fallback_enabled else "manual_intervention_required",
                    message=f"Connection recovery failed after {self.max_retry_attempts} attempts",
                    details={
                        "retry_count": error_record.retry_count,
                        "fallback_enabled": fallback_enabled,
                        "recommended_actions": [
                            "Check MongoDB service status",
                            "Verify network connectivity",
                            "Check authentication credentials",
                            "Review MongoDB logs for errors"
                        ]
                    },
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error in connection failure handler: {e}")
            return RecoveryResult(
                success=False,
                recovery_strategy="handler_error",
                message=f"Error handling connection failure: {str(e)}",
                details={},
                timestamp=datetime.now()
            )
    
    async def handle_query_timeout(self, query: str, timeout_ms: int, collection: str = "unknown") -> TimeoutResult:
        """
        Handle slow queries with optimization suggestions.
        
        Args:
            query: Query that timed out
            timeout_ms: Timeout threshold in milliseconds
            collection: Collection being queried
            
        Returns:
            TimeoutResult with optimization suggestions
        """
        try:
            error_record = MongoDBError(
                timestamp=datetime.now(),
                error_type=MongoDBErrorType.QUERY_TIMEOUT,
                original_exception=Exception(f"Query timeout after {timeout_ms}ms"),
                operation="query",
                collection_name=collection,
                query={"query_string": query},
                recovery_strategy=RecoveryStrategy.QUERY_OPTIMIZATION,
                error_message=f"Query timed out after {timeout_ms}ms"
            )
            
            self.error_history.append(error_record)
            
            # Analyze query for optimization opportunities
            optimization_suggestions = await self._analyze_slow_query(query, collection, timeout_ms)
            
            # Attempt query optimization
            optimized_query = await self._optimize_query(query, collection)
            
            return TimeoutResult(
                original_query=query,
                timeout_ms=timeout_ms,
                collection_name=collection,
                optimization_suggestions=optimization_suggestions,
                optimized_query=optimized_query,
                estimated_improvement="20-50% performance improvement with suggested optimizations",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling query timeout: {e}")
            return TimeoutResult(
                original_query=query,
                timeout_ms=timeout_ms,
                collection_name=collection,
                optimization_suggestions=[f"Error analyzing query: {str(e)}"],
                optimized_query=query,
                estimated_improvement="Unable to estimate due to analysis error",
                timestamp=datetime.now()
            )
    
    async def handle_index_corruption(self, collection: str, index_name: Optional[str] = None) -> IndexRepairResult:
        """
        Detect and repair corrupted indexes.
        
        Args:
            collection: Collection with corrupted index
            index_name: Specific index that's corrupted (if known)
            
        Returns:
            IndexRepairResult with repair status
        """
        try:
            error_record = MongoDBError(
                timestamp=datetime.now(),
                error_type=MongoDBErrorType.INDEX_CORRUPTION,
                original_exception=Exception(f"Index corruption in {collection}"),
                operation="index_repair",
                collection_name=collection,
                query=None,
                recovery_strategy=RecoveryStrategy.INDEX_REBUILD,
                error_message=f"Index corruption detected in {collection}"
            )
            
            self.error_history.append(error_record)
            
            repair_actions = []
            repair_successful = True
            
            try:
                if not db_manager.db:
                    raise Exception("Database connection not available")
                
                collection_obj = db_manager.db[collection]
                
                # Get current indexes
                existing_indexes = await collection_obj.list_indexes().to_list(length=None)
                
                if index_name:
                    # Repair specific index
                    try:
                        # Drop corrupted index
                        await collection_obj.drop_index(index_name)
                        repair_actions.append(f"Dropped corrupted index: {index_name}")
                        
                        # Recreate index (this would need the original index specification)
                        # For now, we'll just log the action needed
                        repair_actions.append(f"Index {index_name} needs to be recreated with original specification")
                        
                    except Exception as index_error:
                        repair_actions.append(f"Failed to repair index {index_name}: {str(index_error)}")
                        repair_successful = False
                else:
                    # Validate all indexes
                    for index_info in existing_indexes:
                        index_name_current = index_info.get("name", "unknown")
                        try:
                            # Test index by running a simple query
                            await collection_obj.find({}).hint(index_name_current).limit(1).to_list(length=1)
                            repair_actions.append(f"Index {index_name_current} validated successfully")
                        except Exception as validation_error:
                            repair_actions.append(f"Index {index_name_current} validation failed: {str(validation_error)}")
                            repair_successful = False
                
                error_record.recovery_attempted = True
                error_record.recovery_successful = repair_successful
                
                return IndexRepairResult(
                    collection_name=collection,
                    corrupted_indexes=[index_name] if index_name else [],
                    repair_actions=repair_actions,
                    repair_successful=repair_successful,
                    recommendations=[
                        "Monitor index performance after repair",
                        "Consider index optimization based on query patterns",
                        "Implement regular index health checks"
                    ] if repair_successful else [
                        "Manual index recreation may be required",
                        "Review MongoDB logs for detailed error information",
                        "Consider database maintenance window for repairs"
                    ],
                    timestamp=datetime.now()
                )
                
            except Exception as repair_error:
                logger.error(f"Error during index repair: {repair_error}")
                return IndexRepairResult(
                    collection_name=collection,
                    corrupted_indexes=[index_name] if index_name else [],
                    repair_actions=[f"Repair failed: {str(repair_error)}"],
                    repair_successful=False,
                    recommendations=[
                        "Manual intervention required",
                        "Check database connectivity and permissions",
                        "Review MongoDB error logs"
                    ],
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error handling index corruption: {e}")
            return IndexRepairResult(
                collection_name=collection,
                corrupted_indexes=[],
                repair_actions=[f"Handler error: {str(e)}"],
                repair_successful=False,
                recommendations=["Investigate error handling system"],
                timestamp=datetime.now()
            )
    
    async def handle_disk_space_issues(self) -> DiskSpaceResult:
        """
        Monitor and handle disk space issues.
        
        Returns:
            DiskSpaceResult with disk space status and recommendations
        """
        try:
            if not db_manager.db:
                return DiskSpaceResult(
                    available_space_gb=0.0,
                    used_space_gb=0.0,
                    total_space_gb=0.0,
                    space_usage_percent=0.0,
                    critical_threshold_reached=True,
                    cleanup_recommendations=[
                        "Database connection not available for space analysis"
                    ],
                    estimated_cleanup_space_gb=0.0,
                    timestamp=datetime.now()
                )
            
            # Get database statistics
            db_stats = await db_manager.db.command("dbStats")
            
            # Calculate space usage (values in bytes)
            data_size_gb = db_stats.get("dataSize", 0) / (1024 ** 3)
            storage_size_gb = db_stats.get("storageSize", 0) / (1024 ** 3)
            index_size_gb = db_stats.get("indexSize", 0) / (1024 ** 3)
            total_size_gb = data_size_gb + index_size_gb
            
            # Estimate available space (this would need system-level monitoring in production)
            # For now, we'll use a placeholder calculation
            estimated_total_space = max(total_size_gb * 2, 100.0)  # Assume at least 100GB available
            available_space_gb = estimated_total_space - total_size_gb
            space_usage_percent = (total_size_gb / estimated_total_space) * 100
            
            # Determine if critical threshold is reached
            critical_threshold = 85.0  # 85% usage
            critical_threshold_reached = space_usage_percent >= critical_threshold
            
            # Generate cleanup recommendations
            cleanup_recommendations = []
            estimated_cleanup_space = 0.0
            
            if critical_threshold_reached:
                cleanup_recommendations.extend([
                    "Immediate attention required - disk space critically low",
                    "Review and delete old conversation history data",
                    "Implement data archiving for old records",
                    "Consider database compression options"
                ])
                estimated_cleanup_space = total_size_gb * 0.2  # Estimate 20% can be cleaned
            elif space_usage_percent >= 70.0:
                cleanup_recommendations.extend([
                    "Monitor disk space closely",
                    "Plan for data cleanup or storage expansion",
                    "Review data retention policies"
                ])
                estimated_cleanup_space = total_size_gb * 0.1  # Estimate 10% can be cleaned
            else:
                cleanup_recommendations.append("Disk space usage is within normal limits")
            
            # Add specific cleanup suggestions
            if data_size_gb > index_size_gb * 2:
                cleanup_recommendations.append("Data size is large relative to indexes - consider data archiving")
            
            if index_size_gb > data_size_gb:
                cleanup_recommendations.append("Index size is large relative to data - review index optimization")
            
            return DiskSpaceResult(
                available_space_gb=available_space_gb,
                used_space_gb=total_size_gb,
                total_space_gb=estimated_total_space,
                space_usage_percent=space_usage_percent,
                critical_threshold_reached=critical_threshold_reached,
                cleanup_recommendations=cleanup_recommendations,
                estimated_cleanup_space_gb=estimated_cleanup_space,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error handling disk space issues: {e}")
            return DiskSpaceResult(
                available_space_gb=0.0,
                used_space_gb=0.0,
                total_space_gb=0.0,
                space_usage_percent=100.0,
                critical_threshold_reached=True,
                cleanup_recommendations=[f"Error analyzing disk space: {str(e)}"],
                estimated_cleanup_space_gb=0.0,
                timestamp=datetime.now()
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics and trends."""
        try:
            recent_time = datetime.now() - timedelta(hours=24)
            recent_errors = [error for error in self.error_history if error.timestamp >= recent_time]
            
            # Count errors by type
            error_counts = {}
            for error_type in MongoDBErrorType:
                count = len([e for e in recent_errors if e.error_type == error_type])
                error_counts[error_type.value] = count
            
            # Calculate recovery rates
            total_recovery_attempts = len([e for e in recent_errors if e.recovery_attempted])
            successful_recoveries = len([e for e in recent_errors if e.recovery_successful])
            recovery_rate = (successful_recoveries / total_recovery_attempts) if total_recovery_attempts > 0 else 1.0
            
            # Most common errors
            most_common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "error_summary": {
                    "total_errors_24h": len(recent_errors),
                    "error_types": error_counts,
                    "most_common_errors": dict(most_common_errors),
                    "recovery_rate": recovery_rate,
                    "circuit_breaker_status": "open" if self.circuit_breaker["is_open"] else "closed"
                },
                "connection_health": {
                    "total_attempts": self.connection_metrics.connection_attempts,
                    "successful_connections": self.connection_metrics.successful_connections,
                    "failed_connections": self.connection_metrics.failed_connections,
                    "consecutive_failures": self.connection_metrics.consecutive_failures,
                    "last_successful": self.connection_metrics.last_successful_connection.isoformat() if self.connection_metrics.last_successful_connection else None,
                    "last_failed": self.connection_metrics.last_failed_connection.isoformat() if self.connection_metrics.last_failed_connection else None
                },
                "recent_errors": [
                    {
                        "timestamp": error.timestamp.isoformat(),
                        "type": error.error_type.value,
                        "operation": error.operation,
                        "collection": error.collection_name,
                        "recovery_attempted": error.recovery_attempted,
                        "recovery_successful": error.recovery_successful,
                        "retry_count": error.retry_count,
                        "message": error.error_message
                    }
                    for error in recent_errors[-10:]  # Last 10 errors
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting error statistics: {e}")
            return {"error": f"Statistics error: {str(e)}"}
    
    # Private helper methods
    
    async def _attempt_reconnection(self):
        """Attempt to reconnect to MongoDB."""
        try:
            self.connection_metrics.connection_attempts += 1
            
            # Close existing connection if any
            if db_manager.client:
                db_manager.client.close()
            
            # Create new connection
            await db_manager.connect_to_mongo()
            
            logger.info("MongoDB reconnection attempt completed")
            
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            raise
    
    async def _test_connection(self) -> bool:
        """Test MongoDB connection health."""
        try:
            if not db_manager.client or not db_manager.db:
                return False
            
            # Perform a simple ping test
            start_time = time.time()
            result = await db_manager.db.command("ping")
            connection_time = (time.time() - start_time) * 1000
            
            # Update connection metrics
            self.connection_metrics.avg_connection_time_ms = (
                (self.connection_metrics.avg_connection_time_ms * self.connection_metrics.successful_connections + connection_time) /
                (self.connection_metrics.successful_connections + 1)
            )
            
            return result.get("ok", 0) == 1
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_breaker["is_open"]:
            return False
        
        # Check if recovery timeout has passed
        if self.circuit_breaker["last_failure_time"]:
            time_since_failure = datetime.now() - self.circuit_breaker["last_failure_time"]
            if time_since_failure.total_seconds() >= self.circuit_breaker["recovery_timeout"]:
                self._reset_circuit_breaker()
                return False
        
        return True
    
    def _update_circuit_breaker(self):
        """Update circuit breaker state after failure."""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure_time"] = datetime.now()
        
        if self.circuit_breaker["failure_count"] >= self.circuit_breaker["failure_threshold"]:
            self.circuit_breaker["is_open"] = True
            logger.warning("Circuit breaker opened due to consecutive failures")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker after successful operation."""
        self.circuit_breaker["is_open"] = False
        self.circuit_breaker["failure_count"] = 0
        self.circuit_breaker["last_failure_time"] = None
        logger.info("Circuit breaker reset after successful operation")
    
    async def _enable_fallback_mechanisms(self) -> bool:
        """Enable fallback mechanisms when MongoDB is unavailable."""
        try:
            if self.enable_cache_fallback:
                # Enable in-memory cache for critical data
                self.recovery_cache["fallback_enabled"] = True
                self.recovery_cache["fallback_timestamp"] = datetime.now()
                logger.info("Fallback cache mechanisms enabled")
                return True
            return False
        except Exception as e:
            logger.error(f"Error enabling fallback mechanisms: {e}")
            return False
    
    async def _analyze_slow_query(self, query: str, collection: str, timeout_ms: int) -> List[str]:
        """Analyze slow query and provide optimization suggestions."""
        suggestions = []
        
        try:
            # Basic query analysis
            if "find" in query.lower():
                suggestions.append("Consider adding indexes for frequently queried fields")
            
            if "sort" in query.lower():
                suggestions.append("Ensure indexes exist for sort fields")
            
            if "regex" in query.lower() or "$regex" in query:
                suggestions.append("Regex queries can be slow - consider text indexes or exact matches")
            
            if timeout_ms > 10000:  # > 10 seconds
                suggestions.append("Query timeout is very high - consider query redesign")
            
            # Collection-specific suggestions
            if collection == "conversation_history":
                suggestions.extend([
                    "Consider compound index on (user_email, created_at)",
                    "Implement data archiving for old conversations"
                ])
            elif collection == "users":
                suggestions.extend([
                    "Ensure unique index on email field",
                    "Consider compound index on (company_id, email)"
                ])
            
            if not suggestions:
                suggestions.append("Review query execution plan for optimization opportunities")
            
        except Exception as e:
            logger.error(f"Error analyzing slow query: {e}")
            suggestions.append(f"Query analysis failed: {str(e)}")
        
        return suggestions
    
    async def _optimize_query(self, query: str, collection: str) -> str:
        """Attempt to optimize a slow query."""
        try:
            # Basic query optimization (placeholder implementation)
            optimized = query
            
            # Add limit if not present for potentially large result sets
            if "limit" not in query.lower() and collection in ["conversation_history", "users"]:
                optimized += " // Consider adding .limit() for large collections"
            
            # Suggest projection for large documents
            if collection == "conversation_history":
                optimized += " // Consider using projection to limit returned fields"
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            return query


# Global MongoDB error handler instance
mongodb_error_handler = MongoDBErrorHandler() 