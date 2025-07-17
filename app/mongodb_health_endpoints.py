# /app/mongodb_health_endpoints.py
"""
MongoDB Health Monitoring REST API Endpoints

This module provides comprehensive REST API endpoints for MongoDB health monitoring,
performance analysis, optimization recommendations, and system status reporting.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from app.mongodb_health_monitor import mongodb_health_monitor
from app.chromadb_performance_optimizer import chromadb_performance_optimizer
from app.concurrent_operations_manager import concurrent_operations_manager
from app.models import (
    DatabaseHealthStatus, PerformanceMetrics, QueryRequest, RequestPriority
)

logger = logging.getLogger(__name__)

# Create router for MongoDB health endpoints
mongodb_health_router = APIRouter(prefix="/api/mongodb-health", tags=["mongodb-health"])

@mongodb_health_router.get("/status")
async def get_mongodb_health_status():
    """
    Get comprehensive MongoDB health status including connection, performance, and optimization metrics.
    
    Returns:
        Detailed health status with recommendations and performance metrics
    """
    try:
        # Get database connection health
        db_status = await mongodb_health_monitor.check_connection_health()
        
        # Get query performance analysis
        query_performance = await mongodb_health_monitor.analyze_query_performance()
        
        # Get collection statistics
        collection_stats = await mongodb_health_monitor.monitor_collection_stats()
        
        # Get connection pool status
        pool_status = await concurrent_operations_manager.manage_connection_pool()
        
        # Determine overall health status
        overall_status = DatabaseHealthStatus.HEALTHY
        if not db_status.connection_available:
            overall_status = DatabaseHealthStatus.UNHEALTHY
        elif db_status.ping_time_ms > 1000 or query_performance.avg_duration_ms > 500:
            overall_status = DatabaseHealthStatus.DEGRADED
        
        # Compile recommendations
        recommendations = []
        recommendations.extend(query_performance.recommendations)
        
        if db_status.ping_time_ms > 500:
            recommendations.append("High database latency detected - check network connectivity")
        
        if pool_status.pool_utilization > 0.8:
            recommendations.append("High connection pool utilization - consider scaling")
        
        if not recommendations:
            recommendations.append("MongoDB is operating optimally")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "overall_health": overall_status.value,
            "database_status": {
                "connection_available": db_status.connection_available,
                "ping_time_ms": db_status.ping_time_ms,
                "server_info": db_status.server_info,
                "last_check": db_status.last_check.isoformat(),
                "error_message": db_status.error_message
            },
            "query_performance": {
                "total_queries": query_performance.total_queries,
                "avg_duration_ms": query_performance.avg_duration_ms,
                "slow_queries_count": query_performance.slow_queries_count,
                "slow_query_threshold_ms": query_performance.slow_query_threshold_ms,
                "query_patterns": query_performance.query_patterns,
                "report_period": query_performance.report_period
            },
            "collection_statistics": [
                {
                    "collection_name": stats.collection_name,
                    "document_count": stats.document_count,
                    "storage_size_mb": stats.storage_size_mb,
                    "avg_object_size_bytes": stats.avg_object_size_bytes,
                    "index_count": stats.index_count,
                    "index_size_mb": stats.index_size_mb,
                    "last_updated": stats.last_updated.isoformat()
                }
                for stats in collection_stats
            ],
            "connection_pool": {
                "active_connections": pool_status.active_connections,
                "max_connections": pool_status.max_connections,
                "waiting_connections": pool_status.waiting_connections,
                "pool_utilization": pool_status.pool_utilization,
                "connection_errors": pool_status.connection_errors
            },
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting MongoDB health status: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@mongodb_health_router.get("/performance-analysis")
async def get_performance_analysis(
    threshold_ms: int = Query(100, description="Slow query threshold in milliseconds")
):
    """
    Get detailed performance analysis including slow query detection and optimization suggestions.
    
    Args:
        threshold_ms: Threshold for considering queries as slow (default: 100ms)
        
    Returns:
        Comprehensive performance analysis with optimization recommendations
    """
    try:
        # Get query performance report
        performance_report = await mongodb_health_monitor.analyze_query_performance()
        
        # Detect slow queries
        slow_queries = await mongodb_health_monitor.detect_slow_queries(threshold_ms)
        
        # Get index optimization recommendations
        index_optimization = await mongodb_health_monitor.optimize_indexes()
        
        # Get concurrent operations performance
        concurrent_stats = concurrent_operations_manager.get_performance_stats()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "performance_overview": {
                "total_queries": performance_report.total_queries,
                "avg_duration_ms": performance_report.avg_duration_ms,
                "slow_queries_count": performance_report.slow_queries_count,
                "slow_query_threshold_ms": threshold_ms,
                "query_patterns": performance_report.query_patterns
            },
            "slow_queries": [
                {
                    "query_pattern": sq.query_pattern,
                    "avg_duration_ms": sq.avg_duration_ms,
                    "execution_count": sq.execution_count,
                    "last_seen": sq.last_seen.isoformat(),
                    "suggested_indexes": sq.suggested_indexes
                }
                for sq in slow_queries
            ],
            "index_optimization": {
                "collection_name": index_optimization.collection_name,
                "created_indexes": index_optimization.created_indexes,
                "dropped_indexes": index_optimization.dropped_indexes,
                "optimization_suggestions": index_optimization.optimization_suggestions,
                "performance_impact": index_optimization.performance_impact,
                "timestamp": index_optimization.timestamp.isoformat()
            },
            "concurrent_operations": concurrent_stats,
            "recommendations": performance_report.recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting performance analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

@mongodb_health_router.get("/chromadb-status")
async def get_chromadb_status():
    """
    Get ChromaDB integration status and performance metrics.
    
    Returns:
        ChromaDB health status and optimization recommendations
    """
    try:
        # Note: This is a general status - for user-specific status, use the user endpoint
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "chromadb_integration": {
                "status": "operational",
                "message": "ChromaDB integration is active",
                "features": [
                    "User-specific collections",
                    "Persistent vector storage",
                    "Collection health validation",
                    "Performance benchmarking",
                    "Embedding integrity validation",
                    "Automatic cleanup operations"
                ]
            },
            "optimization_features": [
                "Collection structure optimization",
                "Stale embedding cleanup",
                "Performance benchmarking",
                "Integrity validation",
                "Health monitoring"
            ],
            "recommendations": [
                "Use user-specific endpoints for detailed collection analysis",
                "Regular health validation recommended",
                "Monitor collection performance metrics"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting ChromaDB status: {e}")
        raise HTTPException(status_code=500, detail=f"ChromaDB status check failed: {str(e)}")

@mongodb_health_router.get("/chromadb-user-status/{user_id}")
async def get_user_chromadb_status(user_id: str):
    """
    Get detailed ChromaDB status for a specific user.
    
    Args:
        user_id: User identifier for collection analysis
        
    Returns:
        User-specific ChromaDB health and performance metrics
    """
    try:
        # Validate collection health
        health_validation = await chromadb_performance_optimizer.validate_collection_health(user_id)
        
        # Get collection status
        collection_status = await chromadb_performance_optimizer.get_collection_status(user_id)
        
        # Benchmark retrieval performance
        benchmark_result = await chromadb_performance_optimizer.benchmark_retrieval_performance(user_id)
        
        # Validate embeddings integrity
        integrity_validation = await chromadb_performance_optimizer.validate_embeddings_integrity(user_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "collection_health": {
                "is_valid": health_validation.is_valid,
                "issues_found": health_validation.issues_found,
                "validation_time_ms": health_validation.validation_time_ms,
                "recommendations": health_validation.recommendations,
                "last_validated": health_validation.timestamp.isoformat()
            },
            "collection_status": {
                "collections_healthy": collection_status.collections_healthy,
                "total_collections": collection_status.total_collections,
                "storage_size_mb": collection_status.storage_size_mb,
                "last_check": collection_status.last_check.isoformat(),
                "error_message": collection_status.error_message
            },
            "performance_benchmark": {
                "operation_type": benchmark_result.operation_type,
                "avg_latency_ms": benchmark_result.avg_latency_ms,
                "p95_latency_ms": benchmark_result.p95_latency_ms,
                "p99_latency_ms": benchmark_result.p99_latency_ms,
                "throughput_ops_per_sec": benchmark_result.throughput_ops_per_sec,
                "success_rate": benchmark_result.success_rate,
                "benchmark_duration_sec": benchmark_result.benchmark_duration_sec,
                "timestamp": benchmark_result.timestamp.isoformat()
            },
            "embeddings_integrity": {
                "is_valid": integrity_validation.is_valid,
                "issues_found": integrity_validation.issues_found,
                "validation_time_ms": integrity_validation.validation_time_ms,
                "recommendations": integrity_validation.recommendations,
                "timestamp": integrity_validation.timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting user ChromaDB status for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"User ChromaDB status check failed: {str(e)}")

@mongodb_health_router.post("/chromadb-optimize/{user_id}")
async def optimize_user_chromadb(user_id: str):
    """
    Optimize ChromaDB collection for a specific user.
    
    Args:
        user_id: User identifier for optimization
        
    Returns:
        Optimization results and performance improvements
    """
    try:
        # Optimize collection structure
        structure_optimization = await chromadb_performance_optimizer.optimize_collection_structure(user_id)
        
        # Clean up stale embeddings
        cleanup_result = await chromadb_performance_optimizer.cleanup_stale_embeddings(user_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "structure_optimization": {
                "collection_name": structure_optimization.collection_name,
                "optimizations_applied": structure_optimization.optimizations_applied,
                "before_performance": structure_optimization.before_performance,
                "after_performance": structure_optimization.after_performance,
                "recommendations": structure_optimization.recommendations,
                "timestamp": structure_optimization.timestamp.isoformat()
            },
            "cleanup_results": {
                "collections_cleaned": cleanup_result.collections_cleaned,
                "documents_removed": cleanup_result.documents_removed,
                "storage_freed_mb": cleanup_result.storage_freed_mb,
                "cleanup_duration_sec": cleanup_result.cleanup_duration_sec,
                "errors_encountered": cleanup_result.errors_encountered,
                "timestamp": cleanup_result.timestamp.isoformat()
            },
            "optimization_summary": {
                "total_improvements": len(structure_optimization.optimizations_applied),
                "documents_cleaned": cleanup_result.documents_removed,
                "storage_saved_mb": cleanup_result.storage_freed_mb,
                "optimization_successful": len(cleanup_result.errors_encountered) == 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing ChromaDB for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"ChromaDB optimization failed: {str(e)}")

@mongodb_health_router.get("/resource-monitoring")
async def get_resource_monitoring():
    """
    Get comprehensive system resource monitoring and usage statistics.
    
    Returns:
        System resource usage and performance monitoring data
    """
    try:
        # Get current resource usage
        resource_report = await concurrent_operations_manager.monitor_resource_usage()
        
        # Get performance statistics
        performance_stats = concurrent_operations_manager.get_performance_stats()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "resource_usage": {
                "cpu_usage_percent": resource_report.cpu_usage_percent,
                "memory_usage_mb": resource_report.memory_usage_mb,
                "connection_pool_usage": resource_report.connection_pool_usage,
                "disk_io_ops_per_sec": resource_report.disk_io_ops_per_sec,
                "network_bandwidth_mbps": resource_report.network_bandwidth_mbps,
                "timestamp": resource_report.timestamp.isoformat()
            },
            "performance_statistics": performance_stats,
            "system_health": {
                "overall_status": "healthy" if resource_report.cpu_usage_percent < 80 else "degraded",
                "active_operations": performance_stats.get("active_operations", 0),
                "queued_operations": performance_stats.get("queued_operations", 0),
                "circuit_breaker_status": "open" if performance_stats.get("circuit_breaker_open", False) else "closed",
                "resource_pressure": performance_stats.get("resource_pressure", "unknown")
            },
            "recommendations": [
                "Monitor CPU usage - current: {:.1f}%".format(resource_report.cpu_usage_percent),
                "Connection pool utilization: {:.1f}%".format(resource_report.connection_pool_usage * 100),
                "System is operating within normal parameters" if resource_report.cpu_usage_percent < 80 else "Consider scaling resources"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting resource monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Resource monitoring failed: {str(e)}")

@mongodb_health_router.post("/record-performance")
async def record_performance_metric(
    operation_type: str,
    duration_ms: float,
    success: bool,
    user_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    query_pattern: Optional[str] = None
):
    """
    Record a performance metric for analysis.
    
    Args:
        operation_type: Type of operation performed
        duration_ms: Duration in milliseconds
        success: Whether the operation was successful
        user_id: Optional user identifier
        collection_name: Optional collection name
        query_pattern: Optional query pattern description
        
    Returns:
        Confirmation of metric recording
    """
    try:
        # Create performance metric
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            operation_type=operation_type,
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            collection_name=collection_name,
            query_pattern=query_pattern,
            resource_usage={}
        )
        
        # Record in MongoDB health monitor
        await mongodb_health_monitor.record_performance_metric(metric)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": "Performance metric recorded successfully",
            "metric": {
                "operation_type": operation_type,
                "duration_ms": duration_ms,
                "success": success,
                "user_id": user_id,
                "collection_name": collection_name,
                "query_pattern": query_pattern
            }
        }
        
    except Exception as e:
        logger.error(f"Error recording performance metric: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metric recording failed: {str(e)}")

@mongodb_health_router.get("/optimization-recommendations")
async def get_optimization_recommendations():
    """
    Get comprehensive optimization recommendations for the entire system.
    
    Returns:
        System-wide optimization recommendations and priorities
    """
    try:
        # Get MongoDB performance analysis
        performance_report = await mongodb_health_monitor.analyze_query_performance()
        
        # Get index optimization recommendations
        index_optimization = await mongodb_health_monitor.optimize_indexes()
        
        # Get resource monitoring
        resource_report = await concurrent_operations_manager.monitor_resource_usage()
        
        # Compile optimization recommendations
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "maintenance": []
        }
        
        # High priority recommendations
        if performance_report.avg_duration_ms > 500:
            recommendations["high_priority"].append({
                "type": "performance",
                "title": "Optimize slow queries",
                "description": f"Average query duration is {performance_report.avg_duration_ms:.1f}ms",
                "action": "Review and optimize database indexes",
                "impact": "high"
            })
        
        if resource_report.cpu_usage_percent > 90:
            recommendations["high_priority"].append({
                "type": "scaling",
                "title": "High CPU usage detected",
                "description": f"CPU usage at {resource_report.cpu_usage_percent:.1f}%",
                "action": "Consider scaling resources or optimizing operations",
                "impact": "high"
            })
        
        # Medium priority recommendations
        if performance_report.slow_queries_count > 0:
            recommendations["medium_priority"].append({
                "type": "optimization",
                "title": "Slow queries detected",
                "description": f"Found {performance_report.slow_queries_count} slow queries",
                "action": "Review query patterns and add appropriate indexes",
                "impact": "medium"
            })
        
        if resource_report.connection_pool_usage > 0.7:
            recommendations["medium_priority"].append({
                "type": "connection_pool",
                "title": "High connection pool usage",
                "description": f"Pool utilization at {resource_report.connection_pool_usage * 100:.1f}%",
                "action": "Monitor connection usage and consider pool tuning",
                "impact": "medium"
            })
        
        # Low priority recommendations
        recommendations["low_priority"].extend([
            {
                "type": "monitoring",
                "title": "Regular health checks",
                "description": "Maintain regular system health monitoring",
                "action": "Schedule automated health checks",
                "impact": "low"
            }
        ])
        
        # Maintenance recommendations
        recommendations["maintenance"].extend(index_optimization.optimization_suggestions)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "overall_health": "healthy" if len(recommendations["high_priority"]) == 0 else "needs_attention",
                "total_recommendations": sum(len(recs) for recs in recommendations.values()),
                "performance_score": max(0, 100 - (performance_report.avg_duration_ms / 10)),
                "resource_utilization": resource_report.cpu_usage_percent
            },
            "recommendations": recommendations,
            "implementation_priority": [
                "Address high priority items immediately",
                "Plan medium priority optimizations for next maintenance window",
                "Schedule low priority improvements for regular maintenance",
                "Review maintenance recommendations monthly"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization recommendations failed: {str(e)}")

@mongodb_health_router.get("/system-dashboard")
async def get_system_dashboard():
    """
    Get comprehensive system dashboard with all key metrics and status information.
    
    Returns:
        Complete system dashboard with health, performance, and optimization data
    """
    try:
        # Get all key metrics
        db_status = await mongodb_health_monitor.check_connection_health()
        performance_report = await mongodb_health_monitor.analyze_query_performance()
        resource_report = await concurrent_operations_manager.monitor_resource_usage()
        performance_stats = concurrent_operations_manager.get_performance_stats()
        
        # Calculate overall system health score
        health_score = 100
        
        if not db_status.connection_available:
            health_score -= 50
        elif db_status.ping_time_ms > 1000:
            health_score -= 20
        
        if performance_report.avg_duration_ms > 500:
            health_score -= 15
        
        if resource_report.cpu_usage_percent > 90:
            health_score -= 15
        elif resource_report.cpu_usage_percent > 70:
            health_score -= 5
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "needs_attention" if health_score >= 50 else "critical"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "dashboard": {
                "overall_health": {
                    "score": health_score,
                    "status": health_status,
                    "database_connected": db_status.connection_available,
                    "ping_time_ms": db_status.ping_time_ms,
                    "last_updated": datetime.now().isoformat()
                },
                "performance_metrics": {
                    "total_queries": performance_report.total_queries,
                    "avg_response_time_ms": performance_report.avg_duration_ms,
                    "slow_queries": performance_report.slow_queries_count,
                    "success_rate": performance_stats.get("success_rate", 0.0),
                    "active_operations": performance_stats.get("active_operations", 0),
                    "queued_operations": performance_stats.get("queued_operations", 0)
                },
                "resource_utilization": {
                    "cpu_usage_percent": resource_report.cpu_usage_percent,
                    "memory_usage_mb": resource_report.memory_usage_mb,
                    "connection_pool_usage_percent": resource_report.connection_pool_usage * 100,
                    "network_bandwidth_mbps": resource_report.network_bandwidth_mbps,
                    "disk_io_ops_per_sec": resource_report.disk_io_ops_per_sec
                },
                "system_status": {
                    "circuit_breaker_open": performance_stats.get("circuit_breaker_open", False),
                    "resource_pressure": performance_stats.get("resource_pressure", "unknown"),
                    "load_balancer_active": True,
                    "mongodb_optimization_enabled": True,
                    "chromadb_optimization_enabled": True
                },
                "alerts": [
                    {
                        "type": "warning",
                        "message": f"High CPU usage: {resource_report.cpu_usage_percent:.1f}%",
                        "priority": "high"
                    }
                ] if resource_report.cpu_usage_percent > 80 else [],
                "quick_stats": {
                    "uptime_status": "operational",
                    "last_optimization": "system_startup",
                    "monitoring_active": True,
                    "auto_optimization_enabled": True
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"System dashboard failed: {str(e)}") 