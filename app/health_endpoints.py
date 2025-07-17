# /app/health_endpoints.py
"""
Health monitoring and system status endpoints for the RAG chatbot.

This module provides endpoints for:
1. System health monitoring and alerting
2. Error tracking and analysis
3. Performance metrics
4. Recovery status reporting
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
import asyncio
import time
from datetime import datetime, timedelta

from app.error_handler import rag_error_handler
from app.chroma_client import chroma_manager
import app.rag as rag_module

logger = logging.getLogger(__name__)

# Create router for health endpoints
health_router = APIRouter(prefix="/api/health", tags=["health"])

# Import performance monitor
try:
    from app.performance_monitor import performance_monitor, OperationType
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    performance_monitor = None


@health_router.get("/status")
async def get_system_health_status():
    """
    Get comprehensive system health status including error rates and recovery metrics.
    
    Returns:
        - Overall system status (healthy, stressed, degraded, critical)
        - Error counts by category and severity
        - Recovery success rates
        - Recent error patterns
        - System performance metrics
    """
    try:
        health_report = rag_error_handler.get_health_status()
        
        # Add additional system checks
        additional_checks = await _perform_system_checks()
        health_report["system_checks"] = additional_checks
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "health_report": health_report
        }
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@health_router.get("/errors")
async def get_error_history(
    hours: int = 24,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """
    Get error history with filtering options.
    
    Args:
        hours: Number of hours to look back (default: 24)
        category: Filter by error category
        severity: Filter by error severity
        limit: Maximum number of errors to return
    """
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get filtered error history
        error_history = rag_error_handler.health_monitor.error_history
        filtered_errors = [
            error for error in error_history
            if error.timestamp >= cutoff_time
        ]
        
        # Apply category filter
        if category:
            filtered_errors = [
                error for error in filtered_errors
                if error.category.value == category
            ]
        
        # Apply severity filter
        if severity:
            filtered_errors = [
                error for error in filtered_errors
                if error.severity.value == severity
            ]
        
        # Limit results
        filtered_errors = filtered_errors[-limit:]
        
        # Format for response
        formatted_errors = []
        for error in filtered_errors:
            formatted_errors.append({
                "timestamp": error.timestamp.isoformat(),
                "error_type": error.error_type,
                "error_message": error.error_message,
                "category": error.category.value,
                "severity": error.severity.value,
                "user_id": error.context.user_id,
                "operation": error.context.operation,
                "recovery_attempted": error.recovery_attempted,
                "recovery_successful": error.recovery_successful,
                "recovery_actions": error.recovery_actions,
                "user_message": error.user_message
            })
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "filters": {
                "hours": hours,
                "category": category,
                "severity": severity,
                "limit": limit
            },
            "total_errors": len(formatted_errors),
            "errors": formatted_errors
        }
    except Exception as e:
        logger.error(f"Error getting error history: {e}")
        raise HTTPException(status_code=500, detail=f"Error history retrieval failed: {str(e)}")


@health_router.get("/metrics")
async def get_system_metrics():
    """
    Get detailed system performance metrics.
    
    Returns:
        - Error rates by time period
        - Recovery success rates by category
        - System component health
        - Performance indicators
    """
    try:
        health_monitor = rag_error_handler.health_monitor
        
        # Calculate time-based metrics
        now = datetime.now()
        time_periods = {
            "last_hour": now - timedelta(hours=1),
            "last_6_hours": now - timedelta(hours=6),
            "last_24_hours": now - timedelta(hours=24),
            "last_week": now - timedelta(days=7)
        }
        
        metrics = {}
        for period_name, cutoff_time in time_periods.items():
            period_errors = [
                error for error in health_monitor.error_history
                if error.timestamp >= cutoff_time
            ]
            
            metrics[period_name] = {
                "total_errors": len(period_errors),
                "errors_by_category": _count_by_attribute(period_errors, "category"),
                "errors_by_severity": _count_by_attribute(period_errors, "severity"),
                "recovery_attempts": len([e for e in period_errors if e.recovery_attempted]),
                "recovery_successes": len([e for e in period_errors if e.recovery_successful]),
                "recovery_rate": _calculate_recovery_rate(period_errors)
            }
        
        # Add component health checks
        component_health = await _check_component_health()
        
        return {
            "status": "success",
            "timestamp": now.isoformat(),
            "metrics_by_period": metrics,
            "component_health": component_health,
            "overall_health": health_monitor.health_metrics
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@health_router.get("/recovery-status")
async def get_recovery_status():
    """
    Get detailed recovery mechanism status and effectiveness.
    
    Returns:
        - Recovery action success rates
        - Most common failure patterns
        - Recovery recommendations
    """
    try:
        health_monitor = rag_error_handler.health_monitor
        
        # Analyze recovery patterns
        recovery_stats = {}
        for error in health_monitor.error_history:
            if error.recovery_attempted:
                category = error.category.value
                if category not in recovery_stats:
                    recovery_stats[category] = {
                        "total_attempts": 0,
                        "successful_recoveries": 0,
                        "failed_recoveries": 0,
                        "recovery_actions_used": {},
                        "common_errors": {}
                    }
                
                recovery_stats[category]["total_attempts"] += 1
                
                if error.recovery_successful:
                    recovery_stats[category]["successful_recoveries"] += 1
                else:
                    recovery_stats[category]["failed_recoveries"] += 1
                
                # Track recovery actions
                for action in error.recovery_actions:
                    if action not in recovery_stats[category]["recovery_actions_used"]:
                        recovery_stats[category]["recovery_actions_used"][action] = 0
                    recovery_stats[category]["recovery_actions_used"][action] += 1
                
                # Track common error types
                error_type = error.error_type
                if error_type not in recovery_stats[category]["common_errors"]:
                    recovery_stats[category]["common_errors"][error_type] = 0
                recovery_stats[category]["common_errors"][error_type] += 1
        
        # Calculate success rates
        for category_stats in recovery_stats.values():
            total = category_stats["total_attempts"]
            if total > 0:
                category_stats["success_rate"] = category_stats["successful_recoveries"] / total
            else:
                category_stats["success_rate"] = 0.0
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "recovery_statistics": recovery_stats,
            "overall_recovery_rate": rag_error_handler.health_monitor._calculate_recovery_rate(),
            "recommendations": _generate_recovery_recommendations(recovery_stats)
        }
    except Exception as e:
        logger.error(f"Error getting recovery status: {e}")
        raise HTTPException(status_code=500, detail=f"Recovery status retrieval failed: {str(e)}")


@health_router.post("/test-recovery/{category}")
async def test_recovery_mechanism(category: str, user_id: Optional[str] = None):
    """
    Test recovery mechanisms for a specific error category.
    
    This endpoint allows testing of recovery actions without waiting for actual errors.
    Useful for validating that recovery mechanisms work correctly.
    """
    try:
        from app.error_handler import ErrorCategory, ErrorContext
        
        # Validate category
        try:
            error_category = ErrorCategory(category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid error category: {category}")
        
        # Create test context
        test_context = ErrorContext(
            user_id=user_id,
            operation=f"test_recovery_{category}",
            additional_data={"test": True}
        )
        
        # Create a test error
        test_error = Exception(f"Test error for category {category}")
        
        # Attempt recovery
        recovery_successful, user_message, recovered_result = await rag_error_handler.handle_error(
            test_error, test_context, error_category
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "test_category": category,
            "recovery_successful": recovery_successful,
            "user_message": user_message,
            "recovered_result": str(recovered_result) if recovered_result else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing recovery mechanism: {e}")
        raise HTTPException(status_code=500, detail=f"Recovery test failed: {str(e)}")


@health_router.get("/alerts")
async def get_active_alerts():
    """
    Get currently active system alerts based on error thresholds.
    
    Returns:
        - Active alerts with severity levels
        - Alert conditions and thresholds
        - Recommended actions
    """
    try:
        health_monitor = rag_error_handler.health_monitor
        now = datetime.now()
        recent_errors = [
            error for error in health_monitor.error_history
            if now - error.timestamp < timedelta(hours=1)
        ]
        
        alerts = []
        thresholds = health_monitor.alert_thresholds
        
        # Check critical error threshold
        critical_errors = [e for e in recent_errors if e.severity.value == "critical"]
        if len(critical_errors) >= thresholds["critical_errors_per_hour"]:
            alerts.append({
                "severity": "critical",
                "type": "high_critical_error_rate",
                "message": f"{len(critical_errors)} critical errors in the last hour",
                "threshold": thresholds["critical_errors_per_hour"],
                "current_value": len(critical_errors),
                "recommended_action": "Immediate investigation required"
            })
        
        # Check high error threshold
        high_errors = [e for e in recent_errors if e.severity.value == "high"]
        if len(high_errors) >= thresholds["high_errors_per_hour"]:
            alerts.append({
                "severity": "high",
                "type": "high_error_rate",
                "message": f"{len(high_errors)} high-severity errors in the last hour",
                "threshold": thresholds["high_errors_per_hour"],
                "current_value": len(high_errors),
                "recommended_action": "Review error patterns and consider scaling"
            })
        
        # Check total error threshold
        if len(recent_errors) >= thresholds["total_errors_per_hour"]:
            alerts.append({
                "severity": "medium",
                "type": "high_total_error_rate",
                "message": f"{len(recent_errors)} total errors in the last hour",
                "threshold": thresholds["total_errors_per_hour"],
                "current_value": len(recent_errors),
                "recommended_action": "Monitor system load and performance"
            })
        
        # Check recovery failure rate
        recovery_rate = health_monitor._calculate_recovery_rate()
        if recovery_rate < (1 - thresholds["recovery_failure_rate"]):
            alerts.append({
                "severity": "medium",
                "type": "low_recovery_success_rate",
                "message": f"Recovery success rate is {recovery_rate:.2%}",
                "threshold": f"{(1 - thresholds['recovery_failure_rate']):.2%}",
                "current_value": f"{recovery_rate:.2%}",
                "recommended_action": "Review and improve recovery mechanisms"
            })
        
        return {
            "status": "success",
            "timestamp": now.isoformat(),
            "active_alerts": alerts,
            "alert_count": len(alerts),
            "system_status": health_monitor.health_metrics["system_status"]
        }
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")


# Debug endpoints for RAG system testing and validation
@health_router.get("/debug/user-index/{user_id}")
async def debug_user_index_status(user_id: str):
    """
    Get detailed information about a user's index status and components.
    
    Returns:
        - Index existence and health
        - Document count and sample documents
        - Retriever availability
        - ChromaDB collection status
    """
    try:
        import app.rag as rag_module
        
        debug_info = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "index_status": {},
            "retriever_status": {},
            "chromadb_status": {},
            "document_info": {}
        }
        
        # Check if user index exists in cache
        user_index = rag_module.user_indexes.get(user_id)
        debug_info["index_status"]["cached"] = user_index is not None
        
        if user_index:
            try:
                # Get document count from index
                doc_count = len(user_index.docstore.docs) if hasattr(user_index.docstore, 'docs') else 0
                debug_info["index_status"]["document_count"] = doc_count
                debug_info["index_status"]["health"] = "healthy"
                
                # Sample a few document IDs
                if hasattr(user_index.docstore, 'docs') and user_index.docstore.docs:
                    sample_doc_ids = list(user_index.docstore.docs.keys())[:5]
                    debug_info["document_info"]["sample_doc_ids"] = sample_doc_ids
                    
                    # Get sample document content
                    sample_docs = []
                    for doc_id in sample_doc_ids[:3]:
                        doc = user_index.docstore.docs[doc_id]
                        sample_docs.append({
                            "doc_id": doc_id,
                            "text_preview": doc.get_content()[:200] + "..." if len(doc.get_content()) > 200 else doc.get_content(),
                            "metadata": doc.metadata
                        })
                    debug_info["document_info"]["sample_documents"] = sample_docs
                
            except Exception as e:
                debug_info["index_status"]["health"] = "unhealthy"
                debug_info["index_status"]["error"] = str(e)
        else:
            debug_info["index_status"]["health"] = "not_found"
        
        # Check BM25 retriever
        user_bm25 = rag_module.user_bm25_retrievers.get(user_id)
        debug_info["retriever_status"]["bm25_available"] = user_bm25 is not None
        
        if user_bm25:
            try:
                # Try to get node count from BM25 retriever safely
                node_count = 0
                nodes = getattr(user_bm25, '_nodes', None)
                if nodes:
                    node_count = len(nodes)
                elif hasattr(user_bm25, 'docstore'):
                    docstore = getattr(user_bm25, 'docstore', None)
                    if docstore and hasattr(docstore, 'docs'):
                        node_count = len(docstore.docs)
                
                debug_info["retriever_status"]["bm25_node_count"] = node_count
                debug_info["retriever_status"]["bm25_health"] = "healthy"
            except Exception as e:
                debug_info["retriever_status"]["bm25_health"] = "unhealthy"
                debug_info["retriever_status"]["bm25_error"] = str(e)
        
        # Check fusion retriever creation
        try:
            fusion_retriever = rag_module.get_fusion_retriever(user_id)
            debug_info["retriever_status"]["fusion_available"] = fusion_retriever is not None
            if fusion_retriever:
                debug_info["retriever_status"]["fusion_health"] = "healthy"
            else:
                debug_info["retriever_status"]["fusion_health"] = "creation_failed"
        except Exception as e:
            debug_info["retriever_status"]["fusion_health"] = "error"
            debug_info["retriever_status"]["fusion_error"] = str(e)
        
        # Check ChromaDB collection
        try:
            collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
            if collection:
                doc_count = await asyncio.to_thread(collection.count)
                debug_info["chromadb_status"]["collection_exists"] = True
                debug_info["chromadb_status"]["document_count"] = doc_count
                debug_info["chromadb_status"]["health"] = "healthy"
                
                # Get sample documents from ChromaDB
                if doc_count > 0:
                    try:
                        sample_results = await asyncio.to_thread(collection.peek, limit=3)
                        debug_info["chromadb_status"]["sample_documents"] = {
                            "ids": sample_results.get("ids", []),
                            "metadatas": sample_results.get("metadatas", []),
                            "documents": [doc[:200] + "..." if doc and len(doc) > 200 else doc 
                                        for doc in (sample_results.get("documents", []) or [])]
                        }
                    except Exception as peek_error:
                        debug_info["chromadb_status"]["sample_error"] = str(peek_error)
            else:
                debug_info["chromadb_status"]["collection_exists"] = False
                debug_info["chromadb_status"]["health"] = "not_found"
        except Exception as e:
            debug_info["chromadb_status"]["health"] = "error"
            debug_info["chromadb_status"]["error"] = str(e)
        
        # Overall status assessment
        overall_healthy = (
            debug_info["index_status"].get("health") == "healthy" and
            debug_info["retriever_status"].get("fusion_health") == "healthy" and
            debug_info["chromadb_status"].get("health") == "healthy"
        )
        debug_info["overall_status"] = "healthy" if overall_healthy else "issues_detected"
        
        return {
            "status": "success",
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error in debug_user_index_status: {e}")
        raise HTTPException(status_code=500, detail=f"Debug operation failed: {str(e)}")


@health_router.post("/debug/test-search/{user_id}")
async def debug_test_search(user_id: str, query: str, strategy: Optional[str] = None):
    """
    Test search functionality with detailed results and performance metrics.
    
    Args:
        user_id: User ID to test search for
        query: Search query to test
        strategy: Optional specific strategy to test ('exact', 'multi', 'fusion')
    
    Returns:
        Detailed search results with performance metrics and debugging info
    """
    try:
        import app.rag as rag_module
        from app.multi_strategy_search import multi_strategy_search
        import time
        
        start_time = time.time()
        
        test_result = {
            "user_id": user_id,
            "query": query,
            "strategy_requested": strategy,
            "timestamp": datetime.now().isoformat(),
            "results": {},
            "performance": {},
            "debug_info": {}
        }
        
        # Check if user has index and retrievers
        user_index = rag_module.user_indexes.get(user_id)
        user_bm25 = rag_module.user_bm25_retrievers.get(user_id)
        
        test_result["debug_info"]["user_index_available"] = user_index is not None
        test_result["debug_info"]["user_bm25_available"] = user_bm25 is not None
        
        if not user_index:
            test_result["results"]["error"] = "User index not available"
            test_result["debug_info"]["suggestion"] = "Build user index first"
            return {"status": "error", "test_result": test_result}
        
        # Test fusion retriever creation
        fusion_start = time.time()
        fusion_retriever = rag_module.get_fusion_retriever(user_id)
        fusion_time = (time.time() - fusion_start) * 1000
        
        test_result["debug_info"]["fusion_retriever_available"] = fusion_retriever is not None
        test_result["performance"]["fusion_creation_time_ms"] = fusion_time
        
        if not fusion_retriever:
            test_result["results"]["error"] = "Fusion retriever creation failed"
            return {"status": "error", "test_result": test_result}
        
        # Perform different types of searches based on strategy
        if strategy == "exact" or strategy is None:
            # Test exact fusion retriever search
            exact_start = time.time()
            try:
                exact_nodes = await fusion_retriever.aretrieve(query)
                exact_time = (time.time() - exact_start) * 1000
                
                test_result["results"]["exact_search"] = {
                    "nodes_found": len(exact_nodes),
                    "execution_time_ms": exact_time,
                    "success": True,
                    "nodes": [
                        {
                            "score": node.score,
                            "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                            "metadata": node.metadata
                        }
                        for node in exact_nodes[:5]  # Limit to first 5 for readability
                    ]
                }
            except Exception as e:
                test_result["results"]["exact_search"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time_ms": (time.time() - exact_start) * 1000
                }
        
        if strategy == "multi" or strategy is None:
            # Test multi-strategy search
            multi_start = time.time()
            try:
                multi_result = await multi_strategy_search(fusion_retriever, query)
                multi_time = (time.time() - multi_start) * 1000
                
                test_result["results"]["multi_strategy_search"] = {
                    "total_nodes_found": len(multi_result.nodes_found),
                    "strategies_tried": len(multi_result.all_results),
                    "execution_time_ms": multi_time,
                    "best_strategy": multi_result.best_result.strategy if multi_result.best_result else None,
                    "success": True,
                    "strategy_details": [
                        {
                            "strategy": result.strategy,
                            "query_used": result.query_used,
                            "nodes_found": len(result.nodes),
                            "success": result.success,
                            "execution_time_ms": result.execution_time_ms
                        }
                        for result in multi_result.all_results
                    ],
                    "best_nodes": [
                        {
                            "score": node.score,
                            "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                            "metadata": node.metadata
                        }
                        for node in multi_result.nodes_found[:5]  # Limit to first 5
                    ]
                }
            except Exception as e:
                test_result["results"]["multi_strategy_search"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time_ms": (time.time() - multi_start) * 1000
                }
        
        # Test individual retrievers if available
        if user_bm25:
            bm25_start = time.time()
            try:
                bm25_nodes = user_bm25.retrieve(query)
                bm25_time = (time.time() - bm25_start) * 1000
                
                test_result["results"]["bm25_search"] = {
                    "nodes_found": len(bm25_nodes),
                    "execution_time_ms": bm25_time,
                    "success": True,
                    "nodes": [
                        {
                            "score": node.score,
                            "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
                        }
                        for node in bm25_nodes[:3]
                    ]
                }
            except Exception as e:
                test_result["results"]["bm25_search"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time_ms": (time.time() - bm25_start) * 1000
                }
        
        # Test vector search
        vector_start = time.time()
        try:
            vector_retriever = user_index.as_retriever(similarity_top_k=5)
            vector_nodes = await vector_retriever.aretrieve(query)
            vector_time = (time.time() - vector_start) * 1000
            
            test_result["results"]["vector_search"] = {
                "nodes_found": len(vector_nodes),
                "execution_time_ms": vector_time,
                "success": True,
                "nodes": [
                    {
                        "score": node.score,
                        "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    }
                    for node in vector_nodes[:3]
                ]
            }
        except Exception as e:
            test_result["results"]["vector_search"] = {
                "success": False,
                "error": str(e),
                "execution_time_ms": (time.time() - vector_start) * 1000
            }
        
        # Overall performance summary
        total_time = (time.time() - start_time) * 1000
        test_result["performance"]["total_test_time_ms"] = total_time
        
        # Success assessment
        successful_searches = sum(1 for result in test_result["results"].values() 
                                if isinstance(result, dict) and result.get("success", False))
        total_searches = len([r for r in test_result["results"].values() if isinstance(r, dict)])
        
        test_result["overall_success"] = successful_searches > 0
        test_result["success_rate"] = successful_searches / total_searches if total_searches > 0 else 0
        
        return {
            "status": "success",
            "test_result": test_result
        }
        
    except Exception as e:
        logger.error(f"Error in debug_test_search: {e}")
        raise HTTPException(status_code=500, detail=f"Search test failed: {str(e)}")


@health_router.get("/debug/validate-documents/{user_id}")
async def debug_validate_user_documents(user_id: str):
    """
    Validate user document processing and file availability.
    
    Returns:
        - File system status for user documents
        - Document processing validation
        - File format and content analysis
    """
    try:
        import os
        import pandas as pd
        
        validation_result = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "file_system_status": {},
            "document_analysis": {},
            "processing_validation": {}
        }
        
        # Check user documents directory
        user_docs_path = f"user_documents/{user_id}"
        validation_result["file_system_status"]["user_directory"] = user_docs_path
        validation_result["file_system_status"]["directory_exists"] = os.path.exists(user_docs_path)
        
        if os.path.exists(user_docs_path):
            # List files in user directory
            try:
                files = os.listdir(user_docs_path)
                csv_files = [f for f in files if f.endswith('.csv')]
                
                validation_result["file_system_status"]["total_files"] = len(files)
                validation_result["file_system_status"]["csv_files"] = len(csv_files)
                validation_result["file_system_status"]["file_list"] = files
                validation_result["file_system_status"]["csv_file_list"] = csv_files
                
                # Analyze CSV files
                document_details = []
                for csv_file in csv_files:
                    file_path = os.path.join(user_docs_path, csv_file)
                    try:
                        df = pd.read_csv(file_path)
                        file_info = {
                            "filename": csv_file,
                            "file_size_bytes": os.path.getsize(file_path),
                            "rows": len(df),
                            "columns": len(df.columns),
                            "column_names": df.columns.tolist(),
                            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
                            "processing_status": "readable"
                        }
                        
                        # Check for address-like columns
                        address_columns = [col for col in df.columns 
                                         if any(addr_term in col.lower() 
                                               for addr_term in ['address', 'street', 'location', 'property'])]
                        file_info["address_columns"] = address_columns
                        
                        # Check for specific test data (84 Mulberry St)
                        test_address_found = False
                        for col in df.columns:
                            if df[col].astype(str).str.contains("84 Mulberry", case=False, na=False).any():
                                test_address_found = True
                                file_info["contains_test_address"] = True
                                # Get the specific row
                                matching_rows = df[df[col].astype(str).str.contains("84 Mulberry", case=False, na=False)]
                                file_info["test_address_data"] = [row.to_dict() for _, row in matching_rows.iterrows()]
                                break
                        
                        if not test_address_found:
                            file_info["contains_test_address"] = False
                        
                        document_details.append(file_info)
                        
                    except Exception as e:
                        document_details.append({
                            "filename": csv_file,
                            "processing_status": "error",
                            "error": str(e)
                        })
                
                validation_result["document_analysis"]["files"] = document_details
                
            except Exception as e:
                validation_result["file_system_status"]["directory_read_error"] = str(e)
        
        # Test document processing pipeline
        try:
            import app.rag as rag_module
            
            # Check if user already has processed documents
            user_index = rag_module.user_indexes.get(user_id)
            validation_result["processing_validation"]["index_exists"] = user_index is not None
            
            if user_index and hasattr(user_index.docstore, 'docs'):
                doc_count = len(user_index.docstore.docs)
                validation_result["processing_validation"]["processed_document_count"] = doc_count
                
                # Sample processed documents
                if doc_count > 0:
                    sample_docs = []
                    for doc_id, doc in list(user_index.docstore.docs.items())[:3]:
                        sample_docs.append({
                            "doc_id": doc_id,
                            "text_preview": doc.get_content()[:300] + "..." if len(doc.get_content()) > 300 else doc.get_content(),
                            "metadata": doc.metadata
                        })
                    validation_result["processing_validation"]["sample_processed_docs"] = sample_docs
            
            # Test if we can build index from current files
            if os.path.exists(user_docs_path):
                csv_files = [f for f in os.listdir(user_docs_path) if f.endswith('.csv')]
                if csv_files:
                    file_paths = [os.path.join(user_docs_path, f) for f in csv_files]
                    validation_result["processing_validation"]["files_ready_for_processing"] = True
                    validation_result["processing_validation"]["processable_files"] = file_paths
                else:
                    validation_result["processing_validation"]["files_ready_for_processing"] = False
                    validation_result["processing_validation"]["reason"] = "No CSV files found"
            else:
                validation_result["processing_validation"]["files_ready_for_processing"] = False
                validation_result["processing_validation"]["reason"] = "User directory does not exist"
                
        except Exception as e:
            validation_result["processing_validation"]["error"] = str(e)
        
        # Overall validation status
        has_files = validation_result["file_system_status"].get("csv_files", 0) > 0
        can_process = validation_result["processing_validation"].get("files_ready_for_processing", False)
        has_test_data = any(
            file_info.get("contains_test_address", False) 
            for file_info in validation_result["document_analysis"].get("files", [])
        )
        
        validation_result["overall_status"] = {
            "has_files": has_files,
            "can_process": can_process,
            "has_test_data": has_test_data,
            "ready_for_rag": has_files and can_process
        }
        
        return {
            "status": "success",
            "validation_result": validation_result
        }
        
    except Exception as e:
        logger.error(f"Error in debug_validate_user_documents: {e}")
        raise HTTPException(status_code=500, detail=f"Document validation failed: {str(e)}")


@health_router.post("/debug/rebuild-index/{user_id}")
async def debug_force_index_rebuild(user_id: str, clear_existing: bool = True):
    """
    Force rebuild of user index with progress tracking.
    
    Args:
        user_id: User ID to rebuild index for
        clear_existing: Whether to clear existing index first (default: True)
    
    Returns:
        Progress information and rebuild results
    """
    try:
        import app.rag as rag_module
        import os
        import time
        
        rebuild_result = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "clear_existing": clear_existing,
            "progress": [],
            "final_status": {}
        }
        
        start_time = time.time()
        
        # Step 1: Clear existing index if requested
        if clear_existing:
            rebuild_result["progress"].append({
                "step": "clear_existing_index",
                "timestamp": datetime.now().isoformat(),
                "status": "starting"
            })
            
            try:
                success = await rag_module.clear_user_index(user_id)
                rebuild_result["progress"].append({
                    "step": "clear_existing_index",
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed" if success else "failed",
                    "details": f"Clear operation {'succeeded' if success else 'failed'}"
                })
            except Exception as e:
                rebuild_result["progress"].append({
                    "step": "clear_existing_index",
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e)
                })
        
        # Step 2: Find user documents
        rebuild_result["progress"].append({
            "step": "find_documents",
            "timestamp": datetime.now().isoformat(),
            "status": "starting"
        })
        
        user_docs_path = f"user_documents/{user_id}"
        if not os.path.exists(user_docs_path):
            rebuild_result["progress"].append({
                "step": "find_documents",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": f"User documents directory not found: {user_docs_path}"
            })
            rebuild_result["final_status"]["success"] = False
            return {"status": "error", "rebuild_result": rebuild_result}
        
        csv_files = [f for f in os.listdir(user_docs_path) if f.endswith('.csv')]
        if not csv_files:
            rebuild_result["progress"].append({
                "step": "find_documents",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": "No CSV files found in user directory"
            })
            rebuild_result["final_status"]["success"] = False
            return {"status": "error", "rebuild_result": rebuild_result}
        
        file_paths = [os.path.join(user_docs_path, f) for f in csv_files]
        rebuild_result["progress"].append({
            "step": "find_documents",
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "details": f"Found {len(csv_files)} CSV files: {csv_files}"
        })
        
        # Step 3: Build new index
        rebuild_result["progress"].append({
            "step": "build_index",
            "timestamp": datetime.now().isoformat(),
            "status": "starting",
            "details": f"Building index from {len(file_paths)} files"
        })
        
        try:
            build_start = time.time()
            new_index = await rag_module.build_user_index(user_id, file_paths)
            build_time = time.time() - build_start
            
            if new_index:
                doc_count = len(new_index.docstore.docs) if hasattr(new_index.docstore, 'docs') else 0
                rebuild_result["progress"].append({
                    "step": "build_index",
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "details": f"Index built successfully with {doc_count} documents in {build_time:.2f}s"
                })
            else:
                rebuild_result["progress"].append({
                    "step": "build_index",
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": "Index building returned None"
                })
                rebuild_result["final_status"]["success"] = False
                return {"status": "error", "rebuild_result": rebuild_result}
                
        except Exception as e:
            rebuild_result["progress"].append({
                "step": "build_index",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            })
            rebuild_result["final_status"]["success"] = False
            return {"status": "error", "rebuild_result": rebuild_result}
        
        # Step 4: Validate new index
        rebuild_result["progress"].append({
            "step": "validate_index",
            "timestamp": datetime.now().isoformat(),
            "status": "starting"
        })
        
        try:
            # Test fusion retriever creation
            fusion_retriever = rag_module.get_fusion_retriever(user_id)
            fusion_available = fusion_retriever is not None
            
            # Test a simple search
            test_query = "test query"
            search_results = None
            if fusion_retriever:
                try:
                    search_results = await fusion_retriever.aretrieve(test_query)
                except Exception as search_error:
                    logger.warning(f"Test search failed: {search_error}")
            
            rebuild_result["progress"].append({
                "step": "validate_index",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "details": {
                    "fusion_retriever_available": fusion_available,
                    "test_search_successful": search_results is not None,
                    "test_search_results": len(search_results) if search_results else 0
                }
            })
            
        except Exception as e:
            rebuild_result["progress"].append({
                "step": "validate_index",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            })
        
        # Final status
        total_time = time.time() - start_time
        rebuild_result["final_status"] = {
            "success": True,
            "total_time_seconds": total_time,
            "files_processed": len(file_paths),
            "documents_indexed": doc_count if 'doc_count' in locals() else 0,
            "fusion_retriever_available": fusion_available if 'fusion_available' in locals() else False
        }
        
        return {
            "status": "success",
            "rebuild_result": rebuild_result
        }
        
    except Exception as e:
        logger.error(f"Error in debug_force_index_rebuild: {e}")
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")


# Helper functions
async def _perform_system_checks() -> Dict[str, Any]:
    """Perform additional system health checks."""
    checks = {}
    
    # ChromaDB connectivity check
    try:
        client = chroma_manager.get_client()
        checks["chromadb"] = {
            "status": "healthy",
            "message": "ChromaDB client connected successfully"
        }
    except Exception as e:
        checks["chromadb"] = {
            "status": "unhealthy",
            "message": f"ChromaDB connection failed: {str(e)}"
        }
    
    # RAG system check
    try:
        # Check if we can create a basic index
        test_index = len(rag_module.user_indexes)
        checks["rag_system"] = {
            "status": "healthy",
            "message": f"RAG system operational with {test_index} cached indexes"
        }
    except Exception as e:
        checks["rag_system"] = {
            "status": "unhealthy",
            "message": f"RAG system check failed: {str(e)}"
        }
    
    return checks


async def _check_component_health() -> Dict[str, Any]:
    """Check health of individual system components."""
    components = {}
    
    # Check ChromaDB
    try:
        client = chroma_manager.get_client()
        components["chromadb"] = {"status": "healthy", "details": "Connected"}
    except Exception as e:
        components["chromadb"] = {"status": "unhealthy", "details": str(e)}
    
    # Check RAG indexes
    try:
        index_count = len(rag_module.user_indexes)
        retriever_count = len(rag_module.user_bm25_retrievers)
        components["rag_indexes"] = {
            "status": "healthy",
            "details": f"{index_count} indexes, {retriever_count} retrievers"
        }
    except Exception as e:
        components["rag_indexes"] = {"status": "unhealthy", "details": str(e)}
    
    return components


def _count_by_attribute(errors: List, attribute: str) -> Dict[str, int]:
    """Count errors by a specific attribute."""
    counts = {}
    for error in errors:
        value = getattr(error, attribute).value
        counts[value] = counts.get(value, 0) + 1
    return counts


def _calculate_recovery_rate(errors: List) -> float:
    """Calculate recovery success rate for a list of errors."""
    recovery_attempts = [e for e in errors if e.recovery_attempted]
    if not recovery_attempts:
        return 1.0
    
    successful = len([e for e in recovery_attempts if e.recovery_successful])
    return successful / len(recovery_attempts)


def _generate_recovery_recommendations(recovery_stats: Dict) -> List[str]:
    """Generate recommendations based on recovery statistics."""
    recommendations = []
    
    for category, stats in recovery_stats.items():
        success_rate = stats.get("success_rate", 0.0)
        
        if success_rate < 0.5:
            recommendations.append(
                f"Low recovery success rate ({success_rate:.2%}) for {category} errors. "
                f"Consider improving recovery mechanisms."
            )
        
        if stats.get("total_attempts", 0) > 10:
            recommendations.append(
                f"High number of recovery attempts ({stats['total_attempts']}) for {category}. "
                f"Consider addressing root causes."
            )
    
    if not recommendations:
        recommendations.append("Recovery mechanisms are performing well. No immediate action needed.")
    
    return recommendations


# Performance monitoring endpoints
@health_router.get("/performance/dashboard")
async def get_performance_dashboard():
    """
    Get comprehensive performance dashboard with metrics and optimization opportunities.
    
    Returns:
        - Overall system performance overview
        - Performance metrics by operation type
        - Recent performance alerts
        - Optimization recommendations
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        dashboard = performance_monitor.get_performance_dashboard()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "performance_dashboard": dashboard
        }
    except Exception as e:
        logger.error(f"Error getting performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Performance dashboard failed: {str(e)}")


@health_router.get("/performance/metrics/{operation_type}")
async def get_operation_performance_metrics(operation_type: str):
    """
    Get detailed performance metrics for a specific operation type.
    
    Args:
        operation_type: Type of operation (index_building, search_operation, etc.)
    
    Returns:
        Detailed performance statistics for the specified operation
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Validate operation type
        try:
            op_type = OperationType(operation_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid operation type: {operation_type}")
        
        stats = performance_monitor.get_operation_performance(op_type)
        
        if not stats:
            return {
                "status": "success",
                "operation_type": operation_type,
                "message": "No performance data available for this operation type"
            }
        
        return {
            "status": "success",
            "operation_type": operation_type,
            "performance_stats": {
                "total_operations": stats.total_operations,
                "successful_operations": stats.successful_operations,
                "failed_operations": stats.failed_operations,
                "success_rate": stats.success_rate,
                "avg_duration_ms": stats.avg_duration_ms,
                "min_duration_ms": stats.min_duration_ms,
                "max_duration_ms": stats.max_duration_ms,
                "median_duration_ms": stats.median_duration_ms,
                "p95_duration_ms": stats.p95_duration_ms,
                "p99_duration_ms": stats.p99_duration_ms,
                "performance_level": stats.performance_level.value,
                "last_updated": stats.last_updated.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting operation performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")


@health_router.get("/performance/alerts")
async def get_performance_alerts(hours: int = 24):
    """
    Get recent performance alerts and recommendations.
    
    Args:
        hours: Number of hours to look back for alerts (default: 24)
    
    Returns:
        Recent performance alerts with recommendations
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in (performance_monitor.alerts or [])
            if alert.timestamp >= cutoff_time
        ]
        
        formatted_alerts = []
        for alert in recent_alerts:
            formatted_alerts.append({
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": alert.alert_type,
                "operation_type": alert.operation_type.value,
                "message": alert.message,
                "severity": alert.severity,
                "metrics": alert.metrics,
                "recommendations": alert.recommendations
            })
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "time_range_hours": hours,
            "total_alerts": len(formatted_alerts),
            "alerts": formatted_alerts
        }
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Performance alerts failed: {str(e)}")


@health_router.get("/performance/optimization-opportunities")
async def get_optimization_opportunities():
    """
    Get current optimization opportunities based on performance analysis.
    
    Returns:
        List of optimization opportunities with priorities and recommendations
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        dashboard = performance_monitor.get_performance_dashboard()
        opportunities = dashboard.get("optimization_opportunities", [])
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_opportunities": len(opportunities),
            "optimization_opportunities": opportunities
        }
    except Exception as e:
        logger.error(f"Error getting optimization opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization opportunities failed: {str(e)}")


@health_router.post("/performance/export")
async def export_performance_metrics(hours: int = 24):
    """
    Export performance metrics for analysis.
    
    Args:
        hours: Number of hours of metrics to export (default: 24)
    
    Returns:
        Exported performance metrics in structured format
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        exported_data = performance_monitor.export_metrics(hours=hours)
        
        return {
            "status": "success",
            "export_data": exported_data
        }
    except Exception as e:
        logger.error(f"Error exporting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Performance export failed: {str(e)}")


@health_router.post("/performance/cleanup")
async def cleanup_performance_data(days: int = 7):
    """
    Clean up old performance metrics and alerts.
    
    Args:
        days: Number of days of data to keep (default: 7)
    
    Returns:
        Cleanup results
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Clean up old metrics
        performance_monitor.clear_old_metrics(days=days)
        
        # Clean up cache if available
        cache_cleaned = 0
        if hasattr(chroma_manager, 'cleanup_cache'):
            cache_cleaned = chroma_manager.cleanup_cache(max_age_minutes=days * 24 * 60)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cleanup_results": {
                "days_kept": days,
                "metrics_cleaned": "completed",
                "cache_entries_cleaned": cache_cleaned
            }
        }
    except Exception as e:
        logger.error(f"Error cleaning up performance data: {e}")
        raise HTTPException(status_code=500, detail=f"Performance cleanup failed: {str(e)}")


@health_router.get("/performance/system-health")
async def get_performance_system_health():
    """
    Get overall system health from performance perspective.
    
    Returns:
        System health assessment based on performance metrics
    """
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {
            "status": "success",
            "system_health": {
                "overall_status": "monitoring_unavailable",
                "message": "Performance monitoring is not available",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    try:
        dashboard = performance_monitor.get_performance_dashboard()
        
        # Get ChromaDB health if available
        chromadb_health = {}
        if hasattr(chroma_manager, 'get_health_status'):
            chromadb_health = chroma_manager.get_health_status()
        
        # Combine performance and system health
        system_health = {
            "overall_status": "healthy",
            "performance_overview": dashboard["overview"],
            "chromadb_health": chromadb_health,
            "active_alerts": len([
                alert for alert in (performance_monitor.alerts or [])
                if datetime.now() - alert.timestamp < timedelta(hours=1)
            ]),
            "optimization_opportunities": len(dashboard.get("optimization_opportunities", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine overall status
        if system_health["active_alerts"] > 5:
            system_health["overall_status"] = "degraded"
        elif system_health["performance_overview"]["success_rate_24h"] < 0.9:
            system_health["overall_status"] = "degraded"
        elif system_health["optimization_opportunities"] > 10:
            system_health["overall_status"] = "needs_optimization"
        
        return {
            "status": "success",
            "system_health": system_health
        }
    except Exception as e:
        logger.error(f"Error getting performance system health: {e}")
        raise HTTPException(status_code=500, detail=f"Performance system health failed: {str(e)}")


@health_router.get("/performance/chatbot-optimization")
async def get_chatbot_performance_optimization_status():
    """
    Get comprehensive status of chatbot performance optimization system.
    
    Returns:
        Status of fast classification, conversational handling, intent detection,
        index health validation, async operations, and performance monitoring
    """
    try:
        from app.fast_message_classifier import FastMessageClassifier
        from app.conversational_response_handler import ConversationalResponseHandler
        from app.enhanced_intent_detection import EnhancedIntentDetectionService
        from app.index_health_validator import IndexHealthValidator
        from app.async_index_manager import AsyncIndexManager
        from app.circuit_breaker import circuit_breaker_manager
        from app.performance_logger import performance_logger
        
        # Test fast classification performance
        classifier = FastMessageClassifier()
        test_start = time.time()
        test_classification = classifier.classify_message("hello test")
        classification_time_ms = (time.time() - test_start) * 1000
        
        # Get component statuses
        fast_classifier_stats = classifier.get_performance_stats()
        
        conversational_handler = ConversationalResponseHandler()
        handler_stats = conversational_handler.get_performance_stats()
        
        enhanced_intent_service = EnhancedIntentDetectionService()
        intent_error_stats = enhanced_intent_service.get_error_statistics()
        
        health_validator = IndexHealthValidator()
        validator_stats = health_validator.get_performance_stats()
        
        async_manager = AsyncIndexManager()
        async_stats = async_manager.get_performance_stats()
        
        circuit_breaker_health = circuit_breaker_manager.get_health_summary()
        
        performance_summary = performance_logger.get_performance_summary(hours=1)
        performance_alerts = []
        try:
            from app.performance_monitor import performance_monitor
            chatbot_summary = performance_monitor.get_chatbot_performance_summary()
            performance_alerts = performance_monitor.get_performance_alerts()
        except ImportError:
            chatbot_summary = {"message": "Enhanced performance monitoring not available"}
        
        # Calculate overall health score
        health_indicators = {
            "fast_classification_under_100ms": classification_time_ms < 100.0,
            "no_critical_circuit_breakers": circuit_breaker_health["open_circuits"] == 0,
            "no_active_rebuilds": async_stats["active_rebuilds"] == 0,
            "low_error_rate": len(performance_alerts) < 3,
            "intent_detection_working": len(intent_error_stats["llm_failures"]) < 5
        }
        
        health_score = sum(health_indicators.values()) / len(health_indicators) * 100
        
        overall_status = "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "critical"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "health_score": f"{health_score:.1f}%",
            "health_indicators": health_indicators,
            "components": {
                "fast_message_classifier": {
                    "status": "healthy" if classification_time_ms < 100 else "slow",
                    "last_test_time_ms": f"{classification_time_ms:.2f}",
                    "performance_stats": fast_classifier_stats
                },
                "conversational_handler": {
                    "status": "healthy",
                    "stats": handler_stats
                },
                "enhanced_intent_detection": {
                    "status": "healthy" if len(intent_error_stats["llm_failures"]) < 5 else "degraded",
                    "error_statistics": intent_error_stats
                },
                "index_health_validator": {
                    "status": "healthy",
                    "performance_stats": validator_stats
                },
                "async_index_manager": {
                    "status": "healthy" if async_stats["active_rebuilds"] == 0 else "busy",
                    "stats": async_stats
                },
                "circuit_breakers": {
                    "status": circuit_breaker_health["overall_health"],
                    "summary": circuit_breaker_health
                },
                "performance_monitoring": {
                    "status": "healthy" if len(performance_alerts) == 0 else "alerts",
                    "chatbot_summary": chatbot_summary,
                    "recent_alerts": performance_alerts,
                    "log_summary": performance_summary
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting chatbot optimization status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@health_router.get("/performance/classification-test")
async def test_message_classification_performance():
    """
    Test message classification performance with various message types.
    
    Returns:
        Performance results for different message classifications
    """
    try:
        from app.fast_message_classifier import FastMessageClassifier
        from app.enhanced_intent_detection import EnhancedIntentDetectionService
        
        classifier = FastMessageClassifier()
        enhanced_service = EnhancedIntentDetectionService()
        
        test_messages = [
            {"message": "hello there", "expected_type": "greeting"},
            {"message": "find me 2 bedroom apartments", "expected_type": "property_search"},
            {"message": "book an appointment", "expected_type": "appointment_request"},
            {"message": "thank you so much", "expected_type": "thank_you"},
            {"message": "what can you help me with", "expected_type": "help_request"},
            {"message": "how are you doing today", "expected_type": "conversational"},
            {"message": "random unclear message xyz", "expected_type": "unknown"}
        ]
        
        results = []
        
        for test_case in test_messages:
            message = test_case["message"]
            expected_type = test_case["expected_type"]
            
            # Test fast classification
            fast_start = time.time()
            fast_result = classifier.classify_message(message)
            fast_time_ms = (time.time() - fast_start) * 1000
            
            # Test enhanced classification
            enhanced_start = time.time()
            enhanced_result = await enhanced_service.detect_intent_with_fallback(message)
            enhanced_time_ms = (time.time() - enhanced_start) * 1000
            
            results.append({
                "message": message,
                "expected_type": expected_type,
                "fast_classification": {
                    "message_type": fast_result.message_type.value,
                    "confidence": fast_result.confidence,
                    "time_ms": f"{fast_time_ms:.2f}",
                    "strategy": fast_result.processing_strategy.value,
                    "meets_target": fast_time_ms < 100.0
                },
                "enhanced_classification": {
                    "message_type": enhanced_result.message_type.value,
                    "confidence": enhanced_result.confidence,
                    "time_ms": f"{enhanced_time_ms:.2f}",
                    "reasoning": enhanced_result.reasoning
                },
                "classification_matches": fast_result.message_type == enhanced_result.message_type
            })
        
        # Calculate summary stats
        fast_times = [float(r["fast_classification"]["time_ms"]) for r in results]
        enhanced_times = [float(r["enhanced_classification"]["time_ms"]) for r in results]
        matches = sum(1 for r in results if r["classification_matches"])
        fast_under_target = sum(1 for r in results if r["fast_classification"]["meets_target"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": len(results),
                "classification_agreement": f"{matches}/{len(results)} ({matches/len(results)*100:.1f}%)",
                "fast_under_100ms": f"{fast_under_target}/{len(results)} ({fast_under_target/len(results)*100:.1f}%)",
                "avg_fast_time_ms": f"{sum(fast_times)/len(fast_times):.2f}",
                "avg_enhanced_time_ms": f"{sum(enhanced_times)/len(enhanced_times):.2f}",
                "performance_target_met": fast_under_target == len(results)
            },
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"Error in classification performance test: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@health_router.get("/performance/index-health/{user_id}")
async def test_user_index_health(user_id: str):
    """
    Test index health for a specific user with detailed diagnostics.
    
    Args:
        user_id: User identifier to test
        
    Returns:
        Comprehensive index health report
    """
    try:
        from app.index_health_validator import IndexHealthValidator
        from app.async_index_manager import AsyncIndexManager
        
        validator = IndexHealthValidator()
        async_manager = AsyncIndexManager()
        
        # Perform health validation
        health_result = await validator.validate_user_index_health(user_id, force_refresh=True)
        
        # Check if rebuilding
        is_rebuilding = async_manager.is_user_rebuilding(user_id)
        rebuild_status = async_manager.get_rebuild_status(user_id) if is_rebuilding else None
        
        # Get user's rebuild history
        rebuild_history = async_manager.get_user_rebuild_history(user_id, limit=5)
        
        # Health check with auto-rebuild option
        auto_check_result = await async_manager.health_check_and_auto_rebuild(user_id)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "health_validation": {
                "is_healthy": health_result.is_healthy,
                "validation_time_ms": f"{health_result.validation_time:.2f}",
                "index_exists": health_result.index_exists,
                "retriever_functional": health_result.retriever_functional,
                "document_count": health_result.document_count,
                "issues_found": health_result.issues_found,
                "recommendations": health_result.recommendations
            },
            "rebuild_status": {
                "is_rebuilding": is_rebuilding,
                "current_operation": rebuild_status.status.value if rebuild_status else None,
                "progress_messages": rebuild_status.progress_messages if rebuild_status else [],
                "started_at": rebuild_status.started_at.isoformat() if rebuild_status else None
            },
            "rebuild_history": [
                {
                    "status": op.status.value,
                    "started_at": op.started_at.isoformat(),
                    "completed_at": op.completed_at.isoformat() if op.completed_at else None,
                    "document_count": op.document_count,
                    "error_message": op.error_message
                }
                for op in rebuild_history
            ],
            "auto_check_result": auto_check_result
        }
        
    except Exception as e:
        logger.error(f"Error testing index health for {user_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }


@health_router.get("/performance/circuit-breakers")
async def get_circuit_breaker_status():
    """
    Get status of all circuit breakers in the system.
    
    Returns:
        Detailed status of circuit breakers and their recent performance
    """
    try:
        from app.circuit_breaker import circuit_breaker_manager
        
        all_stats = circuit_breaker_manager.get_all_stats()
        health_summary = circuit_breaker_manager.get_health_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": health_summary,
            "circuit_breakers": all_stats,
            "recommendations": [
                "Monitor open circuits and investigate causes",
                "Check timeout configurations for frequently failing operations",
                "Consider adjusting failure thresholds if too sensitive",
                "Review half-open circuits for recovery patterns"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@health_router.post("/performance/circuit-breakers/reset")
async def reset_circuit_breakers():
    """Reset all circuit breakers to closed state."""
    try:
        from app.circuit_breaker import circuit_breaker_manager
        
        circuit_breaker_manager.reset_all()
        
        return {
            "status": "success",
            "message": "All circuit breakers reset to CLOSED state",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resetting circuit breakers: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@health_router.get("/performance/logs")
async def get_performance_logs(
    limit: int = 50,
    level: Optional[str] = None,
    hours: int = 1
):
    """
    Get recent performance logs with optional filtering.
    
    Args:
        limit: Maximum number of log entries to return
        level: Optional log level filter (debug, info, warning, error, critical)
        hours: Number of hours of history to include
        
    Returns:
        Recent performance log entries and summary
    """
    try:
        from app.performance_logger import performance_logger, LogLevel
        
        # Parse level filter
        level_filter = None
        if level:
            try:
                level_filter = LogLevel(level.lower())
            except ValueError:
                return {"error": f"Invalid log level: {level}. Valid levels: debug, info, warning, error, critical"}
        
        # Get logs
        recent_logs = performance_logger.get_recent_logs(limit=limit, level=level_filter)
        summary = performance_logger.get_performance_summary(hours=hours)
        error_analysis = performance_logger.get_error_analysis()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "filters": {
                "limit": limit,
                "level": level,
                "hours": hours
            },
            "summary": summary,
            "error_analysis": error_analysis,
            "recent_logs": recent_logs
        }
        
    except Exception as e:
        logger.error(f"Error getting performance logs: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@health_router.delete("/performance/logs")
async def clear_performance_logs(older_than_hours: Optional[int] = None):
    """
    Clear performance logs.
    
    Args:
        older_than_hours: Optional - only clear logs older than this many hours
        
    Returns:
        Number of log entries cleared
    """
    try:
        from app.performance_logger import performance_logger
        
        cleared_count = performance_logger.clear_logs(older_than_hours=older_than_hours)
        
        return {
            "status": "success",
            "cleared_count": cleared_count,
            "filter": f"older than {older_than_hours} hours" if older_than_hours else "all logs",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing performance logs: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }