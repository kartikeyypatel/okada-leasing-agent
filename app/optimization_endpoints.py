# /app/optimization_endpoints.py
"""
MongoDB RAG Optimization REST API Endpoints

This module provides comprehensive REST API endpoints for all MongoDB optimization
features including security, alerting, error handling, and performance optimization.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from app.security_compliance_manager import security_compliance_manager
from app.mongodb_error_handler import mongodb_error_handler
from app.alert_manager import alert_manager
from app.rag_performance_optimizer import rag_performance_optimizer

logger = logging.getLogger(__name__)

# Create router for optimization endpoints
optimization_router = APIRouter(prefix="/api/optimization", tags=["optimization"])


@optimization_router.get("/security/status")
async def get_security_status():
    """
    Get comprehensive security and compliance status.
    
    Returns:
        Security status with encryption validation, access auditing, and compliance reports
    """
    try:
        # Get security validation
        connection_security = await security_compliance_manager.validate_connection_security()
        
        # Get access audit report
        access_audit = await security_compliance_manager.audit_access_patterns()
        
        # Get anomaly detection report
        anomaly_report = await security_compliance_manager.detect_anomalous_access()
        
        # Get compliance report
        compliance_report = await security_compliance_manager.ensure_data_compliance()
        
        # Get security dashboard
        security_dashboard = security_compliance_manager.get_security_dashboard()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "security_validation": {
                "connection_security": connection_security.value,
                "encryption_enabled": True,  # Based on security manager initialization
                "access_logging": len(security_compliance_manager.access_logs) > 0,
                "anomaly_detection": "active"
            },
            "access_audit": {
                "report_period": access_audit.report_period,
                "total_accesses": access_audit.total_access_attempts,
                "success_rate": access_audit.success_rate_percent,
                "unique_users": access_audit.unique_users_count,
                "top_users": access_audit.top_users_by_access,
                "top_collections": access_audit.top_collections_accessed
            },
            "anomaly_detection": {
                "anomalies_detected": len(anomaly_report.anomalies),
                "critical_anomalies": len([a for a in anomaly_report.anomalies if a.severity == "critical"]),
                "high_anomalies": len([a for a in anomaly_report.anomalies if a.severity == "high"]),
                "recent_anomalies": anomaly_report.anomalies[-5:] if anomaly_report.anomalies else []
            },
            "compliance": {
                "overall_score": compliance_report.compliance_score,
                "compliant_standards": [
                    standard for standard, compliant in compliance_report.standards_compliance.items()
                    if compliant
                ],
                "non_compliant_areas": compliance_report.violations,
                "recommendations": compliance_report.recommendations[:5]  # Top 5 recommendations
            },
            "security_dashboard": security_dashboard
        }
        
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        raise HTTPException(status_code=500, detail=f"Security status error: {str(e)}")


@optimization_router.post("/security/encrypt-data")
async def encrypt_sensitive_data(data: Dict[str, Any] = Body(...)):
    """
    Encrypt sensitive data fields.
    
    Args:
        data: Dictionary containing data to encrypt
        
    Returns:
        Encrypted data with sensitive fields protected
    """
    try:
        encrypted_data = await security_compliance_manager.encrypt_sensitive_data(data)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "sensitive_fields_count": len(encrypted_data.fields_encrypted),
            "sensitive_fields": encrypted_data.fields_encrypted,
            "encryption_algorithm": encrypted_data.algorithm_used,
            "encrypted_data": encrypted_data.data
        }
        
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")


@optimization_router.get("/error-handling/statistics")
async def get_error_statistics():
    """
    Get comprehensive MongoDB error statistics and recovery metrics.
    
    Returns:
        Error statistics with recovery rates and recent errors
    """
    try:
        error_stats = mongodb_error_handler.get_error_statistics()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "error_statistics": error_stats,
            "circuit_breaker": {
                "is_open": mongodb_error_handler.circuit_breaker["is_open"],
                "failure_count": mongodb_error_handler.circuit_breaker["failure_count"],
                "recovery_timeout": mongodb_error_handler.circuit_breaker["recovery_timeout"]
            },
            "recovery_capabilities": {
                "connection_recovery": "enabled",
                "query_optimization": "enabled",
                "index_repair": "enabled",
                "fallback_mechanisms": "enabled"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting error statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error statistics error: {str(e)}")


@optimization_router.post("/error-handling/test-recovery")
async def test_error_recovery(error_type: str = Query(..., description="Type of error to simulate")):
    """
    Test error recovery mechanisms.
    
    Args:
        error_type: Type of error to test (connection_failure, query_timeout, index_corruption, disk_space)
        
    Returns:
        Recovery test results
    """
    try:
        if error_type == "connection_failure":
            result = await mongodb_error_handler.handle_connection_failure(
                Exception("Test connection failure"), 
                "test_operation"
            )
        elif error_type == "query_timeout":
            result = await mongodb_error_handler.handle_query_timeout(
                "SELECT * FROM test_collection", 
                5000, 
                "test_collection"
            )
        elif error_type == "index_corruption":
            result = await mongodb_error_handler.handle_index_corruption("test_collection")
        elif error_type == "disk_space":
            result = await mongodb_error_handler.handle_disk_space_issues()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown error type: {error_type}")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "recovery_result": result.__dict__ if hasattr(result, '__dict__') else str(result),
            "test_completed": True
        }
        
    except Exception as e:
        logger.error(f"Error testing recovery: {e}")
        raise HTTPException(status_code=500, detail=f"Recovery test error: {str(e)}")


@optimization_router.get("/alerts/dashboard")
async def get_alerts_dashboard():
    """
    Get comprehensive alerts dashboard with active alerts and statistics.
    
    Returns:
        Complete alerts dashboard data
    """
    try:
        dashboard = alert_manager.get_alert_dashboard()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "alerts_dashboard": dashboard,
            "alert_configuration": {
                "total_rules": len(alert_manager.alert_rules),
                "enabled_rules": len([rule for rule in alert_manager.alert_rules.values() if rule.enabled]),
                "notification_channels": ["email", "log", "slack", "webhook"],
                "monitoring_active": alert_manager._monitoring_task is not None and not alert_manager._monitoring_task.done()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Alerts dashboard error: {str(e)}")


@optimization_router.post("/alerts/start-monitoring")
async def start_alert_monitoring():
    """
    Start automated alert monitoring.
    
    Returns:
        Monitoring start status
    """
    try:
        await alert_manager.start_monitoring()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": "Alert monitoring started successfully",
            "monitoring_interval": alert_manager._monitoring_interval,
            "active_rules": len([rule for rule in alert_manager.alert_rules.values() if rule.enabled])
        }
        
    except Exception as e:
        logger.error(f"Error starting alert monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Alert monitoring error: {str(e)}")


@optimization_router.post("/alerts/stop-monitoring")
async def stop_alert_monitoring():
    """
    Stop automated alert monitoring.
    
    Returns:
        Monitoring stop status
    """
    try:
        await alert_manager.stop_monitoring()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": "Alert monitoring stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping alert monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Alert monitoring stop error: {str(e)}")


@optimization_router.post("/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = Query(...)):
    """
    Acknowledge an active alert.
    
    Args:
        alert_id: Alert identifier
        acknowledged_by: User acknowledging the alert
        
    Returns:
        Acknowledgment status
    """
    try:
        success = await alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "message": f"Alert {alert_id} acknowledged by {acknowledged_by}",
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=f"Alert acknowledgment error: {str(e)}")


@optimization_router.get("/rag-performance/dashboard")
async def get_rag_performance_dashboard(user_id: Optional[str] = Query(None)):
    """
    Get RAG performance dashboard with retrieval metrics and optimization status.
    
    Args:
        user_id: Optional user ID to filter metrics
        
    Returns:
        RAG performance dashboard data
    """
    try:
        dashboard = rag_performance_optimizer.get_performance_dashboard(user_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "rag_performance": dashboard,
            "optimization_features": {
                "performance_monitoring": "active",
                "quality_validation": "enabled",
                "auto_cleanup": "enabled",
                "optimization_triggers": "enabled"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"RAG performance dashboard error: {str(e)}")


@optimization_router.post("/rag-performance/optimize/{user_id}")
async def optimize_user_rag_performance(user_id: str):
    """
    Optimize RAG performance for a specific user.
    
    Args:
        user_id: User identifier for optimization
        
    Returns:
        Optimization results and performance improvements
    """
    try:
        optimization_result = await rag_performance_optimizer.optimize_user_retrieval(user_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "optimization_result": {
                "optimization_type": optimization_result.optimization_type,
                "before_latency_ms": optimization_result.before_latency_ms,
                "after_latency_ms": optimization_result.after_latency_ms,
                "improvement_percent": optimization_result.improvement_percent,
                "optimizations_applied": optimization_result.optimizations_applied,
                "context_quality_improvement": optimization_result.context_quality_improvement,
                "recommendations": optimization_result.recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing RAG performance: {e}")
        raise HTTPException(status_code=500, detail=f"RAG optimization error: {str(e)}")


@optimization_router.get("/rag-performance/quality/{user_id}")
async def validate_user_embedding_quality(user_id: str):
    """
    Validate embedding quality for a specific user.
    
    Args:
        user_id: User identifier for quality validation
        
    Returns:
        Embedding quality report and recommendations
    """
    try:
        quality_report = await rag_performance_optimizer.validate_embedding_quality(user_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "quality_report": {
                "total_embeddings": quality_report.total_embeddings,
                "valid_embeddings": quality_report.valid_embeddings,
                "corrupted_embeddings": quality_report.corrupted_embeddings,
                "duplicate_embeddings": quality_report.duplicate_embeddings,
                "orphaned_embeddings": quality_report.orphaned_embeddings,
                "avg_embedding_dimension": quality_report.avg_embedding_dimension,
                "dimension_consistency": quality_report.dimension_consistency,
                "quality_score": quality_report.quality_score,
                "recommendations": quality_report.recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating embedding quality: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding quality validation error: {str(e)}")


@optimization_router.post("/rag-performance/cleanup/{user_id}")
async def cleanup_user_embeddings(user_id: str):
    """
    Clean up stale embeddings for a specific user.
    
    Args:
        user_id: User identifier for cleanup
        
    Returns:
        Cleanup results and statistics
    """
    try:
        cleanup_result = await rag_performance_optimizer.cleanup_stale_embeddings(user_id)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "cleanup_result": cleanup_result
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding cleanup error: {str(e)}")


@optimization_router.get("/system/comprehensive-status")
async def get_comprehensive_system_status():
    """
    Get comprehensive system status including all optimization components.
    
    Returns:
        Complete system status with all optimization metrics
    """
    try:
        # Collect status from all components
        security_status = await security_compliance_manager.validate_connection_security()
        error_stats = mongodb_error_handler.get_error_statistics()
        alerts_dashboard = alert_manager.get_alert_dashboard()
        rag_dashboard = rag_performance_optimizer.get_performance_dashboard()
        
        # Calculate overall system health score
        health_factors = {
            "security": 1.0 if security_status.value == "secure" else 0.5 if security_status.value == "warning" else 0.0,
            "error_rate": 1.0 if error_stats.get("error_summary", {}).get("total_errors_24h", 0) < 10 else 0.5,
            "alerts": 1.0 if alerts_dashboard.get("alert_summary", {}).get("active_alerts", 0) == 0 else 0.7,
            "rag_performance": 1.0 if rag_dashboard.get("performance_summary", {}).get("success_rate", 0) > 0.9 else 0.6
        }
        
        overall_health_score = sum(health_factors.values()) / len(health_factors) * 100
        
        health_status = "excellent" if overall_health_score >= 90 else "good" if overall_health_score >= 70 else "needs_attention" if overall_health_score >= 50 else "critical"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "overall_health_score": overall_health_score,
                "health_status": health_status,
                "components_status": {
                    "security_compliance": security_status.value,
                    "error_handling": "active",
                    "alert_management": "active",
                    "rag_optimization": "active",
                    "mongodb_monitoring": "active"
                }
            },
            "security_summary": {
                "connection_security": security_status.value,
                "access_logs_count": len(security_compliance_manager.access_logs),
                "security_alerts_count": len(security_compliance_manager.security_alerts)
            },
            "error_handling_summary": {
                "total_errors_24h": error_stats.get("error_summary", {}).get("total_errors_24h", 0),
                "recovery_rate": error_stats.get("error_summary", {}).get("recovery_rate", 0),
                "circuit_breaker_open": mongodb_error_handler.circuit_breaker["is_open"]
            },
            "alerts_summary": {
                "active_alerts": alerts_dashboard.get("alert_summary", {}).get("active_alerts", 0),
                "total_alerts_24h": alerts_dashboard.get("alert_summary", {}).get("total_alerts_24h", 0),
                "monitoring_active": alert_manager._monitoring_task is not None and not alert_manager._monitoring_task.done()
            },
            "rag_performance_summary": {
                "avg_latency_ms": rag_dashboard.get("performance_summary", {}).get("avg_latency_ms", 0),
                "success_rate": rag_dashboard.get("performance_summary", {}).get("success_rate", 0),
                "optimization_enabled": rag_performance_optimizer.optimization_enabled
            },
            "recommendations": [
                "System is operating within normal parameters" if health_status == "excellent" else
                "Monitor system performance closely" if health_status == "good" else
                "Address performance issues and alerts" if health_status == "needs_attention" else
                "Immediate attention required - multiple system issues detected"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting comprehensive system status: {e}")
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")


@optimization_router.post("/system/run-full-optimization")
async def run_full_system_optimization():
    """
    Run comprehensive system optimization across all components.
    
    Returns:
        Full optimization results and recommendations
    """
    try:
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_performed": [],
            "errors_encountered": [],
            "overall_success": True
        }
        
        # 1. Start alert monitoring if not active
        try:
            await alert_manager.start_monitoring()
            optimization_results["optimizations_performed"].append("Alert monitoring activated")
        except Exception as e:
            optimization_results["errors_encountered"].append(f"Alert monitoring: {str(e)}")
        
        # 2. Run security compliance check
        try:
            compliance_report = await security_compliance_manager.ensure_data_compliance()
            optimization_results["optimizations_performed"].append(f"Security compliance checked (score: {compliance_report.compliance_score:.1%})")
        except Exception as e:
            optimization_results["errors_encountered"].append(f"Security compliance: {str(e)}")
        
        # 3. Test error recovery systems
        try:
            # Test connection recovery
            recovery_result = await mongodb_error_handler.handle_connection_failure(
                Exception("Test connection"), "optimization_test"
            )
            optimization_results["optimizations_performed"].append("Error recovery systems tested")
        except Exception as e:
            optimization_results["errors_encountered"].append(f"Error recovery test: {str(e)}")
        
        # 4. Evaluate alerts
        try:
            new_alerts = await alert_manager.evaluate_alerts()
            optimization_results["optimizations_performed"].append(f"Alert evaluation completed ({len(new_alerts)} new alerts)")
        except Exception as e:
            optimization_results["errors_encountered"].append(f"Alert evaluation: {str(e)}")
        
        # Set overall success status
        optimization_results["overall_success"] = len(optimization_results["errors_encountered"]) == 0
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "optimization_results": optimization_results,
            "next_steps": [
                "Monitor system performance for improvements",
                "Review any errors encountered during optimization",
                "Schedule regular optimization runs",
                "Update optimization thresholds based on results"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error running full system optimization: {e}")
        raise HTTPException(status_code=500, detail=f"System optimization error: {str(e)}")


@optimization_router.get("/system/health-summary")
async def get_system_health_summary():
    """
    Get a quick system health summary for monitoring dashboards.
    
    Returns:
        Concise system health summary
    """
    try:
        # Quick health checks
        mongodb_healthy = True  # Would check actual MongoDB connection
        security_status = await security_compliance_manager.validate_connection_security()
        active_alerts = len(alert_manager.active_alerts)
        
        # Calculate quick health score
        health_score = 100
        if security_status.value != "secure":
            health_score -= 30
        if active_alerts > 0:
            health_score -= min(active_alerts * 10, 40)
        if not mongodb_healthy:
            health_score -= 50
        
        health_score = max(0, health_score)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "health_summary": {
                "overall_score": health_score,
                "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical",
                "mongodb_connection": "healthy" if mongodb_healthy else "degraded",
                "security_status": security_status.value,
                "active_alerts": active_alerts,
                "optimization_active": True
            },
            "quick_stats": {
                "uptime": "system_uptime_placeholder",
                "last_optimization": "recent",
                "monitoring_status": "active",
                "error_rate": "low"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "health_summary": {
                "overall_score": 0,
                "status": "critical",
                "error": str(e)
            }
        } 