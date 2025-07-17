# /app/performance_monitor.py
"""
Performance Monitoring and Optimization System

This module provides comprehensive performance monitoring and optimization for the RAG chatbot:
1. Performance tracking for index building and search operations
2. Metrics collection for response times and success rates
3. Performance dashboards and alerting
4. Optimization of slow operations identified through monitoring

Requirements addressed: 2.4, 3.4
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from contextlib import asynccontextmanager
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations to monitor."""
    INDEX_BUILDING = "index_building"
    SEARCH_OPERATION = "search_operation"
    RESPONSE_GENERATION = "response_generation"
    DOCUMENT_PROCESSING = "document_processing"
    CHROMADB_OPERATION = "chromadb_operation"
    USER_CONTEXT_VALIDATION = "user_context_validation"
    MULTI_STRATEGY_SEARCH = "multi_strategy_search"
    STRICT_RESPONSE_GENERATION = "strict_response_generation"


class PerformanceLevel(Enum):
    """Performance level classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    SLOW = "slow"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric record."""
    timestamp: datetime
    operation_type: OperationType
    operation_name: str
    duration_ms: float
    success: bool
    user_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    performance_level: Optional[PerformanceLevel] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    operation_type: OperationType
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    median_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    success_rate: float
    performance_level: PerformanceLevel
    last_updated: datetime


@dataclass
class PerformanceAlert:
    """Performance alert for slow operations or degraded performance."""
    timestamp: datetime
    alert_type: str
    operation_type: OperationType
    message: str
    severity: str
    metrics: Dict[str, Any]
    recommendations: List[str]


class PerformanceThresholds:
    """Performance thresholds for different operations."""
    
    def __init__(self):
        self.thresholds = {
            OperationType.INDEX_BUILDING: {
                PerformanceLevel.EXCELLENT: 5000,    # < 5s
                PerformanceLevel.GOOD: 15000,        # < 15s
                PerformanceLevel.ACCEPTABLE: 30000,  # < 30s
                PerformanceLevel.SLOW: 60000,        # < 60s
                # > 60s = CRITICAL
            },
            OperationType.SEARCH_OPERATION: {
                PerformanceLevel.EXCELLENT: 500,     # < 0.5s
                PerformanceLevel.GOOD: 2000,         # < 2s
                PerformanceLevel.ACCEPTABLE: 5000,   # < 5s
                PerformanceLevel.SLOW: 10000,        # < 10s
                # > 10s = CRITICAL
            },
            OperationType.RESPONSE_GENERATION: {
                PerformanceLevel.EXCELLENT: 2000,    # < 2s
                PerformanceLevel.GOOD: 5000,         # < 5s
                PerformanceLevel.ACCEPTABLE: 10000,  # < 10s
                PerformanceLevel.SLOW: 20000,        # < 20s
                # > 20s = CRITICAL
            },
            OperationType.DOCUMENT_PROCESSING: {
                PerformanceLevel.EXCELLENT: 1000,    # < 1s
                PerformanceLevel.GOOD: 3000,         # < 3s
                PerformanceLevel.ACCEPTABLE: 8000,   # < 8s
                PerformanceLevel.SLOW: 15000,        # < 15s
                # > 15s = CRITICAL
            },
            OperationType.CHROMADB_OPERATION: {
                PerformanceLevel.EXCELLENT: 200,     # < 0.2s
                PerformanceLevel.GOOD: 1000,         # < 1s
                PerformanceLevel.ACCEPTABLE: 3000,   # < 3s
                PerformanceLevel.SLOW: 8000,         # < 8s
                # > 8s = CRITICAL
            },
            OperationType.USER_CONTEXT_VALIDATION: {
                PerformanceLevel.EXCELLENT: 100,     # < 0.1s
                PerformanceLevel.GOOD: 500,          # < 0.5s
                PerformanceLevel.ACCEPTABLE: 1500,   # < 1.5s
                PerformanceLevel.SLOW: 3000,         # < 3s
                # > 3s = CRITICAL
            },
            OperationType.MULTI_STRATEGY_SEARCH: {
                PerformanceLevel.EXCELLENT: 1000,    # < 1s
                PerformanceLevel.GOOD: 3000,         # < 3s
                PerformanceLevel.ACCEPTABLE: 8000,   # < 8s
                PerformanceLevel.SLOW: 15000,        # < 15s
                # > 15s = CRITICAL
            },
            OperationType.STRICT_RESPONSE_GENERATION: {
                PerformanceLevel.EXCELLENT: 3000,    # < 3s
                PerformanceLevel.GOOD: 8000,         # < 8s
                PerformanceLevel.ACCEPTABLE: 15000,  # < 15s
                PerformanceLevel.SLOW: 30000,        # < 30s
                # > 30s = CRITICAL
            }
        }
    
    def classify_performance(self, operation_type: OperationType, duration_ms: float) -> PerformanceLevel:
        """Classify performance level based on duration."""
        thresholds = self.thresholds.get(operation_type, {})
        
        if duration_ms <= thresholds.get(PerformanceLevel.EXCELLENT, 0):
            return PerformanceLevel.EXCELLENT
        elif duration_ms <= thresholds.get(PerformanceLevel.GOOD, 0):
            return PerformanceLevel.GOOD
        elif duration_ms <= thresholds.get(PerformanceLevel.ACCEPTABLE, 0):
            return PerformanceLevel.ACCEPTABLE
        elif duration_ms <= thresholds.get(PerformanceLevel.SLOW, 0):
            return PerformanceLevel.SLOW
        else:
            return PerformanceLevel.CRITICAL


class PerformanceMonitor:
    """Enhanced performance monitoring for chatbot operations."""
    
    def __init__(self, max_metrics_history: int = 10000):
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics_history = max_metrics_history
        self.thresholds = PerformanceThresholds()
        self.alerts: List[PerformanceAlert] = []
        self.performance_stats: Dict[OperationType, PerformanceStats] = {}
        self.operation_counters: Dict[str, int] = defaultdict(int)
        self.start_times: Dict[str, float] = {}
        
        # Performance monitoring settings
        self.alert_thresholds = {
            "slow_operation_count": 5,      # Alert if 5+ slow operations in window
            "critical_operation_count": 1,  # Alert immediately for critical operations
            "success_rate_threshold": 0.9,  # Alert if success rate < 90%
            "monitoring_window_minutes": 15  # Time window for alerts
        }
        
        # Chatbot-specific metrics
        self.message_type_times: Dict[str, List[float]] = defaultdict(list)
        self.slow_operations: List[PerformanceMetric] = []
        self.response_time_targets = {
            "greeting": 2000.0,        # 2 seconds
            "conversational": 2000.0,  # 2 seconds
            "help_request": 2000.0,    # 2 seconds
            "property_search": 5000.0, # 5 seconds
            "appointment": 3000.0,     # 3 seconds
            "index_building": 30000.0, # 30 seconds
            "fallback": 2000.0         # 2 seconds
        }
        
        # Performance thresholds
        self.SLOW_OPERATION_THRESHOLD = 10000.0  # 10 seconds
        self.CRITICAL_OPERATION_THRESHOLD = 30000.0  # 30 seconds
        
        # Stats tracking
        self.total_requests = 0
        self.fast_responses = 0  # Under target time
        self.slow_responses = 0  # Over target time
        self.critical_responses = 0  # Over critical threshold


    def record_metric(
        self,
        operation_type: OperationType,
        operation_name: str,
        duration_ms: float,
        success: bool,
        user_id: Optional[str] = None,
        **additional_data
    ):
        """Record a performance metric."""
        # Classify performance level
        performance_level = self.thresholds.classify_performance(operation_type, duration_ms)
        
        # Create metric record
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation_type=operation_type,
            operation_name=operation_name,
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            additional_data=additional_data,
            performance_level=performance_level
        )
        
        # Add to history
        self.metrics.append(metric)
        
        # Maintain history size limit
        if len(self.metrics) > self.max_metrics_history:
            self.metrics = self.metrics[-self.max_metrics_history:]
        
        # Update aggregated stats
        self._update_performance_stats(operation_type)
        
        # Check for alerts
        self._check_performance_alerts(metric)
        
        # Log performance issues
        if performance_level in [PerformanceLevel.SLOW, PerformanceLevel.CRITICAL]:
            logger.warning(
                f"Performance issue detected: {operation_name} took {duration_ms:.2f}ms "
                f"(level: {performance_level.value})"
            )
        
        logger.debug(
            f"Performance metric recorded: {operation_name} - {duration_ms:.2f}ms "
            f"(success: {success}, level: {performance_level.value})"
        )
    
    def record_message_response(self, message_type: str, duration_ms: float, 
                              success: bool, user_id: Optional[str] = None,
                              additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Record chatbot message response performance.
        
        Args:
            message_type: Type of message (greeting, property_search, etc.)
            duration_ms: Response time in milliseconds
            success: Whether the response was successful
            user_id: Optional user identifier
            additional_data: Additional performance data
        """
        self.total_requests += 1
        
        # Track by message type
        self.message_type_times[message_type].append(duration_ms)
        
        # Check against targets
        target_time = self.response_time_targets.get(message_type, 5000.0)
        
        if duration_ms <= target_time:
            self.fast_responses += 1
        elif duration_ms <= self.SLOW_OPERATION_THRESHOLD:
            self.slow_responses += 1
        else:
            self.critical_responses += 1
        
        # Create performance metric
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation_type=OperationType.RESPONSE_GENERATION, # Assuming RESPONSE_GENERATION for message responses
            operation_name=f"chatbot_response_{message_type}",
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            additional_data={"message_type": message_type, **(additional_data or {})},
            performance_level=self.thresholds.classify_performance(OperationType.RESPONSE_GENERATION, duration_ms)
        )
        
        # Store metric
        self.metrics.append(metric)
        
        # Maintain history size limit
        if len(self.metrics) > self.max_metrics_history:
            self.metrics = self.metrics[-self.max_metrics_history:]
        
        # Update aggregated stats
        self._update_performance_stats(OperationType.RESPONSE_GENERATION)
        
        # Check for alerts
        self._check_performance_alerts(metric)
        
        # Track slow operations
        if duration_ms > self.SLOW_OPERATION_THRESHOLD:
            self.slow_operations.append(metric)
            logger.warning(f"Slow {message_type} response: {duration_ms:.2f}ms for user {user_id}")
            
            # Keep only last 50 slow operations
            if len(self.slow_operations) > 50:
                self.slow_operations.pop(0)
    
    def _update_performance_stats(self, operation_type: OperationType):
        """Update aggregated performance statistics for an operation type."""
        # Get recent metrics for this operation type (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_metrics = [
            m for m in self.metrics
            if m.operation_type == operation_type and m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return
        
        # Calculate statistics
        durations = [m.duration_ms for m in recent_metrics]
        successful_ops = [m for m in recent_metrics if m.success]
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        n = len(sorted_durations)
        
        p95_index = int(0.95 * n)
        p99_index = int(0.99 * n)
        
        # Create performance stats
        stats = PerformanceStats(
            operation_type=operation_type,
            total_operations=len(recent_metrics),
            successful_operations=len(successful_ops),
            failed_operations=len(recent_metrics) - len(successful_ops),
            avg_duration_ms=statistics.mean(durations),
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            median_duration_ms=statistics.median(durations),
            p95_duration_ms=sorted_durations[p95_index] if p95_index < n else sorted_durations[-1],
            p99_duration_ms=sorted_durations[p99_index] if p99_index < n else sorted_durations[-1],
            success_rate=len(successful_ops) / len(recent_metrics),
            performance_level=self.thresholds.classify_performance(operation_type, statistics.mean(durations)),
            last_updated=datetime.now()
        )
        
        self.performance_stats[operation_type] = stats
    
    def _check_performance_alerts(self, metric: PerformanceMetric):
        """Check if a metric triggers any performance alerts."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.alert_thresholds["monitoring_window_minutes"])
        
        # Get recent metrics for this operation type
        recent_metrics = [
            m for m in self.metrics
            if (m.operation_type == metric.operation_type and 
                m.timestamp >= window_start)
        ]
        
        # Check for critical operation alert
        if metric.performance_level == PerformanceLevel.CRITICAL:
            alert = PerformanceAlert(
                timestamp=now,
                alert_type="critical_operation",
                operation_type=metric.operation_type,
                message=f"Critical performance detected: {metric.operation_name} took {metric.duration_ms:.2f}ms",
                severity="critical",
                metrics={
                    "duration_ms": metric.duration_ms,
                    "operation_name": metric.operation_name,
                    "user_id": metric.user_id
                },
                recommendations=self._generate_optimization_recommendations(metric.operation_type, metric.duration_ms)
            )
            self.alerts.append(alert)
            logger.critical(f"CRITICAL PERFORMANCE ALERT: {alert.message}")
        
        # Check for slow operation pattern alert
        slow_operations = [
            m for m in recent_metrics
            if m.performance_level in [PerformanceLevel.SLOW, PerformanceLevel.CRITICAL]
        ]
        
        if len(slow_operations) >= self.alert_thresholds["slow_operation_count"]:
            alert = PerformanceAlert(
                timestamp=now,
                alert_type="slow_operation_pattern",
                operation_type=metric.operation_type,
                message=f"High number of slow operations: {len(slow_operations)} slow operations in {self.alert_thresholds['monitoring_window_minutes']} minutes",
                severity="high",
                metrics={
                    "slow_operation_count": len(slow_operations),
                    "window_minutes": self.alert_thresholds["monitoring_window_minutes"],
                    "avg_duration_ms": statistics.mean([m.duration_ms for m in slow_operations])
                },
                recommendations=self._generate_optimization_recommendations(metric.operation_type)
            )
            self.alerts.append(alert)
            logger.warning(f"SLOW OPERATION PATTERN ALERT: {alert.message}")
        
        # Check for low success rate alert
        if recent_metrics:
            success_rate = len([m for m in recent_metrics if m.success]) / len(recent_metrics)
            if success_rate < self.alert_thresholds["success_rate_threshold"]:
                alert = PerformanceAlert(
                    timestamp=now,
                    alert_type="low_success_rate",
                    operation_type=metric.operation_type,
                    message=f"Low success rate detected: {success_rate:.2%} for {metric.operation_type.value}",
                    severity="high",
                    metrics={
                        "success_rate": success_rate,
                        "total_operations": len(recent_metrics),
                        "failed_operations": len([m for m in recent_metrics if not m.success])
                    },
                    recommendations=["Review error logs", "Check system resources", "Validate input data"]
                )
                self.alerts.append(alert)
                logger.warning(f"LOW SUCCESS RATE ALERT: {alert.message}")
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def _generate_optimization_recommendations(
        self, 
        operation_type: OperationType, 
        duration_ms: Optional[float] = None
    ) -> List[str]:
        """Generate optimization recommendations based on operation type and performance."""
        recommendations = []
        
        if operation_type == OperationType.INDEX_BUILDING:
            recommendations.extend([
                "Consider reducing document batch size",
                "Check ChromaDB connection performance",
                "Optimize document preprocessing",
                "Consider parallel processing for large datasets",
                "Review embedding model performance"
            ])
            
        elif operation_type == OperationType.SEARCH_OPERATION:
            recommendations.extend([
                "Optimize search query complexity",
                "Consider caching frequent searches",
                "Review retriever configuration",
                "Check index health and size",
                "Consider search result limiting"
            ])
            
        elif operation_type == OperationType.RESPONSE_GENERATION:
            recommendations.extend([
                "Reduce context length if possible",
                "Optimize LLM prompt structure",
                "Consider response caching",
                "Review strict response validation overhead",
                "Check LLM API performance"
            ])
            
        elif operation_type == OperationType.CHROMADB_OPERATION:
            recommendations.extend([
                "Check ChromaDB server performance",
                "Optimize collection size",
                "Consider connection pooling",
                "Review batch operation efficiency",
                "Check network latency to ChromaDB"
            ])
            
        elif operation_type == OperationType.MULTI_STRATEGY_SEARCH:
            recommendations.extend([
                "Reduce number of search strategies",
                "Optimize individual strategy performance",
                "Consider early termination on good results",
                "Review strategy ordering",
                "Cache strategy results"
            ])
        
        # Add duration-specific recommendations
        if duration_ms and duration_ms > 30000:  # > 30 seconds
            recommendations.extend([
                "Consider implementing operation timeouts",
                "Review system resource usage",
                "Check for memory leaks or resource contention",
                "Consider breaking operation into smaller chunks"
            ])
        
        return recommendations
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        now = datetime.now()
        
        # Overall system performance
        recent_metrics = [
            m for m in self.metrics
            if now - m.timestamp < timedelta(hours=24)
        ]
        
        dashboard = {
            "timestamp": now.isoformat(),
            "overview": {
                "total_operations_24h": len(recent_metrics),
                "success_rate_24h": len([m for m in recent_metrics if m.success]) / len(recent_metrics) if recent_metrics else 1.0,
                "avg_duration_ms_24h": statistics.mean([m.duration_ms for m in recent_metrics]) if recent_metrics else 0,
                "active_alerts": len([a for a in self.alerts if now - a.timestamp < timedelta(hours=1)])
            },
            "performance_by_operation": {},
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "type": alert.alert_type,
                    "operation": alert.operation_type.value,
                    "message": alert.message,
                    "severity": alert.severity,
                    "recommendations": alert.recommendations
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ],
            "optimization_opportunities": self._identify_optimization_opportunities()
        }
        
        # Performance stats by operation type
        for op_type, stats in self.performance_stats.items():
            dashboard["performance_by_operation"][op_type.value] = {
                "total_operations": stats.total_operations,
                "success_rate": stats.success_rate,
                "avg_duration_ms": stats.avg_duration_ms,
                "p95_duration_ms": stats.p95_duration_ms,
                "performance_level": stats.performance_level.value,
                "last_updated": stats.last_updated.isoformat()
            }
        
        return dashboard
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on performance data."""
        opportunities = []
        
        for op_type, stats in self.performance_stats.items():
            # Identify slow operations
            if stats.performance_level in [PerformanceLevel.SLOW, PerformanceLevel.CRITICAL]:
                opportunities.append({
                    "type": "slow_operation",
                    "operation": op_type.value,
                    "current_performance": stats.performance_level.value,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "recommendations": self._generate_optimization_recommendations(op_type, stats.avg_duration_ms),
                    "priority": "high" if stats.performance_level == PerformanceLevel.CRITICAL else "medium"
                })
            
            # Identify operations with low success rates
            if stats.success_rate < 0.95:
                opportunities.append({
                    "type": "low_success_rate",
                    "operation": op_type.value,
                    "success_rate": stats.success_rate,
                    "failed_operations": stats.failed_operations,
                    "recommendations": ["Review error logs", "Improve error handling", "Validate input data"],
                    "priority": "high" if stats.success_rate < 0.8 else "medium"
                })
        
        return opportunities
    
    def get_operation_performance(self, operation_type: OperationType) -> Optional[PerformanceStats]:
        """Get performance statistics for a specific operation type."""
        return self.performance_stats.get(operation_type)
    
    def clear_old_metrics(self, days: int = 7):
        """Clear metrics older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.metrics)
        self.metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ]
        
        cleared_count = original_count - len(self.metrics)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old performance metrics")
        
        # Also clear old alerts
        original_alert_count = len(self.alerts)
        self.alerts = [
            a for a in self.alerts
            if a.timestamp >= cutoff_time
        ]
        
        cleared_alert_count = original_alert_count - len(self.alerts)
        if cleared_alert_count > 0:
            logger.info(f"Cleared {cleared_alert_count} old performance alerts")
    
    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Export performance metrics for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ]
        
        return {
            "export_timestamp": datetime.now().isoformat(),
            "time_range_hours": hours,
            "total_metrics": len(recent_metrics),
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "operation_type": m.operation_type.value,
                    "operation_name": m.operation_name,
                    "duration_ms": m.duration_ms,
                    "success": m.success,
                    "user_id": m.user_id,
                    "performance_level": m.performance_level.value if m.performance_level else None,
                    "additional_data": m.additional_data
                }
                for m in recent_metrics
            ]
        }

    def get_chatbot_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive chatbot performance summary."""
        
        if self.total_requests == 0:
            return {"message": "No chatbot requests recorded yet"}
        
        # Calculate overall stats
        fast_percentage = (self.fast_responses / self.total_requests) * 100
        slow_percentage = (self.slow_responses / self.total_requests) * 100
        critical_percentage = (self.critical_responses / self.total_requests) * 100
        
        # Message type performance
        message_type_stats = {}
        for msg_type, times in self.message_type_times.items():
            if times:
                target_time = self.response_time_targets.get(msg_type, 5000.0)
                under_target = sum(1 for t in times if t <= target_time)
                
                message_type_stats[msg_type] = {
                    "total_requests": len(times),
                    "avg_response_time_ms": sum(times) / len(times),
                    "min_response_time_ms": min(times),
                    "max_response_time_ms": max(times),
                    "target_time_ms": target_time,
                    "target_met_percentage": (under_target / len(times)) * 100,
                    "p95_response_time_ms": self._calculate_percentile(times, 95)
                }
        
        # Recent slow operations
        recent_slow = [
            {
                "operation": op.operation_name,
                "duration_ms": op.duration_ms,
                "user_id": op.user_id,
                "message_type": op.additional_data.get("message_type"),
                "timestamp": op.timestamp
            }
            for op in self.slow_operations[-10:]  # Last 10 slow operations
        ]
        
        return {
            "overall_stats": {
                "total_requests": self.total_requests,
                "fast_responses": self.fast_responses,
                "slow_responses": self.slow_responses,
                "critical_responses": self.critical_responses,
                "fast_percentage": f"{fast_percentage:.1f}%",
                "slow_percentage": f"{slow_percentage:.1f}%", 
                "critical_percentage": f"{critical_percentage:.1f}%",
                "target_95_percent_under_5s": fast_percentage >= 95.0
            },
            "message_type_performance": message_type_stats,
            "recent_slow_operations": recent_slow,
            "performance_targets": self.response_time_targets
        }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts for monitoring."""
        
        alerts = []
        
        # Check overall performance
        if self.total_requests >= 10:  # Only alert after sufficient data
            fast_percentage = (self.fast_responses / self.total_requests) * 100
            
            if fast_percentage < 95.0:
                alerts.append({
                    "type": "performance_degradation",
                    "severity": "warning" if fast_percentage >= 90.0 else "critical",
                    "message": f"Only {fast_percentage:.1f}% of responses meet target times (target: 95%)",
                    "metric": "overall_performance",
                    "value": fast_percentage
                })
        
        # Check message type performance
        for msg_type, times in self.message_type_times.items():
            if len(times) >= 5:  # Need at least 5 samples
                target_time = self.response_time_targets.get(msg_type, 5000.0)
                under_target = sum(1 for t in times if t <= target_time)
                target_percentage = (under_target / len(times)) * 100
                
                if target_percentage < 90.0:
                    alerts.append({
                        "type": "message_type_slow",
                        "severity": "warning" if target_percentage >= 80.0 else "critical",
                        "message": f"{msg_type} responses are slow: {target_percentage:.1f}% meet target",
                        "metric": f"{msg_type}_performance",
                        "value": target_percentage,
                        "target": 90.0
                    })
        
        # Check for recent critical operations
        recent_critical = [
            op for op in self.slow_operations[-20:]  # Last 20 slow ops
            if op.duration_ms > self.CRITICAL_OPERATION_THRESHOLD
        ]
        
        if recent_critical:
            alerts.append({
                "type": "critical_response_time",
                "severity": "critical",
                "message": f"{len(recent_critical)} critical response times (>30s) in recent operations",
                "metric": "critical_operations",
                "value": len(recent_critical)
            })
        
        return alerts
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def log_performance_warning(self, operation: str, duration_ms: float, 
                              threshold_ms: float, user_id: Optional[str] = None,
                              additional_context: Optional[Dict[str, Any]] = None) -> None:
        """Log performance warning for operations exceeding thresholds."""
        
        context_str = ""
        if additional_context:
            context_str = f" | Context: {additional_context}"
        
        user_str = f" | User: {user_id}" if user_id else ""
        
        if duration_ms > self.CRITICAL_OPERATION_THRESHOLD:
            logger.critical(f"CRITICAL PERFORMANCE: {operation} took {duration_ms:.2f}ms (threshold: {threshold_ms:.2f}ms){user_str}{context_str}")
        elif duration_ms > threshold_ms:
            logger.warning(f"SLOW OPERATION: {operation} took {duration_ms:.2f}ms (threshold: {threshold_ms:.2f}ms){user_str}{context_str}")
    
    def get_user_performance_profile(self, user_id: str) -> Dict[str, Any]:
        """Get performance profile for a specific user."""
        
        user_metrics = [m for m in self.metrics if m.user_id == user_id]
        
        if not user_metrics:
            return {"message": f"No performance data for user {user_id}"}
        
        # Calculate user-specific stats
        user_times = [m.duration_ms for m in user_metrics]
        user_message_types = defaultdict(list)
        
        for metric in user_metrics:
            if metric.additional_data.get("message_type"):
                user_message_types[metric.additional_data["message_type"]].append(metric.duration_ms)
        
        return {
            "user_id": user_id,
            "total_requests": len(user_metrics),
            "avg_response_time_ms": sum(user_times) / len(user_times),
            "min_response_time_ms": min(user_times),
            "max_response_time_ms": max(user_times),
            "message_type_breakdown": {
                msg_type: {
                    "count": len(times),
                    "avg_time_ms": sum(times) / len(times),
                    "max_time_ms": max(times)
                }
                for msg_type, times in user_message_types.items()
            },
            "recent_operations": [
                {
                    "operation": m.operation_name,
                    "duration_ms": m.duration_ms,
                    "success": m.success,
                    "timestamp": m.timestamp
                }
                for m in user_metrics[-10:]  # Last 10 operations
            ]
        }


# Context manager for performance monitoring
@asynccontextmanager
async def monitor_performance(
    monitor: PerformanceMonitor,
    operation_type: OperationType,
    operation_name: str,
    user_id: Optional[str] = None,
    **additional_data
):
    """Async context manager for monitoring operation performance."""
    start_time = time.time()
    success = False
    
    try:
        yield
        success = True
    except Exception as e:
        success = False
        additional_data["error"] = str(e)
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        monitor.record_metric(
            operation_type=operation_type,
            operation_name=operation_name,
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            **additional_data
        )


# Decorator for performance monitoring
def monitor_performance_sync(
    monitor: PerformanceMonitor,
    operation_type: OperationType,
    operation_name: Optional[str] = None
):
    """Decorator for monitoring synchronous function performance."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_metric(
                    operation_type=operation_type,
                    operation_name=name,
                    duration_ms=duration_ms,
                    success=success
                )
        
        return wrapper
    return decorator


def monitor_performance_async(
    monitor: PerformanceMonitor,
    operation_type: OperationType,
    operation_name: Optional[str] = None
):
    """Decorator for monitoring asynchronous function performance."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_metric(
                    operation_type=operation_type,
                    operation_name=name,
                    duration_ms=duration_ms,
                    success=success
                )
        
        return wrapper
    return decorator


# Global performance monitor instance
performance_monitor = PerformanceMonitor()