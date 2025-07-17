# /app/performance_logger.py
"""
Performance Logger for Chatbot Performance Optimization

This service provides structured logging for performance issues, errors,
and monitoring data to help identify and resolve performance problems.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

from app.models import MessageType, ProcessingStrategy, PerformanceMetrics

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"  
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceLogEntry:
    """Structured performance log entry."""
    timestamp: str
    level: LogLevel
    operation: str
    message: str
    duration_ms: Optional[float] = None
    user_id: Optional[str] = None
    message_type: Optional[str] = None
    processing_strategy: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class PerformanceLogger:
    """
    Enhanced logging service for performance monitoring and debugging.
    
    Provides structured logging with context, performance metrics tracking,
    and error aggregation for debugging and monitoring purposes.
    """
    
    def __init__(self):
        # Configure structured logger
        self.logger = logging.getLogger("performance")
        
        # Performance log entries (in-memory for recent analysis)
        self.log_entries: List[PerformanceLogEntry] = []
        self.max_entries = 1000
        
        # Error aggregation
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[PerformanceLogEntry] = []
        
        # Performance thresholds for automatic logging
        self.thresholds = {
            MessageType.GREETING: 2000.0,        # 2 seconds
            MessageType.CONVERSATIONAL: 2000.0,  # 2 seconds  
            MessageType.HELP_REQUEST: 2000.0,    # 2 seconds
            MessageType.PROPERTY_SEARCH: 5000.0, # 5 seconds
            MessageType.APPOINTMENT_REQUEST: 3000.0, # 3 seconds
            MessageType.THANK_YOU: 1000.0,       # 1 second
            MessageType.UNKNOWN: 3000.0          # 3 seconds
        }
    
    def log_slow_operation(self, operation: str, duration_ms: float, 
                          threshold_ms: float, user_id: Optional[str] = None,
                          message_type: Optional[MessageType] = None,
                          processing_strategy: Optional[ProcessingStrategy] = None,
                          additional_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log slow operation with context and performance details.
        
        Args:
            operation: Operation name
            duration_ms: Actual duration in milliseconds
            threshold_ms: Expected threshold in milliseconds
            user_id: Optional user identifier
            message_type: Optional message type
            processing_strategy: Optional processing strategy
            additional_context: Additional context information
        """
        
        level = LogLevel.WARNING
        if duration_ms > threshold_ms * 3:  # 3x threshold is critical
            level = LogLevel.CRITICAL
        elif duration_ms > threshold_ms * 2:  # 2x threshold is error
            level = LogLevel.ERROR
        
        context = {
            "threshold_ms": threshold_ms,
            "performance_ratio": duration_ms / threshold_ms,
            **(additional_context or {})
        }
        
        message = f"Slow operation: {operation} took {duration_ms:.2f}ms (threshold: {threshold_ms:.2f}ms, {duration_ms/threshold_ms:.1f}x slower)"
        
        self._log_entry(
            level=level,
            operation=operation,
            message=message,
            duration_ms=duration_ms,
            user_id=user_id,
            message_type=message_type.value if message_type else None,
            processing_strategy=processing_strategy.value if processing_strategy else None,
            context=context
        )
    
    def log_intent_detection_failure(self, error: Exception, message: str, 
                                   user_id: Optional[str] = None,
                                   fallback_used: bool = False) -> None:
        """Log intent detection failures with error details."""
        
        error_type = type(error).__name__
        self.error_counts[f"intent_detection_{error_type}"] = self.error_counts.get(f"intent_detection_{error_type}", 0) + 1
        
        context = {
            "message_preview": message[:100],
            "fallback_used": fallback_used,
            "error_traceback": traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
        }
        
        log_message = f"Intent detection failed: {error_type} - {str(error)}"
        if fallback_used:
            log_message += " (fallback classification used)"
        
        self._log_entry(
            level=LogLevel.ERROR,
            operation="intent_detection",
            message=log_message,
            user_id=user_id,
            error_type=error_type,
            error_details=str(error),
            context=context
        )
    
    def log_index_rebuild(self, user_id: str, reason: str, duration_ms: float,
                         success: bool, file_count: Optional[int] = None,
                         document_count: Optional[int] = None,
                         error: Optional[Exception] = None) -> None:
        """Log index rebuild operations."""
        
        level = LogLevel.INFO if success else LogLevel.ERROR
        
        context = {
            "rebuild_reason": reason,
            "file_count": file_count,
            "document_count": document_count,
            "success": success
        }
        
        if error:
            context["error_type"] = type(error).__name__
            context["error_details"] = str(error)
        
        if success:
            message = f"Index rebuilt successfully for user {user_id} in {duration_ms:.2f}ms"
            if document_count:
                message += f" ({document_count} documents from {file_count} files)"
        else:
            message = f"Index rebuild failed for user {user_id} after {duration_ms:.2f}ms"
            if error:
                message += f": {type(error).__name__}: {str(error)}"
        
        self._log_entry(
            level=level,
            operation="index_rebuild",
            message=message,
            duration_ms=duration_ms,
            user_id=user_id,
            error_type=type(error).__name__ if error else None,
            error_details=str(error) if error else None,
            context=context
        )
    
    def log_fallback_usage(self, operation: str, reason: str, user_id: Optional[str] = None,
                          original_error: Optional[Exception] = None,
                          fallback_duration_ms: Optional[float] = None) -> None:
        """Log fallback mechanism usage."""
        
        context = {
            "fallback_reason": reason,
            "original_operation": operation
        }
        
        if original_error:
            context["original_error"] = str(original_error)
            context["original_error_type"] = type(original_error).__name__
        
        message = f"Fallback used for {operation}: {reason}"
        if fallback_duration_ms:
            message += f" (fallback took {fallback_duration_ms:.2f}ms)"
        
        self._log_entry(
            level=LogLevel.WARNING,
            operation=f"{operation}_fallback",
            message=message,
            duration_ms=fallback_duration_ms,
            user_id=user_id,
            context=context
        )
    
    def log_performance_alert(self, alert_type: str, metric_name: str, 
                            current_value: float, threshold_value: float,
                            severity: str = "warning") -> None:
        """Log performance alerts and threshold violations."""
        
        level = LogLevel.WARNING
        if severity == "critical":
            level = LogLevel.CRITICAL
        elif severity == "error":
            level = LogLevel.ERROR
        
        context = {
            "alert_type": alert_type,
            "metric_name": metric_name,
            "current_value": current_value,
            "threshold_value": threshold_value,
            "severity": severity,
            "deviation_percentage": ((current_value - threshold_value) / threshold_value * 100) if threshold_value > 0 else 0
        }
        
        message = f"Performance alert: {alert_type} - {metric_name} is {current_value} (threshold: {threshold_value})"
        
        self._log_entry(
            level=level,
            operation="performance_alert",
            message=message,
            context=context
        )
    
    def log_circuit_breaker_event(self, breaker_name: str, event_type: str,
                                 current_state: str, failure_count: int = 0,
                                 operation: Optional[str] = None) -> None:
        """Log circuit breaker state changes and events."""
        
        level = LogLevel.WARNING if event_type == "opened" else LogLevel.INFO
        
        context = {
            "breaker_name": breaker_name,
            "event_type": event_type,
            "current_state": current_state,
            "failure_count": failure_count,
            "operation": operation
        }
        
        message = f"Circuit breaker {breaker_name} {event_type}"
        if failure_count > 0:
            message += f" (after {failure_count} failures)"
        
        self._log_entry(
            level=level,
            operation="circuit_breaker",
            message=message,
            context=context
        )
    
    def _log_entry(self, level: LogLevel, operation: str, message: str,
                   duration_ms: Optional[float] = None,
                   user_id: Optional[str] = None,
                   message_type: Optional[str] = None,
                   processing_strategy: Optional[str] = None,
                   error_type: Optional[str] = None,
                   error_details: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> None:
        """Create and store structured log entry."""
        
        # Create log entry
        entry = PerformanceLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            operation=operation,
            message=message,
            duration_ms=duration_ms,
            user_id=user_id,
            message_type=message_type,
            processing_strategy=processing_strategy,
            error_type=error_type,
            error_details=error_details,
            context=context or {}
        )
        
        # Store in memory
        self.log_entries.append(entry)
        if len(self.log_entries) > self.max_entries:
            self.log_entries.pop(0)
        
        # Track errors
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.recent_errors.append(entry)
            if len(self.recent_errors) > 50:  # Keep last 50 errors
                self.recent_errors.pop(0)
        
        # Log to Python logger with structured data
        log_data = asdict(entry)
        
        if level == LogLevel.DEBUG:
            self.logger.debug(json.dumps(log_data))
        elif level == LogLevel.INFO:
            self.logger.info(json.dumps(log_data))
        elif level == LogLevel.WARNING:
            self.logger.warning(json.dumps(log_data))
        elif level == LogLevel.ERROR:
            self.logger.error(json.dumps(log_data))
        elif level == LogLevel.CRITICAL:
            self.logger.critical(json.dumps(log_data))
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_entries = [
            entry for entry in self.log_entries
            if datetime.fromisoformat(entry.timestamp) >= cutoff_time
        ]
        
        # Count by operation
        operation_counts = {}
        operation_durations = {}
        
        for entry in recent_entries:
            op = entry.operation
            operation_counts[op] = operation_counts.get(op, 0) + 1
            
            if entry.duration_ms:
                if op not in operation_durations:
                    operation_durations[op] = []
                operation_durations[op].append(entry.duration_ms)
        
        # Count by level
        level_counts = {}
        for entry in recent_entries:
            level = entry.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Recent errors
        recent_error_types = {}
        for entry in self.recent_errors:
            if entry.error_type:
                recent_error_types[entry.error_type] = recent_error_types.get(entry.error_type, 0) + 1
        
        return {
            "time_period_hours": hours,
            "total_log_entries": len(recent_entries),
            "log_level_breakdown": level_counts,
            "operation_counts": operation_counts,
            "avg_durations_ms": {
                op: sum(durations) / len(durations)
                for op, durations in operation_durations.items()
            },
            "recent_error_types": recent_error_types,
            "error_rate": len(self.recent_errors) / len(recent_entries) * 100 if recent_entries else 0
        }
    
    def get_recent_logs(self, limit: int = 50, level: Optional[LogLevel] = None) -> List[Dict[str, Any]]:
        """Get recent log entries, optionally filtered by level."""
        
        entries = self.log_entries
        if level:
            entries = [entry for entry in entries if entry.level == level]
        
        return [asdict(entry) for entry in entries[-limit:]]
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis."""
        
        return {
            "total_error_types": len(self.error_counts),
            "error_frequency": self.error_counts,
            "recent_errors": [asdict(entry) for entry in self.recent_errors[-10:]],
            "most_common_errors": sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def clear_logs(self, older_than_hours: Optional[int] = None) -> int:
        """Clear old log entries."""
        
        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            original_count = len(self.log_entries)
            self.log_entries = [
                entry for entry in self.log_entries
                if datetime.fromisoformat(entry.timestamp) >= cutoff_time
            ]
            cleared_count = original_count - len(self.log_entries)
        else:
            cleared_count = len(self.log_entries)
            self.log_entries.clear()
            self.recent_errors.clear()
            
        self.logger.info(f"Cleared {cleared_count} performance log entries")
        return cleared_count


# Global performance logger instance
performance_logger = PerformanceLogger() 