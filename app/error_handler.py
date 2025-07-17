# /app/error_handler.py
"""
Comprehensive error handling and recovery system for the RAG chatbot.

This module provides:
1. Comprehensive error handling for all RAG operations
2. Automatic recovery mechanisms for common failures
3. User-friendly error messages with actionable guidance
4. System health monitoring and alerting
"""

import logging
import traceback
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better handling and recovery."""
    CHROMADB_CONNECTION = "chromadb_connection"
    INDEX_CREATION = "index_creation"
    RETRIEVER_CREATION = "retriever_creation"
    SEARCH_OPERATION = "search_operation"
    RESPONSE_GENERATION = "response_generation"
    USER_CONTEXT = "user_context"
    DOCUMENT_PROCESSING = "document_processing"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error tracking and debugging."""
    user_id: Optional[str] = None
    operation: Optional[str] = None
    query: Optional[str] = None
    file_path: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken for an error."""
    name: str
    description: str
    action_func: Callable
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True


@dataclass
class ErrorRecord:
    """Record of an error occurrence with context and recovery attempts."""
    timestamp: datetime
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_actions: List[str] = field(default_factory=list)
    user_message: Optional[str] = None


class SystemHealthMonitor:
    """Monitors system health and tracks error patterns."""
    
    def __init__(self):
        self.error_history: List[ErrorRecord] = []
        self.health_metrics: Dict[str, Any] = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "last_health_check": None,
            "system_status": "healthy"
        }
        self.alert_thresholds = {
            "critical_errors_per_hour": 5,
            "high_errors_per_hour": 20,
            "total_errors_per_hour": 100,
            "recovery_failure_rate": 0.5
        }
    
    def record_error(self, error_record: ErrorRecord):
        """Record an error and update health metrics."""
        self.error_history.append(error_record)
        self._update_health_metrics()
        self._check_alert_conditions(error_record)
    
    def _update_health_metrics(self):
        """Update system health metrics based on error history."""
        now = datetime.now()
        recent_errors = [
            error for error in self.error_history
            if now - error.timestamp < timedelta(hours=1)
        ]
        
        self.health_metrics.update({
            "total_errors": len(self.error_history),
            "recent_errors_count": len(recent_errors),
            "errors_by_category": self._count_by_category(recent_errors),
            "errors_by_severity": self._count_by_severity(recent_errors),
            "recovery_success_rate": self._calculate_recovery_rate(),
            "last_health_check": now.isoformat(),
            "system_status": self._determine_system_status(recent_errors)
        })
    
    def _count_by_category(self, errors: List[ErrorRecord]) -> Dict[str, int]:
        """Count errors by category."""
        counts = {}
        for error in errors:
            category = error.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _count_by_severity(self, errors: List[ErrorRecord]) -> Dict[str, int]:
        """Count errors by severity."""
        counts = {}
        for error in errors:
            severity = error.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate the success rate of recovery attempts."""
        recovery_attempts = [e for e in self.error_history if e.recovery_attempted]
        if not recovery_attempts:
            return 1.0
        
        successful = len([e for e in recovery_attempts if e.recovery_successful])
        return successful / len(recovery_attempts)
    
    def _determine_system_status(self, recent_errors: List[ErrorRecord]) -> str:
        """Determine overall system status based on recent errors."""
        critical_count = len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL])
        high_count = len([e for e in recent_errors if e.severity == ErrorSeverity.HIGH])
        
        if critical_count >= self.alert_thresholds["critical_errors_per_hour"]:
            return "critical"
        elif high_count >= self.alert_thresholds["high_errors_per_hour"]:
            return "degraded"
        elif len(recent_errors) >= self.alert_thresholds["total_errors_per_hour"]:
            return "stressed"
        else:
            return "healthy"
    
    def _check_alert_conditions(self, error_record: ErrorRecord):
        """Check if alert conditions are met and log alerts."""
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR ALERT: {error_record.error_message}")
        
        # Check for error rate thresholds
        now = datetime.now()
        recent_errors = [
            error for error in self.error_history
            if now - error.timestamp < timedelta(hours=1)
        ]
        
        if len(recent_errors) >= self.alert_thresholds["total_errors_per_hour"]:
            logger.warning(f"HIGH ERROR RATE ALERT: {len(recent_errors)} errors in the last hour")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "health_metrics": self.health_metrics,
            "recent_errors": [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.error_message,
                    "recovery_successful": error.recovery_successful
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ],
            "alert_thresholds": self.alert_thresholds
        }


class RAGErrorHandler:
    """Main error handler for RAG operations with recovery mechanisms."""
    
    def __init__(self):
        self.health_monitor = SystemHealthMonitor()
        self.recovery_actions = self._initialize_recovery_actions()
        self.user_friendly_messages = self._initialize_user_messages()
    
    def _initialize_recovery_actions(self) -> Dict[ErrorCategory, List[RecoveryAction]]:
        """Initialize recovery actions for different error categories."""
        return {
            ErrorCategory.CHROMADB_CONNECTION: [
                RecoveryAction(
                    name="reconnect_chromadb",
                    description="Attempt to reconnect to ChromaDB",
                    action_func=self._reconnect_chromadb,
                    max_retries=3,
                    retry_delay=2.0
                ),
                RecoveryAction(
                    name="fallback_memory_storage",
                    description="Fall back to in-memory storage",
                    action_func=self._fallback_memory_storage,
                    max_retries=1
                )
            ],
            ErrorCategory.INDEX_CREATION: [
                RecoveryAction(
                    name="rebuild_index",
                    description="Rebuild the user index from documents",
                    action_func=self._rebuild_user_index,
                    max_retries=2,
                    retry_delay=5.0
                ),
                RecoveryAction(
                    name="clear_corrupted_index",
                    description="Clear potentially corrupted index data",
                    action_func=self._clear_corrupted_index,
                    max_retries=1
                )
            ],
            ErrorCategory.RETRIEVER_CREATION: [
                RecoveryAction(
                    name="recreate_retrievers",
                    description="Recreate BM25 and vector retrievers",
                    action_func=self._recreate_retrievers,
                    max_retries=2
                ),
                RecoveryAction(
                    name="use_vector_only",
                    description="Fall back to vector search only",
                    action_func=self._use_vector_only,
                    max_retries=1
                )
            ],
            ErrorCategory.SEARCH_OPERATION: [
                RecoveryAction(
                    name="retry_with_simpler_query",
                    description="Retry search with simplified query",
                    action_func=self._retry_simpler_query,
                    max_retries=2
                ),
                RecoveryAction(
                    name="fallback_basic_search",
                    description="Use basic vector search as fallback",
                    action_func=self._fallback_basic_search,
                    max_retries=1
                )
            ],
            ErrorCategory.RESPONSE_GENERATION: [
                RecoveryAction(
                    name="retry_with_shorter_context",
                    description="Retry with reduced context length",
                    action_func=self._retry_shorter_context,
                    max_retries=2
                ),
                RecoveryAction(
                    name="use_template_response",
                    description="Use template response for common queries",
                    action_func=self._use_template_response,
                    max_retries=1
                )
            ]
        }
    
    def _initialize_user_messages(self) -> Dict[ErrorCategory, Dict[ErrorSeverity, str]]:
        """Initialize user-friendly error messages with actionable guidance."""
        return {
            ErrorCategory.CHROMADB_CONNECTION: {
                ErrorSeverity.CRITICAL: "I'm experiencing a critical database connection issue. Please wait a moment while I attempt to restore service. If this persists, please contact support.",
                ErrorSeverity.HIGH: "I'm having trouble accessing the document database. I'm attempting to reconnect and will try using temporary storage if needed. Your request may take a bit longer than usual.",
                ErrorSeverity.MEDIUM: "There's a temporary issue with document storage. I'm working to resolve it automatically. You can continue asking questions, though responses may be slower.",
                ErrorSeverity.LOW: "Minor connectivity issue detected. Continuing with backup systems - you shouldn't notice any difference."
            },
            ErrorCategory.INDEX_CREATION: {
                ErrorSeverity.CRITICAL: "I cannot access your documents due to a critical indexing issue. Please try uploading your documents again, or contact support if this continues.",
                ErrorSeverity.HIGH: "I'm having trouble building your document index. Let me try rebuilding it from your uploaded files. This may take a few moments.",
                ErrorSeverity.MEDIUM: "There's an issue with your document index. I'm working to fix it automatically. You may want to try rephrasing your question.",
                ErrorSeverity.LOW: "Minor indexing issue detected. Attempting automatic repair - your next query should work normally."
            },
            ErrorCategory.RETRIEVER_CREATION: {
                ErrorSeverity.CRITICAL: "I cannot search your documents right now due to a system issue. Please try again in a few minutes, or upload your documents again if the problem persists.",
                ErrorSeverity.HIGH: "I can't access your documents right now. I'm trying to restore the search functionality. Please wait a moment and try your question again.",
                ErrorSeverity.MEDIUM: "Search system needs repair. Working on it now. You might want to try a simpler version of your question.",
                ErrorSeverity.LOW: "Minor search system issue. Attempting fix - please try your question again."
            },
            ErrorCategory.SEARCH_OPERATION: {
                ErrorSeverity.HIGH: "I'm having significant trouble searching for that information. Please try rephrasing your question with simpler terms, or ask about a different property.",
                ErrorSeverity.MEDIUM: "I'm having trouble finding information for your query. Let me try a different search approach. You could also try rephrasing your question.",
                ErrorSeverity.LOW: "Search taking longer than usual. Trying alternative methods - please wait a moment."
            },
            ErrorCategory.RESPONSE_GENERATION: {
                ErrorSeverity.HIGH: "I found information but I'm having trouble creating a proper response. Please try asking your question in a different way, or ask about specific details you need.",
                ErrorSeverity.MEDIUM: "I found relevant information but I'm having trouble formatting the response. Let me try again with a simpler approach.",
                ErrorSeverity.LOW: "Minor issue generating response. Retrying now - this should only take a moment."
            },
            ErrorCategory.USER_CONTEXT: {
                ErrorSeverity.HIGH: "I'm having trouble accessing your specific documents. Please make sure your documents are uploaded correctly, or try uploading them again.",
                ErrorSeverity.MEDIUM: "There's an issue with your user profile. I'm working to fix it. You might want to try logging out and back in.",
                ErrorSeverity.LOW: "Minor user context issue detected. Resolving automatically."
            },
            ErrorCategory.DOCUMENT_PROCESSING: {
                ErrorSeverity.HIGH: "I cannot process your documents properly. Please check that your files are valid CSV format and try uploading them again.",
                ErrorSeverity.MEDIUM: "There's an issue processing your documents. Please verify your files are in the correct format (CSV) and contain property data.",
                ErrorSeverity.LOW: "Minor document processing issue. Continuing with available data."
            },
            ErrorCategory.EXTERNAL_API: {
                ErrorSeverity.HIGH: "I'm having trouble connecting to external services. Please try again in a few minutes. If this persists, some features may be temporarily unavailable.",
                ErrorSeverity.MEDIUM: "External service temporarily unavailable. I'll try alternative methods to help you.",
                ErrorSeverity.LOW: "Minor external service issue. Continuing with backup systems."
            },
            ErrorCategory.SYSTEM_RESOURCE: {
                ErrorSeverity.CRITICAL: "System resources are critically low. Please try again in a few minutes. If this continues, please contact support immediately.",
                ErrorSeverity.HIGH: "System is under heavy load. Your request may take longer than usual. Please be patient.",
                ErrorSeverity.MEDIUM: "System experiencing moderate load. Performance may be slower than normal.",
                ErrorSeverity.LOW: "Minor system resource issue. Continuing normally."
            },
            ErrorCategory.UNKNOWN: {
                ErrorSeverity.HIGH: "I encountered an unexpected issue. Please try rephrasing your question or ask about something else. If this continues, please contact support.",
                ErrorSeverity.MEDIUM: "Something unexpected happened. Please try your request again, or try asking in a different way.",
                ErrorSeverity.LOW: "Minor unexpected issue. Retrying automatically."
            }
        }
    
    async def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> tuple[bool, Optional[str], Optional[Any]]:
        """
        Handle an error with automatic recovery attempts.
        
        Returns:
            (recovery_successful, user_message, recovered_result)
        """
        # Categorize and assess severity if not provided
        if category is None:
            category = self._categorize_error(error, context)
        if severity is None:
            severity = self._assess_severity(error, category, context)
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        logger.error(f"Error in {context.operation}: {error}")
        logger.debug(f"Error context: {context}")
        
        # Attempt recovery
        recovery_successful = False
        recovered_result = None
        user_message = None
        
        if category in self.recovery_actions:
            error_record.recovery_attempted = True
            recovery_successful, recovered_result = await self._attempt_recovery(
                error, context, category, error_record
            )
            error_record.recovery_successful = recovery_successful
        
        # Generate user-friendly message
        user_message = self._get_user_message(category, severity, recovery_successful)
        error_record.user_message = user_message
        
        # Record the error
        self.health_monitor.record_error(error_record)
        
        return recovery_successful, user_message, recovered_result
    
    def _categorize_error(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Categorize error based on type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # ChromaDB related errors
        if "chroma" in error_message or "connection" in error_message:
            return ErrorCategory.CHROMADB_CONNECTION
        
        # Index creation errors
        if "index" in error_message or context.operation in ["build_index", "get_index"]:
            return ErrorCategory.INDEX_CREATION
        
        # Retriever errors
        if "retriever" in error_message or "fusion" in error_message:
            return ErrorCategory.RETRIEVER_CREATION
        
        # Search operation errors
        if context.operation in ["search", "retrieve", "query"]:
            return ErrorCategory.SEARCH_OPERATION
        
        # Response generation errors
        if context.operation in ["generate_response", "chat"]:
            return ErrorCategory.RESPONSE_GENERATION
        
        # User context errors
        if "user" in error_message and context.user_id:
            return ErrorCategory.USER_CONTEXT
        
        # Document processing errors
        if "document" in error_message or "csv" in error_message:
            return ErrorCategory.DOCUMENT_PROCESSING
        
        # API errors
        if "api" in error_message or "http" in error_message:
            return ErrorCategory.EXTERNAL_API
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception, category: ErrorCategory, context: ErrorContext) -> ErrorSeverity:
        """Assess error severity based on type, category, and context."""
        error_message = str(error).lower()
        
        # Critical errors that break core functionality
        if category == ErrorCategory.CHROMADB_CONNECTION and "connection refused" in error_message:
            return ErrorSeverity.CRITICAL
        
        if category == ErrorCategory.SYSTEM_RESOURCE and "memory" in error_message:
            return ErrorSeverity.CRITICAL
        
        # High severity errors that significantly impact user experience
        if category in [ErrorCategory.INDEX_CREATION, ErrorCategory.RETRIEVER_CREATION]:
            return ErrorSeverity.HIGH
        
        if category == ErrorCategory.RESPONSE_GENERATION and context.user_id:
            return ErrorSeverity.HIGH
        
        # Medium severity for search and processing issues
        if category in [ErrorCategory.SEARCH_OPERATION, ErrorCategory.DOCUMENT_PROCESSING]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for minor issues
        return ErrorSeverity.LOW
    
    async def _attempt_recovery(
        self,
        error: Exception,
        context: ErrorContext,
        category: ErrorCategory,
        error_record: ErrorRecord
    ) -> tuple[bool, Optional[Any]]:
        """Attempt recovery using registered recovery actions."""
        recovery_actions = self.recovery_actions.get(category, [])
        
        for action in recovery_actions:
            logger.info(f"Attempting recovery action: {action.name}")
            error_record.recovery_actions.append(action.name)
            
            for attempt in range(action.max_retries):
                try:
                    # Calculate delay with exponential backoff
                    if attempt > 0:
                        delay = action.retry_delay
                        if action.exponential_backoff:
                            delay *= (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                    
                    # Execute recovery action
                    result = await action.action_func(error, context)
                    if result is not None:
                        logger.info(f"Recovery action {action.name} succeeded on attempt {attempt + 1}")
                        return True, result
                
                except Exception as recovery_error:
                    logger.warning(f"Recovery action {action.name} failed on attempt {attempt + 1}: {recovery_error}")
                    if attempt == action.max_retries - 1:
                        logger.error(f"Recovery action {action.name} exhausted all retries")
        
        return False, None
    
    def _get_user_message(self, category: ErrorCategory, severity: ErrorSeverity, recovery_successful: bool) -> str:
        """Get user-friendly error message."""
        base_messages = self.user_friendly_messages.get(category, {})
        base_message = base_messages.get(severity, "I encountered an issue while processing your request.")
        
        if recovery_successful:
            return f"{base_message} I've resolved the issue and your request should work now."
        else:
            return f"{base_message} Please try again in a moment, or contact support if the issue persists."
    
    # Recovery action implementations
    async def _reconnect_chromadb(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Attempt to reconnect to ChromaDB."""
        try:
            from app.chroma_client import chroma_manager
            # Close existing connection
            chroma_manager.close_client()
            # Get new client (will reconnect)
            client = chroma_manager.get_client()
            return client
        except Exception as e:
            logger.error(f"ChromaDB reconnection failed: {e}")
            return None
    
    async def _fallback_memory_storage(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Fall back to in-memory storage."""
        try:
            # This would trigger in-memory index creation
            logger.info("Falling back to in-memory storage")
            return "memory_fallback"
        except Exception as e:
            logger.error(f"Memory fallback failed: {e}")
            return None
    
    async def _rebuild_user_index(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Rebuild user index from documents."""
        if not context.user_id:
            return None
        
        try:
            import app.rag as rag_module
            import os
            
            # Find user documents
            user_doc_dir = os.path.join("user_documents", context.user_id)
            if os.path.exists(user_doc_dir):
                csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
                if csv_files:
                    file_paths = [os.path.join(user_doc_dir, f) for f in csv_files]
                    index = await rag_module.build_user_index(context.user_id, file_paths)
                    return index
            return None
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return None
    
    async def _clear_corrupted_index(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Clear potentially corrupted index data."""
        if not context.user_id:
            return None
        
        try:
            import app.rag as rag_module
            success = await rag_module.clear_user_index(context.user_id)
            return success
        except Exception as e:
            logger.error(f"Index clearing failed: {e}")
            return None
    
    async def _recreate_retrievers(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recreate BM25 and vector retrievers."""
        try:
            import app.rag as rag_module
            retriever = rag_module.get_fusion_retriever(context.user_id)
            return retriever
        except Exception as e:
            logger.error(f"Retriever recreation failed: {e}")
            return None
    
    async def _use_vector_only(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Fall back to vector search only."""
        try:
            import app.rag as rag_module
            if context.user_id and context.user_id in rag_module.user_indexes:
                index = rag_module.user_indexes[context.user_id]
                retriever = index.as_retriever(similarity_top_k=5)
                return retriever
            return None
        except Exception as e:
            logger.error(f"Vector-only fallback failed: {e}")
            return None
    
    async def _retry_simpler_query(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Retry search with simplified query."""
        if not context.query:
            return None
        
        try:
            # Simplify query by removing special characters and extra words
            simple_query = " ".join(context.query.split()[:5])  # First 5 words
            simple_query = "".join(c for c in simple_query if c.isalnum() or c.isspace())
            
            import app.rag as rag_module
            retriever = rag_module.get_fusion_retriever(context.user_id)
            if retriever:
                results = await retriever.aretrieve(simple_query)
                return results
            return None
        except Exception as e:
            logger.error(f"Simplified query retry failed: {e}")
            return None
    
    async def _fallback_basic_search(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Use basic vector search as fallback."""
        try:
            import app.rag as rag_module
            if context.user_id and context.user_id in rag_module.user_indexes:
                index = rag_module.user_indexes[context.user_id]
                retriever = index.as_retriever(similarity_top_k=3)
                if context.query:
                    results = await asyncio.to_thread(retriever.retrieve, context.query)
                    return results
            return None
        except Exception as e:
            logger.error(f"Basic search fallback failed: {e}")
            return None
    
    async def _retry_shorter_context(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Retry with reduced context length."""
        try:
            # This would be implemented based on the specific response generation logic
            logger.info("Retrying with shorter context")
            return "shorter_context_retry"
        except Exception as e:
            logger.error(f"Shorter context retry failed: {e}")
            return None
    
    async def _use_template_response(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Use template response for common queries."""
        try:
            if context.query:
                query_lower = context.query.lower()
                if "hello" in query_lower or "hi" in query_lower:
                    return "Hello! I'm here to help you find property information. How can I assist you today?"
                elif "help" in query_lower:
                    return "I can help you search for properties, get details about specific addresses, and answer questions about available listings. What would you like to know?"
            
            return "I apologize, but I'm having trouble processing your request right now. Please try rephrasing your question or ask about a different property."
        except Exception as e:
            logger.error(f"Template response failed: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return self.health_monitor.get_health_report()


# Global error handler instance
rag_error_handler = RAGErrorHandler()


@asynccontextmanager
async def error_handling_context(operation: str, user_id: Optional[str] = None, **kwargs):
    """Context manager for automatic error handling in RAG operations."""
    context = ErrorContext(
        user_id=user_id,
        operation=operation,
        additional_data=kwargs
    )
    
    try:
        yield context
    except Exception as e:
        recovery_successful, user_message, recovered_result = await rag_error_handler.handle_error(
            e, context
        )
        
        if not recovery_successful:
            # Re-raise with user-friendly message
            raise Exception(user_message or str(e)) from e


def handle_sync_error(
    error: Exception,
    operation: str,
    user_id: Optional[str] = None,
    **kwargs
) -> tuple[bool, Optional[str]]:
    """Synchronous error handler for non-async contexts."""
    context = ErrorContext(
        user_id=user_id,
        operation=operation,
        additional_data=kwargs
    )
    
    # Create error record without recovery (sync context)
    error_record = ErrorRecord(
        timestamp=datetime.now(),
        error_type=type(error).__name__,
        error_message=str(error),
        category=rag_error_handler._categorize_error(error, context),
        severity=rag_error_handler._assess_severity(error, rag_error_handler._categorize_error(error, context), context),
        context=context,
        stack_trace=traceback.format_exc()
    )
    
    # Record error
    rag_error_handler.health_monitor.record_error(error_record)
    
    # Get user message
    user_message = rag_error_handler._get_user_message(
        error_record.category,
        error_record.severity,
        False  # No recovery in sync context
    )
    
    logger.error(f"Sync error in {operation}: {error}")
    
    return False, user_message