# /app/concurrent_operations_manager.py
"""
Concurrent Operations Management System

This module provides intelligent management of concurrent database operations,
query queuing, load balancing, and resource monitoring for the RAG chatbot system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
import weakref
from collections import defaultdict, deque

from app.models import (
    QueryRequest, QueryResult, QueuePosition, BalancedRequest, ResourceUsageReport,
    RequestPriority, ConnectionPoolStatus
)
from app.database import db_manager

logger = logging.getLogger(__name__)

class OperationStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class QueuedOperation:
    request: QueryRequest
    priority_score: float
    enqueue_time: datetime
    timeout_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        # Higher priority score = higher priority in queue (min-heap, so we negate)
        return -self.priority_score < -other.priority_score

@dataclass
class ConnectionMetrics:
    active_operations: int = 0
    total_operations: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)

class ConcurrentOperationsManager:
    """
    Advanced concurrent operations management with intelligent queuing and load balancing.
    
    Handles multiple simultaneous RAG queries, manages database connections,
    and provides resource monitoring and optimization.
    """
    
    def __init__(self, max_concurrent_operations: int = 50):
        self.max_concurrent_operations = max_concurrent_operations
        self.operation_queue: List[QueuedOperation] = []
        self.active_operations: Dict[str, QueuedOperation] = {}
        self.completed_operations: deque = deque(maxlen=1000)
        
        # Connection pool management
        self.connection_metrics: Dict[str, ConnectionMetrics] = defaultdict(ConnectionMetrics)
        self.connection_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Resource monitoring
        self.resource_usage_history: deque = deque(maxlen=100)
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Load balancing
        self.load_balancer_stats = {
            "requests_balanced": 0,
            "avg_load_factor": 0.0,
            "connection_utilization": {}
        }
        
        # Circuit breaker for overload protection
        self.circuit_breaker = {
            "is_open": False,
            "failure_count": 0,
            "last_failure": None,
            "recovery_timeout": 60  # seconds
        }
        
        # Background task references (will be started when needed)
        self._queue_processor_task = None
        self._monitoring_task = None
        self._background_tasks_started = False
    
    def _start_background_tasks(self):
        """Start background tasks for queue processing and monitoring."""
        try:
            # Only start if we have a running event loop and haven't started yet
            if not self._background_tasks_started:
                asyncio.get_running_loop()  # Check if we have a running loop
                
                if not self._queue_processor_task:
                    self._queue_processor_task = asyncio.create_task(self._process_queue())
                
                if not self._monitoring_task:
                    self._monitoring_task = asyncio.create_task(self._monitor_resources())
                
                self._background_tasks_started = True
        except RuntimeError:
            # No running event loop - tasks will be started when needed
            pass
    
    async def handle_concurrent_queries(self, queries: List[QueryRequest]) -> List[QueryResult]:
        """
        Handle multiple concurrent queries with intelligent queuing and load balancing.
        
        Args:
            queries: List of QueryRequest objects to process
            
        Returns:
            List of QueryResult objects with processing results
        """
        try:
            # Start background tasks if not already started
            if not self._background_tasks_started:
                self._start_background_tasks()
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                return [
                    QueryResult(
                        request_id=f"query_{i}",
                        success=False,
                        duration_ms=0.0,
                        result_data=None,
                        error_message="System overloaded - circuit breaker open",
                        timestamp=datetime.now()
                    )
                    for i, _ in enumerate(queries)
                ]
            
            # Balance requests across available resources
            balanced_requests = await self.load_balance_requests(queries)
            
            # Process requests concurrently with controlled concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_operations)
            
            async def process_single_request(balanced_req: BalancedRequest) -> QueryResult:
                async with semaphore:
                    return await self._execute_balanced_request(balanced_req)
            
            # Execute all requests concurrently
            tasks = [
                process_single_request(balanced_req) 
                for balanced_req in balanced_requests
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(
                        QueryResult(
                            request_id=f"query_{i}",
                            success=False,
                            duration_ms=0.0,
                            result_data=None,
                            error_message=str(result),
                            timestamp=datetime.now()
                        )
                    )
                else:
                    final_results.append(result)
            
            # Update performance metrics
            self._update_performance_metrics(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error handling concurrent queries: {e}")
            return [
                QueryResult(
                    request_id=f"query_{i}",
                    success=False,
                    duration_ms=0.0,
                    result_data=None,
                    error_message=f"Concurrent processing error: {str(e)}",
                    timestamp=datetime.now()
                )
                for i, _ in enumerate(queries)
            ]
    
    async def implement_query_queuing(self, query: QueryRequest) -> QueuePosition:
        """
        Implement intelligent query queuing with priority management.
        
        Args:
            query: QueryRequest to queue
            
        Returns:
            QueuePosition with queue status information
        """
        try:
            # Calculate priority score
            priority_score = self._calculate_priority_score(query)
            
            # Create queued operation
            timeout_at = datetime.now() + timedelta(milliseconds=query.timeout_ms)
            queued_op = QueuedOperation(
                request=query,
                priority_score=priority_score,
                enqueue_time=datetime.now(),
                timeout_at=timeout_at
            )
            
            # Add to priority queue
            heapq.heappush(self.operation_queue, queued_op)
            
            # Calculate queue position
            position = self._calculate_queue_position(priority_score)
            estimated_wait_time = self._estimate_wait_time(position)
            
            return QueuePosition(
                position=position,
                estimated_wait_time_ms=estimated_wait_time,
                queue_size=len(self.operation_queue),
                priority=query.priority
            )
            
        except Exception as e:
            logger.error(f"Error queuing query: {e}")
            return QueuePosition(
                position=-1,
                estimated_wait_time_ms=0.0,
                queue_size=len(self.operation_queue),
                priority=query.priority
            )
    
    async def load_balance_requests(self, requests: List[QueryRequest]) -> List[BalancedRequest]:
        """
        Load balance requests across available database connections.
        
        Args:
            requests: List of requests to balance
            
        Returns:
            List of BalancedRequest objects with connection assignments
        """
        try:
            balanced_requests = []
            
            # Get available connections
            available_connections = await self._get_available_connections()
            
            if not available_connections:
                # No connections available, assign all to default
                for request in requests:
                    balanced_requests.append(
                        BalancedRequest(
                            original_request=request,
                            assigned_connection="default",
                            load_factor=1.0,
                            estimated_processing_time_ms=5000.0  # Default estimate
                        )
                    )
                return balanced_requests
            
            # Load balance across connections
            connection_loads = {conn: 0.0 for conn in available_connections}
            
            for request in requests:
                # Find least loaded connection
                best_connection = min(
                    available_connections,
                    key=lambda conn: (
                        connection_loads[conn] * self.connection_weights[conn]
                    )
                )
                
                # Calculate load factor and processing time estimate
                current_load = connection_loads[best_connection]
                load_factor = current_load / self.max_concurrent_operations
                
                estimated_time = self._estimate_processing_time(request, best_connection)
                
                # Create balanced request
                balanced_request = BalancedRequest(
                    original_request=request,
                    assigned_connection=best_connection,
                    load_factor=load_factor,
                    estimated_processing_time_ms=estimated_time
                )
                
                balanced_requests.append(balanced_request)
                
                # Update connection load
                connection_loads[best_connection] += 1.0
            
            # Update load balancer stats
            self.load_balancer_stats["requests_balanced"] += len(requests)
            self.load_balancer_stats["avg_load_factor"] = sum(
                br.load_factor for br in balanced_requests
            ) / len(balanced_requests) if balanced_requests else 0.0
            
            return balanced_requests
            
        except Exception as e:
            logger.error(f"Error load balancing requests: {e}")
            # Return unbalanced requests as fallback
            return [
                BalancedRequest(
                    original_request=request,
                    assigned_connection="default",
                    load_factor=1.0,
                    estimated_processing_time_ms=5000.0
                )
                for request in requests
            ]
    
    async def monitor_resource_usage(self) -> ResourceUsageReport:
        """
        Monitor system resource usage and performance metrics.
        
        Returns:
            ResourceUsageReport with current resource usage
        """
        try:
            # Get current resource metrics
            current_time = datetime.now()
            
            # Calculate CPU usage (approximate based on active operations)
            cpu_usage = min(100.0, (len(self.active_operations) / self.max_concurrent_operations) * 100)
            
            # Calculate memory usage (rough estimate)
            memory_usage = len(self.active_operations) * 10.0  # MB per operation estimate
            
            # Connection pool usage
            pool_usage = await self._calculate_pool_usage()
            
            # Network and disk I/O (simplified estimates)
            network_bandwidth = len(self.active_operations) * 0.5  # Mbps estimate
            disk_io_ops = len(self.active_operations) * 2.0  # Operations per second
            
            resource_report = ResourceUsageReport(
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                connection_pool_usage=pool_usage,
                disk_io_ops_per_sec=disk_io_ops,
                network_bandwidth_mbps=network_bandwidth,
                timestamp=current_time
            )
            
            # Store in history
            self.resource_usage_history.append(resource_report)
            
            # Check for resource pressure and adjust circuit breaker
            self._check_resource_pressure(resource_report)
            
            return resource_report
            
        except Exception as e:
            logger.error(f"Error monitoring resource usage: {e}")
            return ResourceUsageReport(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                connection_pool_usage=0.0,
                disk_io_ops_per_sec=0.0,
                network_bandwidth_mbps=0.0,
                timestamp=datetime.now()
            )
    
    async def manage_connection_pool(self) -> ConnectionPoolStatus:
        """
        Manage database connection pool and return status.
        
        Returns:
            ConnectionPoolStatus with pool metrics
        """
        try:
            # Get connection pool metrics from MongoDB
            active_connections = len(self.active_operations)
            max_connections = self.max_concurrent_operations
            waiting_connections = len(self.operation_queue)
            
            # Calculate utilization
            pool_utilization = active_connections / max_connections if max_connections > 0 else 0.0
            
            # Count connection errors
            connection_errors = sum(
                1 for op in self.completed_operations
                if hasattr(op, 'error_message') and op.error_message and 'connection' in op.error_message.lower()
            )
            
            return ConnectionPoolStatus(
                active_connections=active_connections,
                max_connections=max_connections,
                waiting_connections=waiting_connections,
                pool_utilization=pool_utilization,
                connection_errors=connection_errors
            )
            
        except Exception as e:
            logger.error(f"Error managing connection pool: {e}")
            return ConnectionPoolStatus(
                active_connections=0,
                max_connections=self.max_concurrent_operations,
                waiting_connections=0,
                pool_utilization=0.0,
                connection_errors=0
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            total_operations = len(self.completed_operations)
            successful_operations = sum(
                1 for op in self.completed_operations
                if hasattr(op, 'success') and op.success
            )
            
            avg_duration = 0.0
            if self.performance_metrics["response_times"]:
                avg_duration = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": successful_operations / total_operations if total_operations > 0 else 0.0,
                "avg_response_time_ms": avg_duration,
                "active_operations": len(self.active_operations),
                "queued_operations": len(self.operation_queue),
                "circuit_breaker_open": self.circuit_breaker["is_open"],
                "load_balancer_stats": self.load_balancer_stats,
                "resource_pressure": self._get_resource_pressure_level()
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    # Private helper methods
    
    async def _process_queue(self):
        """Background task to process the operation queue."""
        while True:
            try:
                if self.operation_queue and len(self.active_operations) < self.max_concurrent_operations:
                    # Get highest priority operation
                    operation = heapq.heappop(self.operation_queue)
                    
                    # Check if operation has timed out
                    if datetime.now() > operation.timeout_at:
                        continue
                    
                    # Start processing
                    operation_id = f"op_{int(time.time() * 1000)}"
                    self.active_operations[operation_id] = operation
                    
                    # Process asynchronously
                    asyncio.create_task(self._process_operation(operation_id, operation))
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1.0)
    
    async def _monitor_resources(self):
        """Background task to monitor system resources."""
        while True:
            try:
                await self.monitor_resource_usage()
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(10.0)
    
    async def _process_operation(self, operation_id: str, operation: QueuedOperation):
        """Process a single operation from the queue."""
        try:
            start_time = time.time()
            
            # Create a balanced request for the operation
            balanced_request = BalancedRequest(
                original_request=operation.request,
                assigned_connection="default",
                load_factor=0.5,
                estimated_processing_time_ms=2000.0
            )
            
            # Execute the operation
            result = await self._execute_balanced_request(balanced_request)
            
            # Store the result
            self.completed_operations.append(result)
            
            # Remove from active operations
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            
            logger.debug(f"Operation {operation_id} completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing operation {operation_id}: {e}")
            
            # Remove from active operations even on error
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
    
    async def _execute_balanced_request(self, balanced_req: BalancedRequest) -> QueryResult:
        """Execute a balanced request and return the result."""
        start_time = time.time()
        request = balanced_req.original_request
        
        try:
            # Simulate request processing (replace with actual implementation)
            processing_time = balanced_req.estimated_processing_time_ms / 1000.0
            await asyncio.sleep(min(processing_time, 10.0))  # Cap at 10 seconds
            
            # Update connection metrics
            connection = balanced_req.assigned_connection
            self.connection_metrics[connection].total_operations += 1
            
            duration_ms = (time.time() - start_time) * 1000
            
            return QueryResult(
                request_id=f"req_{request.user_id}_{int(start_time)}",
                success=True,
                duration_ms=duration_ms,
                result_data={"message": "Request processed successfully"},
                error_message=None,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Update error metrics
            connection = balanced_req.assigned_connection
            self.connection_metrics[connection].total_operations += 1
            
            return QueryResult(
                request_id=f"req_{request.user_id}_{int(start_time)}",
                success=False,
                duration_ms=duration_ms,
                result_data=None,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    def _calculate_priority_score(self, query: QueryRequest) -> float:
        """Calculate priority score for queue ordering."""
        base_score = {
            RequestPriority.CRITICAL: 1000.0,
            RequestPriority.HIGH: 100.0,
            RequestPriority.NORMAL: 10.0,
            RequestPriority.LOW: 1.0
        }.get(query.priority, 10.0)
        
        # Add time-based boost for older requests
        age_seconds = (datetime.now() - query.timestamp).total_seconds()
        age_boost = min(age_seconds / 60.0, 10.0)  # Max 10 point boost after 10 minutes
        
        return base_score + age_boost
    
    def _calculate_queue_position(self, priority_score: float) -> int:
        """Calculate position in queue based on priority score."""
        position = 1
        for op in self.operation_queue:
            if op.priority_score > priority_score:
                position += 1
        return position
    
    def _estimate_wait_time(self, position: int) -> float:
        """Estimate wait time in milliseconds based on queue position."""
        avg_processing_time = 2000.0  # 2 seconds average
        if self.performance_metrics["response_times"]:
            avg_processing_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
        
        return position * avg_processing_time
    
    async def _get_available_connections(self) -> List[str]:
        """Get list of available database connections."""
        # For now, return a simple list - can be enhanced with actual connection pool
        return ["mongodb_primary", "mongodb_secondary", "chromadb_main"]
    
    def _estimate_processing_time(self, request: QueryRequest, connection: str) -> float:
        """Estimate processing time for a request on a specific connection."""
        base_time = 2000.0  # 2 seconds base
        
        # Adjust based on connection type
        connection_multipliers = {
            "mongodb_primary": 1.0,
            "mongodb_secondary": 1.2,
            "chromadb_main": 0.8
        }
        
        multiplier = connection_multipliers.get(connection, 1.0)
        return base_time * multiplier
    
    async def _calculate_pool_usage(self) -> float:
        """Calculate current connection pool usage."""
        if not db_manager.client:
            return 0.0
        
        # Simple calculation based on active operations
        return len(self.active_operations) / self.max_concurrent_operations
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_breaker["is_open"]:
            return False
        
        # Check if recovery timeout has passed
        if self.circuit_breaker["last_failure"]:
            recovery_time = self.circuit_breaker["last_failure"] + timedelta(
                seconds=self.circuit_breaker["recovery_timeout"]
            )
            if datetime.now() > recovery_time:
                self.circuit_breaker["is_open"] = False
                self.circuit_breaker["failure_count"] = 0
                return False
        
        return True
    
    def _check_resource_pressure(self, resource_report: ResourceUsageReport):
        """Check resource pressure and update circuit breaker if needed."""
        high_pressure = (
            resource_report.cpu_usage_percent > 90 or
            resource_report.connection_pool_usage > 0.9 or
            resource_report.memory_usage_mb > 1000
        )
        
        if high_pressure:
            self.circuit_breaker["failure_count"] += 1
            if self.circuit_breaker["failure_count"] > 5:
                self.circuit_breaker["is_open"] = True
                self.circuit_breaker["last_failure"] = datetime.now()
        else:
            # Reset failure count on good performance
            self.circuit_breaker["failure_count"] = max(0, self.circuit_breaker["failure_count"] - 1)
    
    def _get_resource_pressure_level(self) -> str:
        """Get current resource pressure level."""
        if not self.resource_usage_history:
            return "unknown"
        
        latest = self.resource_usage_history[-1]
        
        if latest.cpu_usage_percent > 90 or latest.connection_pool_usage > 0.9:
            return "high"
        elif latest.cpu_usage_percent > 70 or latest.connection_pool_usage > 0.7:
            return "medium"
        else:
            return "low"
    
    def _update_performance_metrics(self, results: List[QueryResult]):
        """Update performance metrics based on query results."""
        for result in results:
            self.performance_metrics["response_times"].append(result.duration_ms)
            if result.success:
                self.performance_metrics["success_count"].append(1)
            else:
                self.performance_metrics["error_count"].append(1)

# Global instance
concurrent_operations_manager = ConcurrentOperationsManager() 