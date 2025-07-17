# /app/circuit_breaker.py
"""
Circuit Breaker for Performance Protection

This service implements circuit breaker pattern to prevent cascading failures
and provides timeout enforcement for slow operations.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    timeout_seconds: float = 30.0  # Timeout for operations
    recovery_timeout: float = 60.0  # Time before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open


@dataclass
class OperationResult:
    """Result of a circuit breaker protected operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    was_timeout: bool = False


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation for protecting against slow operations.
    
    Prevents cascading failures by monitoring operation success/failure rates
    and temporarily blocking operations when they're consistently failing.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        # Operation history (last 100 operations)
        self.operation_history: List[OperationResult] = []
        
        # Performance tracking
        self.total_operations = 0
        self.total_timeouts = 0
        self.total_circuit_opens = 0
    
    async def call(self, operation: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> OperationResult:
        """
        Execute operation with circuit breaker protection.
        
        Args:
            operation: Async function to execute
            fallback: Optional fallback function if circuit is open
            
        Returns:
            OperationResult with success status and result/error
        """
        self.total_operations += 1
        start_time = time.time()
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} attempting recovery (HALF_OPEN)")
            else:
                # Circuit is open, use fallback or fail fast
                if fallback:
                    try:
                        result = await fallback() if asyncio.iscoroutinefunction(fallback) else fallback()
                        return OperationResult(
                            success=True,
                            result=result,
                            duration_ms=(time.time() - start_time) * 1000,
                            circuit_state=self.state
                        )
                    except Exception as e:
                        return OperationResult(
                            success=False,
                            error=e,
                            duration_ms=(time.time() - start_time) * 1000,
                            circuit_state=self.state
                        )
                else:
                    return OperationResult(
                        success=False,
                        error=Exception(f"Circuit breaker {self.name} is OPEN"),
                        duration_ms=(time.time() - start_time) * 1000,
                        circuit_state=self.state
                    )
        
        # Execute operation with timeout
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(operation(), timeout=self.config.timeout_seconds)
            else:
                # For sync operations, run in executor with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(operation),
                    timeout=self.config.timeout_seconds
                )
            
            # Operation succeeded
            operation_result = OperationResult(
                success=True,
                result=result,
                duration_ms=(time.time() - start_time) * 1000,
                circuit_state=self.state
            )
            
            self._record_success()
            return operation_result
            
        except asyncio.TimeoutError:
            # Operation timed out
            self.total_timeouts += 1
            operation_result = OperationResult(
                success=False,
                error=Exception(f"Operation timed out after {self.config.timeout_seconds}s"),
                duration_ms=(time.time() - start_time) * 1000,
                circuit_state=self.state,
                was_timeout=True
            )
            
            self._record_failure()
            return operation_result
            
        except Exception as e:
            # Operation failed
            operation_result = OperationResult(
                success=False,
                error=e,
                duration_ms=(time.time() - start_time) * 1000,
                circuit_state=self.state
            )
            
            self._record_failure()
            return operation_result
        
        finally:
            # Store in history
            if hasattr(operation_result, 'success'):
                self.operation_history.append(operation_result)
                if len(self.operation_history) > 100:
                    self.operation_history.pop(0)
    
    def _record_success(self):
        """Record successful operation."""
        self.success_count += 1
        self.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} recovered (CLOSED)")
    
    def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and 
            self.failure_count >= self.config.failure_threshold):
            
            self.state = CircuitState.OPEN
            self.total_circuit_opens += 1
            self.success_count = 0
            logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN to HALF_OPEN."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        recent_operations = self.operation_history[-20:] if self.operation_history else []
        recent_success_rate = (
            sum(1 for op in recent_operations if op.success) / len(recent_operations) * 100
            if recent_operations else 0
        )
        
        avg_duration = (
            sum(op.duration_ms for op in recent_operations) / len(recent_operations)
            if recent_operations else 0
        )
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_operations": self.total_operations,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_timeouts": self.total_timeouts,
            "total_circuit_opens": self.total_circuit_opens,
            "recent_success_rate": f"{recent_success_rate:.1f}%",
            "avg_duration_ms": f"{avg_duration:.2f}",
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers used throughout the application.
    
    Provides pre-configured circuit breakers for common operations like
    LLM calls, index operations, and search operations.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Initialize default circuit breakers
        self._initialize_default_breakers()
    
    def _initialize_default_breakers(self):
        """Initialize circuit breakers for common operations."""
        
        # LLM operations (can be slow and unreliable)
        self.circuit_breakers["llm_chat"] = CircuitBreaker(
            "llm_chat",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=15.0,
                recovery_timeout=30.0,
                success_threshold=2
            )
        )
        
        # Index building operations (very slow)
        self.circuit_breakers["index_building"] = CircuitBreaker(
            "index_building",
            CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=60.0,
                recovery_timeout=120.0,
                success_threshold=1
            )
        )
        
        # Search operations (moderately slow)
        self.circuit_breakers["search_operations"] = CircuitBreaker(
            "search_operations",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=10.0,
                recovery_timeout=30.0,
                success_threshold=3
            )
        )
        
        # Property recommendation (can be slow due to complex processing)
        self.circuit_breakers["property_recommendation"] = CircuitBreaker(
            "property_recommendation",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=20.0,
                recovery_timeout=60.0,
                success_threshold=2
            )
        )
        
        # Appointment booking (external service calls)
        self.circuit_breakers["appointment_booking"] = CircuitBreaker(
            "appointment_booking",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=10.0,
                recovery_timeout=30.0,
                success_threshold=2
            )
        )
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create new circuit breaker with custom configuration."""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats() 
            for name, breaker in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary of all circuit breakers."""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(1 for b in self.circuit_breakers.values() if b.state == CircuitState.OPEN)
        half_open_breakers = sum(1 for b in self.circuit_breakers.values() if b.state == CircuitState.HALF_OPEN)
        
        return {
            "total_circuit_breakers": total_breakers,
            "open_circuits": open_breakers,
            "half_open_circuits": half_open_breakers,
            "healthy_circuits": total_breakers - open_breakers - half_open_breakers,
            "overall_health": "healthy" if open_breakers == 0 else "degraded" if open_breakers < total_breakers else "critical",
            "circuit_status": {
                name: breaker.state.value 
                for name, breaker in self.circuit_breakers.items()
            }
        }


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager() 