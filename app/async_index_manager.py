# /app/async_index_manager.py
"""
Asynchronous Index Manager

This service manages index rebuilding asynchronously without blocking user responses,
providing fallback responses and coordinating rebuild operations.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import app.rag as rag_module

logger = logging.getLogger(__name__)


class RebuildStatus(str, Enum):
    """Status of index rebuild operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RebuildOperation:
    """Information about an index rebuild operation."""
    user_id: str
    status: RebuildStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    file_paths: List[str] = None
    progress_messages: List[str] = None
    error_message: Optional[str] = None
    document_count: Optional[int] = None
    
    def __post_init__(self):
        if self.file_paths is None:
            self.file_paths = []
        if self.progress_messages is None:
            self.progress_messages = []


class AsyncIndexManager:
    """
    Service for managing asynchronous index operations.
    
    Provides non-blocking index rebuilding with fallback responses and
    coordination to prevent multiple simultaneous rebuilds for the same user.
    """
    
    def __init__(self):
        # Track active rebuild operations
        self.active_rebuilds: Dict[str, RebuildOperation] = {}
        
        # Set of users currently being rebuilt (for coordination)
        self.rebuilding_users: Set[str] = set()
        
        # Completed operations history (last 100)
        self.rebuild_history: List[RebuildOperation] = []
        
        # Fallback responses for users with rebuilding indices
        self.fallback_responses = [
            "I'm currently updating your property database to ensure you get the most accurate results. This won't take long! In the meantime, feel free to ask me general questions about properties or schedule an appointment.",
            
            "Your personalized property index is being refreshed right now to give you the best search results. This process usually takes less than a minute. Would you like to know about our services while we wait?",
            
            "I'm optimizing your property search capabilities in the background. You'll have access to better, faster results very soon! Is there anything general I can help you with right now?",
            
            "Currently updating your property database for improved search performance. This quick process ensures you get the most relevant results. Feel free to ask about our appointment booking service!",
        ]
        
        # Performance tracking
        self.rebuild_times = []
        self.fallback_response_count = 0
    
    async def rebuild_user_index_async(self, user_id: str, file_paths: Optional[List[str]] = None, 
                                     priority: str = "normal") -> RebuildOperation:
        """
        Start asynchronous index rebuild for a user.
        
        Args:
            user_id: User identifier
            file_paths: Optional specific file paths to rebuild from
            priority: Rebuild priority ("high", "normal", "low")
            
        Returns:
            RebuildOperation object to track progress
        """
        logger.info(f"Starting async index rebuild for user {user_id} (priority: {priority})")
        
        # Check if user is already being rebuilt
        if user_id in self.rebuilding_users:
            logger.info(f"Rebuild already in progress for user {user_id}")
            return self.active_rebuilds[user_id]
        
        # Auto-discover file paths if not provided
        if not file_paths:
            file_paths = self._discover_user_files(user_id)
        
        # Create rebuild operation
        rebuild_op = RebuildOperation(
            user_id=user_id,
            status=RebuildStatus.PENDING,
            started_at=datetime.now(),
            file_paths=file_paths
        )
        
        # Register operation
        self.active_rebuilds[user_id] = rebuild_op
        self.rebuilding_users.add(user_id)
        
        # Start rebuild task (fire and forget)
        asyncio.create_task(self._execute_rebuild(rebuild_op))
        
        logger.info(f"Async rebuild initiated for user {user_id} with {len(file_paths)} files")
        return rebuild_op
    
    def is_user_rebuilding(self, user_id: str) -> bool:
        """Check if a user's index is currently being rebuilt."""
        return user_id in self.rebuilding_users
    
    def get_rebuild_status(self, user_id: str) -> Optional[RebuildOperation]:
        """Get current rebuild status for a user."""
        return self.active_rebuilds.get(user_id)
    
    def get_fallback_response(self, user_id: str) -> str:
        """Get a fallback response for a user whose index is being rebuilt."""
        
        self.fallback_response_count += 1
        
        # Get rebuild status for personalized message
        rebuild_op = self.active_rebuilds.get(user_id)
        
        base_response = self.fallback_responses[
            self.fallback_response_count % len(self.fallback_responses)
        ]
        
        # Add progress information if available
        if rebuild_op and rebuild_op.progress_messages:
            latest_progress = rebuild_op.progress_messages[-1]
            time_elapsed = (datetime.now() - rebuild_op.started_at).total_seconds()
            
            progress_note = f"\n\n📊 Status: {latest_progress} (Started {time_elapsed:.0f}s ago)"
            base_response += progress_note
        
        return base_response
    
    async def wait_for_rebuild_completion(self, user_id: str, timeout_seconds: int = 60) -> bool:
        """
        Wait for a rebuild to complete (for testing or critical operations).
        
        Args:
            user_id: User identifier
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if rebuild completed successfully, False if timeout/failed
        """
        if user_id not in self.rebuilding_users:
            return True  # No rebuild in progress
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            rebuild_op = self.active_rebuilds.get(user_id)
            if not rebuild_op:
                return True
            
            if rebuild_op.status == RebuildStatus.COMPLETED:
                return True
            elif rebuild_op.status in [RebuildStatus.FAILED, RebuildStatus.CANCELLED]:
                return False
            
            await asyncio.sleep(1)  # Check every second
        
        logger.warning(f"Timeout waiting for rebuild completion for user {user_id}")
        return False
    
    async def cancel_rebuild(self, user_id: str) -> bool:
        """
        Cancel an ongoing rebuild operation.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successfully cancelled, False if not possible
        """
        rebuild_op = self.active_rebuilds.get(user_id)
        if not rebuild_op:
            return False
        
        if rebuild_op.status == RebuildStatus.IN_PROGRESS:
            # Can't cancel in-progress rebuild safely, but mark for cleanup
            rebuild_op.status = RebuildStatus.CANCELLED
            logger.info(f"Marked rebuild for user {user_id} as cancelled")
            return True
        elif rebuild_op.status == RebuildStatus.PENDING:
            rebuild_op.status = RebuildStatus.CANCELLED
            rebuild_op.completed_at = datetime.now()
            self._cleanup_rebuild_operation(user_id)
            logger.info(f"Cancelled pending rebuild for user {user_id}")
            return True
        
        return False
    
    async def _execute_rebuild(self, rebuild_op: RebuildOperation):
        """Execute the actual rebuild operation asynchronously."""
        
        user_id = rebuild_op.user_id
        
        try:
            # Update status
            rebuild_op.status = RebuildStatus.IN_PROGRESS
            rebuild_op.progress_messages.append("Starting index rebuild...")
            
            logger.info(f"Executing rebuild for user {user_id}")
            
            # Clear existing index
            rebuild_op.progress_messages.append("Clearing existing index...")
            await rag_module.clear_user_index(user_id)
            
            # Build new index
            rebuild_op.progress_messages.append(f"Building index from {len(rebuild_op.file_paths)} files...")
            
            start_time = time.time()
            new_index = await rag_module.build_user_index(user_id, rebuild_op.file_paths)
            rebuild_time = time.time() - start_time
            
            if new_index:
                # Success
                rebuild_op.status = RebuildStatus.COMPLETED
                rebuild_op.completed_at = datetime.now()
                
                # Get document count
                if hasattr(new_index, 'docstore') and hasattr(new_index.docstore, 'docs'):
                    rebuild_op.document_count = len(new_index.docstore.docs)
                
                rebuild_op.progress_messages.append(f"Index rebuilt successfully with {rebuild_op.document_count or 'unknown'} documents")
                
                # Track performance
                self.rebuild_times.append(rebuild_time)
                
                logger.info(f"Successfully rebuilt index for user {user_id} in {rebuild_time:.2f}s")
                
            else:
                # Failed
                rebuild_op.status = RebuildStatus.FAILED
                rebuild_op.completed_at = datetime.now()
                rebuild_op.error_message = "Index building returned None"
                rebuild_op.progress_messages.append("Index rebuild failed")
                
                logger.error(f"Index rebuild failed for user {user_id}: returned None")
            
        except Exception as e:
            # Error occurred
            rebuild_op.status = RebuildStatus.FAILED
            rebuild_op.completed_at = datetime.now()
            rebuild_op.error_message = str(e)
            rebuild_op.progress_messages.append(f"Error: {e}")
            
            logger.error(f"Index rebuild error for user {user_id}: {e}")
            
        finally:
            # Always cleanup
            self._cleanup_rebuild_operation(user_id)
    
    def _discover_user_files(self, user_id: str) -> List[str]:
        """Discover CSV files for a user."""
        
        import os
        file_paths = []
        
        user_doc_dir = os.path.join("user_documents", user_id)
        if os.path.exists(user_doc_dir):
            csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
            file_paths = [os.path.join(user_doc_dir, f) for f in csv_files]
        
        logger.debug(f"Discovered {len(file_paths)} files for user {user_id}")
        return file_paths
    
    def _cleanup_rebuild_operation(self, user_id: str):
        """Clean up completed rebuild operation."""
        
        # Move to history
        if user_id in self.active_rebuilds:
            rebuild_op = self.active_rebuilds[user_id]
            self.rebuild_history.append(rebuild_op)
            
            # Keep only last 100 operations
            if len(self.rebuild_history) > 100:
                self.rebuild_history.pop(0)
            
            # Remove from active
            del self.active_rebuilds[user_id]
        
        # Remove from rebuilding set
        self.rebuilding_users.discard(user_id)
        
        logger.debug(f"Cleaned up rebuild operation for user {user_id}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for async operations."""
        
        active_count = len(self.active_rebuilds)
        completed_count = len([op for op in self.rebuild_history if op.status == RebuildStatus.COMPLETED])
        failed_count = len([op for op in self.rebuild_history if op.status == RebuildStatus.FAILED])
        
        avg_rebuild_time = (
            sum(self.rebuild_times) / len(self.rebuild_times) 
            if self.rebuild_times else 0
        )
        
        return {
            "active_rebuilds": active_count,
            "rebuilding_users": list(self.rebuilding_users),
            "completed_rebuilds": completed_count,
            "failed_rebuilds": failed_count,
            "total_rebuild_operations": len(self.rebuild_history),
            "avg_rebuild_time_seconds": f"{avg_rebuild_time:.2f}",
            "fallback_responses_served": self.fallback_response_count,
            "success_rate": f"{(completed_count / (completed_count + failed_count) * 100):.1f}%" if (completed_count + failed_count) > 0 else "N/A"
        }
    
    def get_user_rebuild_history(self, user_id: str, limit: int = 10) -> List[RebuildOperation]:
        """Get rebuild history for a specific user."""
        
        user_operations = [
            op for op in self.rebuild_history 
            if op.user_id == user_id
        ]
        
        # Sort by most recent first
        user_operations.sort(key=lambda op: op.started_at, reverse=True)
        
        return user_operations[:limit]
    
    async def health_check_and_auto_rebuild(self, user_id: str) -> Dict[str, Any]:
        """
        Perform health check and automatically start rebuild if needed.
        
        Returns:
            Status information about the health check and any actions taken
        """
        from app.index_health_validator import IndexHealthValidator
        
        health_validator = IndexHealthValidator()
        
        # Check health
        health_result = await health_validator.validate_user_index_health(user_id)
        
        action_taken = None
        rebuild_operation = None
        
        # Auto-rebuild if index is unhealthy and we have documents
        if not health_result.is_healthy and not self.is_user_rebuilding(user_id):
            
            # Check if we have documents to rebuild from
            file_paths = self._discover_user_files(user_id)
            if file_paths:
                rebuild_operation = await self.rebuild_user_index_async(user_id, file_paths)
                action_taken = "auto_rebuild_started"
            else:
                action_taken = "no_documents_available"
        elif self.is_user_rebuilding(user_id):
            action_taken = "rebuild_already_in_progress"
            rebuild_operation = self.get_rebuild_status(user_id)
        else:
            action_taken = "index_healthy"
        
        return {
            "user_id": user_id,
            "health_result": {
                "is_healthy": health_result.is_healthy,
                "issues_count": len(health_result.issues_found),
                "index_exists": health_result.index_exists,
                "retriever_functional": health_result.retriever_functional
            },
            "action_taken": action_taken,
            "rebuild_status": rebuild_operation.status.value if rebuild_operation else None,
            "estimated_completion": "1-2 minutes" if rebuild_operation and rebuild_operation.status == RebuildStatus.IN_PROGRESS else None
        } 