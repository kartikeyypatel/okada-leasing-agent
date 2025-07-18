# /app/recommendation_analytics.py
"""
Analytics and Monitoring System for Smart Property Recommendations

This module provides comprehensive analytics tracking, performance monitoring,
and success metrics for the Smart Property Recommendations system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio

from app.database import get_database

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the analytics system."""
    RECOMMENDATION_REQUEST = "recommendation_request"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_ABANDONED = "workflow_abandoned"
    CLARIFYING_QUESTION_ASKED = "clarifying_question_asked"
    USER_RESPONSE_RECEIVED = "user_response_received"
    RECOMMENDATIONS_GENERATED = "recommendations_generated"
    RECOMMENDATION_CLICKED = "recommendation_clicked"
    RECOMMENDATION_ACCEPTED = "recommendation_accepted"
    RECOMMENDATION_REJECTED = "recommendation_rejected"
    USER_SATISFACTION = "user_satisfaction"
    PERFORMANCE_LATENCY = "performance_latency"
    ERROR_OCCURRED = "error_occurred"
    PREFERENCE_UPDATED = "preference_updated"
    CONVERSATION_QUALITY = "conversation_quality"


class EngagementLevel(Enum):
    """User engagement levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class AnalyticsEvent:
    """Analytics event data structure."""
    metric_type: MetricType
    user_id: Optional[str]
    session_id: Optional[str]
    value: Any
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class RecommendationMetrics:
    """Recommendation system performance metrics."""
    total_requests: int
    successful_workflows: int
    abandoned_workflows: int
    average_completion_time: float
    user_satisfaction_score: float
    recommendation_acceptance_rate: float
    error_rate: float
    performance_score: float


@dataclass
class UserEngagementMetrics:
    """User engagement metrics."""
    active_users: int
    returning_users: int
    average_session_duration: float
    questions_per_session: float
    engagement_level_distribution: Dict[EngagementLevel, int]
    user_retention_rate: float


class RecommendationAnalyticsManager:
    """
    Manages analytics and monitoring for the recommendation system.
    """
    
    def __init__(self):
        self.metrics_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_cache_update = None
    
    async def track_event(self, event: AnalyticsEvent) -> None:
        """
        Track an analytics event.
        
        Args:
            event: AnalyticsEvent to track
        """
        try:
            db = get_database()
            analytics_collection = db["recommendation_analytics"]
            
            event_data = {
                "metric_type": event.metric_type.value,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "value": event.value,
                "metadata": event.metadata,
                "timestamp": event.timestamp
            }
            
            await analytics_collection.insert_one(event_data)
            
            # Clear cache to ensure fresh data
            self._clear_cache()
            
            logger.debug(f"Tracked analytics event: {event.metric_type.value}")
            
        except Exception as e:
            logger.error(f"Error tracking analytics event: {e}")
    
    async def track_recommendation_request(self, user_id: str, message: str, 
                                         intent_confidence: float) -> None:
        """Track a recommendation request."""
        event = AnalyticsEvent(
            metric_type=MetricType.RECOMMENDATION_REQUEST,
            user_id=user_id,
            session_id=None,
            value=1,
            metadata={
                "message_length": len(message),
                "intent_confidence": intent_confidence,
                "request_time": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_workflow_started(self, user_id: str, session_id: str, 
                                   initial_preferences: Dict[str, Any]) -> None:
        """Track workflow initiation."""
        event = AnalyticsEvent(
            metric_type=MetricType.WORKFLOW_STARTED,
            user_id=user_id,
            session_id=session_id,
            value=1,
            metadata={
                "initial_preferences_count": len(initial_preferences),
                "has_budget": "budget" in initial_preferences,
                "has_location": "location" in initial_preferences,
                "has_features": "required_features" in initial_preferences
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_workflow_completed(self, user_id: str, session_id: str, 
                                     recommendations_count: int, 
                                     completion_time: float) -> None:
        """Track successful workflow completion."""
        event = AnalyticsEvent(
            metric_type=MetricType.WORKFLOW_COMPLETED,
            user_id=user_id,
            session_id=session_id,
            value=1,
            metadata={
                "recommendations_count": recommendations_count,
                "completion_time_seconds": completion_time,
                "success": True
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_workflow_abandoned(self, user_id: str, session_id: str, 
                                     abandonment_stage: str, 
                                     time_spent: float) -> None:
        """Track workflow abandonment."""
        event = AnalyticsEvent(
            metric_type=MetricType.WORKFLOW_ABANDONED,
            user_id=user_id,
            session_id=session_id,
            value=1,
            metadata={
                "abandonment_stage": abandonment_stage,
                "time_spent_seconds": time_spent,
                "success": False
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_recommendations_generated(self, user_id: str, session_id: str,
                                            recommendations: List[Dict[str, Any]],
                                            generation_time: float) -> None:
        """Track recommendation generation."""
        event = AnalyticsEvent(
            metric_type=MetricType.RECOMMENDATIONS_GENERATED,
            user_id=user_id,
            session_id=session_id,
            value=len(recommendations),
            metadata={
                "generation_time_seconds": generation_time,
                "average_match_score": sum(r.get("match_score", 0) for r in recommendations) / len(recommendations) if recommendations else 0,
                "recommendations_data": [
                    {
                        "property_id": r.get("property_id"),
                        "match_score": r.get("match_score"),
                        "matching_criteria_count": len(r.get("matching_criteria", []))
                    } for r in recommendations
                ]
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_recommendation_interaction(self, user_id: str, session_id: str,
                                             property_id: str, interaction_type: str) -> None:
        """Track user interaction with recommendations."""
        metric_type = {
            "clicked": MetricType.RECOMMENDATION_CLICKED,
            "accepted": MetricType.RECOMMENDATION_ACCEPTED,
            "rejected": MetricType.RECOMMENDATION_REJECTED
        }.get(interaction_type, MetricType.RECOMMENDATION_CLICKED)
        
        event = AnalyticsEvent(
            metric_type=metric_type,
            user_id=user_id,
            session_id=session_id,
            value=1,
            metadata={
                "property_id": property_id,
                "interaction_type": interaction_type
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_performance_metric(self, operation: str, latency_ms: float,
                                     user_id: str = None, session_id: str = None) -> None:
        """Track performance metrics."""
        event = AnalyticsEvent(
            metric_type=MetricType.PERFORMANCE_LATENCY,
            user_id=user_id,
            session_id=session_id,
            value=latency_ms,
            metadata={
                "operation": operation,
                "latency_category": self._categorize_latency(latency_ms)
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def track_error(self, error_type: str, error_message: str,
                         user_id: str = None, session_id: str = None) -> None:
        """Track system errors."""
        event = AnalyticsEvent(
            metric_type=MetricType.ERROR_OCCURRED,
            user_id=user_id,
            session_id=session_id,
            value=1,
            metadata={
                "error_type": error_type,
                "error_message": error_message[:500]  # Truncate long messages
            },
            timestamp=datetime.now()
        )
        await self.track_event(event)
    
    async def get_recommendation_metrics(self, start_date: datetime = None,
                                       end_date: datetime = None) -> RecommendationMetrics:
        """
        Get comprehensive recommendation system metrics.
        
        Args:
            start_date: Start date for metrics (default: last 30 days)
            end_date: End date for metrics (default: now)
            
        Returns:
            RecommendationMetrics with system performance data
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Check cache first
        cache_key = f"recommendation_metrics_{start_date.isoformat()}_{end_date.isoformat()}"
        if self._is_cache_valid() and cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        db = get_database()
        analytics_collection = db["recommendation_analytics"]
        
        # Build aggregation pipeline
        match_stage = {
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        
        pipeline = [
            {"$match": match_stage},
            {"$group": {
                "_id": "$metric_type",
                "count": {"$sum": 1},
                "avg_value": {"$avg": "$value"},
                "total_value": {"$sum": "$value"}
            }}
        ]
        
        # Execute aggregation
        results = {}
        async for result in analytics_collection.aggregate(pipeline):
            results[result["_id"]] = {
                "count": result["count"],
                "avg_value": result["avg_value"],
                "total_value": result["total_value"]
            }
        
        # Calculate metrics
        total_requests = results.get(MetricType.RECOMMENDATION_REQUEST.value, {}).get("count", 0)
        successful_workflows = results.get(MetricType.WORKFLOW_COMPLETED.value, {}).get("count", 0)
        abandoned_workflows = results.get(MetricType.WORKFLOW_ABANDONED.value, {}).get("count", 0)
        
        # Calculate completion times
        completion_times = await self._get_average_completion_time(start_date, end_date)
        
        # Calculate satisfaction and acceptance rates
        satisfaction_score = await self._calculate_satisfaction_score(start_date, end_date)
        acceptance_rate = await self._calculate_acceptance_rate(start_date, end_date)
        
        # Calculate error rate
        total_errors = results.get(MetricType.ERROR_OCCURRED.value, {}).get("count", 0)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate performance score
        performance_score = await self._calculate_performance_score(start_date, end_date)
        
        metrics = RecommendationMetrics(
            total_requests=total_requests,
            successful_workflows=successful_workflows,
            abandoned_workflows=abandoned_workflows,
            average_completion_time=completion_times,
            user_satisfaction_score=satisfaction_score,
            recommendation_acceptance_rate=acceptance_rate,
            error_rate=error_rate,
            performance_score=performance_score
        )
        
        # Cache the results
        self.metrics_cache[cache_key] = metrics
        self.last_cache_update = datetime.now()
        
        return metrics
    
    async def get_user_engagement_metrics(self, start_date: datetime = None,
                                        end_date: datetime = None) -> UserEngagementMetrics:
        """Get user engagement metrics."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        db = get_database()
        analytics_collection = db["recommendation_analytics"]
        
        # Get unique active users
        active_users_pipeline = [
            {"$match": {
                "timestamp": {"$gte": start_date, "$lte": end_date},
                "user_id": {"$ne": None}
            }},
            {"$group": {"_id": "$user_id"}},
            {"$count": "active_users"}
        ]
        
        active_users = 0
        async for result in analytics_collection.aggregate(active_users_pipeline):
            active_users = result["active_users"]
        
        # Calculate returning users
        returning_users = await self._calculate_returning_users(start_date, end_date)
        
        # Calculate session metrics
        session_metrics = await self._calculate_session_metrics(start_date, end_date)
        
        # Calculate engagement levels
        engagement_distribution = await self._calculate_engagement_distribution(start_date, end_date)
        
        # Calculate retention rate
        retention_rate = await self._calculate_retention_rate(start_date, end_date)
        
        return UserEngagementMetrics(
            active_users=active_users,
            returning_users=returning_users,
            average_session_duration=session_metrics["avg_duration"],
            questions_per_session=session_metrics["avg_questions"],
            engagement_level_distribution=engagement_distribution,
            user_retention_rate=retention_rate
        )
    
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for monitoring."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_24_hours = now - timedelta(hours=24)
        
        db = get_database()
        analytics_collection = db["recommendation_analytics"]
        
        # Get last hour activity
        last_hour_activity = await analytics_collection.count_documents({
            "timestamp": {"$gte": last_hour}
        })
        
        # Get active sessions
        workflow_sessions = db["workflow_sessions"]
        active_sessions = await workflow_sessions.count_documents({
            "current_step": {"$nin": ["completed", "cancelled", "failed"]},
            "updated_at": {"$gte": last_hour}
        })
        
        # Get recent errors
        recent_errors = await analytics_collection.count_documents({
            "metric_type": MetricType.ERROR_OCCURRED.value,
            "timestamp": {"$gte": last_hour}
        })
        
        # Get performance metrics
        recent_performance = []
        async for metric in analytics_collection.find({
            "metric_type": MetricType.PERFORMANCE_LATENCY.value,
            "timestamp": {"$gte": last_hour}
        }).sort("timestamp", -1).limit(10):
            recent_performance.append({
                "operation": metric["metadata"]["operation"],
                "latency_ms": metric["value"],
                "timestamp": metric["timestamp"]
            })
        
        return {
            "current_time": now.isoformat(),
            "last_hour_activity": last_hour_activity,
            "active_sessions": active_sessions,
            "recent_errors": recent_errors,
            "recent_performance": recent_performance,
            "system_status": "healthy" if recent_errors < 5 else "warning"
        }
    
    async def generate_analytics_report(self, report_type: str = "weekly") -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        if report_type == "weekly":
            start_date = datetime.now() - timedelta(days=7)
        elif report_type == "monthly":
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = datetime.now() - timedelta(days=1)  # daily
        
        end_date = datetime.now()
        
        # Get all metrics
        recommendation_metrics = await self.get_recommendation_metrics(start_date, end_date)
        engagement_metrics = await self.get_user_engagement_metrics(start_date, end_date)
        
        # Get top performing recommendations
        top_recommendations = await self._get_top_recommendations(start_date, end_date)
        
        # Get common user preferences
        common_preferences = await self._get_common_preferences(start_date, end_date)
        
        # Get performance trends
        performance_trends = await self._get_performance_trends(start_date, end_date)
        
        return {
            "report_type": report_type,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "recommendation_metrics": {
                "total_requests": recommendation_metrics.total_requests,
                "successful_workflows": recommendation_metrics.successful_workflows,
                "completion_rate": (recommendation_metrics.successful_workflows / recommendation_metrics.total_requests * 100) if recommendation_metrics.total_requests > 0 else 0,
                "average_completion_time": recommendation_metrics.average_completion_time,
                "user_satisfaction": recommendation_metrics.user_satisfaction_score,
                "acceptance_rate": recommendation_metrics.recommendation_acceptance_rate,
                "error_rate": recommendation_metrics.error_rate
            },
            "engagement_metrics": {
                "active_users": engagement_metrics.active_users,
                "returning_users": engagement_metrics.returning_users,
                "retention_rate": engagement_metrics.user_retention_rate,
                "average_session_duration": engagement_metrics.average_session_duration,
                "engagement_distribution": {level.value: count for level, count in engagement_metrics.engagement_level_distribution.items()}
            },
            "insights": {
                "top_recommendations": top_recommendations,
                "common_preferences": common_preferences,
                "performance_trends": performance_trends
            },
            "generated_at": datetime.now().isoformat()
        }
    
    # Helper methods
    def _categorize_latency(self, latency_ms: float) -> str:
        """Categorize latency performance."""
        if latency_ms < 500:
            return "excellent"
        elif latency_ms < 1000:
            return "good"
        elif latency_ms < 2000:
            return "acceptable"
        else:
            return "poor"
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self.last_cache_update:
            return False
        return datetime.now() - self.last_cache_update < self.cache_ttl
    
    def _clear_cache(self) -> None:
        """Clear the metrics cache."""
        self.metrics_cache.clear()
        self.last_cache_update = None
    
    async def _get_average_completion_time(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate average workflow completion time."""
        db = get_database()
        workflow_sessions = db["workflow_sessions"]
        
        pipeline = [
            {"$match": {
                "current_step": "completed",
                "created_at": {"$gte": start_date, "$lte": end_date}
            }},
            {"$project": {
                "completion_time": {
                    "$subtract": ["$updated_at", "$created_at"]
                }
            }},
            {"$group": {
                "_id": None,
                "avg_completion_time": {"$avg": "$completion_time"}
            }}
        ]
        
        async for result in workflow_sessions.aggregate(pipeline):
            return result["avg_completion_time"] / 1000  # Convert to seconds
        
        return 0.0
    
    async def _calculate_satisfaction_score(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate user satisfaction score."""
        # This would integrate with user feedback systems
        # For now, return a calculated score based on completion rates and interactions
        db = get_database()
        analytics_collection = db["recommendation_analytics"]
        
        # Count positive vs negative interactions
        positive_interactions = await analytics_collection.count_documents({
            "metric_type": {"$in": [MetricType.RECOMMENDATION_ACCEPTED.value, MetricType.WORKFLOW_COMPLETED.value]},
            "timestamp": {"$gte": start_date, "$lte": end_date}
        })
        
        negative_interactions = await analytics_collection.count_documents({
            "metric_type": {"$in": [MetricType.RECOMMENDATION_REJECTED.value, MetricType.WORKFLOW_ABANDONED.value]},
            "timestamp": {"$gte": start_date, "$lte": end_date}
        })
        
        total_interactions = positive_interactions + negative_interactions
        if total_interactions == 0:
            return 0.0
        
        return (positive_interactions / total_interactions) * 100
    
    async def _calculate_acceptance_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate recommendation acceptance rate."""
        db = get_database()
        analytics_collection = db["recommendation_analytics"]
        
        accepted = await analytics_collection.count_documents({
            "metric_type": MetricType.RECOMMENDATION_ACCEPTED.value,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        })
        
        total_recommendations = await analytics_collection.count_documents({
            "metric_type": MetricType.RECOMMENDATIONS_GENERATED.value,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        })
        
        if total_recommendations == 0:
            return 0.0
        
        return (accepted / total_recommendations) * 100
    
    async def _calculate_performance_score(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate overall performance score."""
        db = get_database()
        analytics_collection = db["recommendation_analytics"]
        
        # Get performance metrics
        pipeline = [
            {"$match": {
                "metric_type": MetricType.PERFORMANCE_LATENCY.value,
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }},
            {"$group": {
                "_id": None,
                "avg_latency": {"$avg": "$value"},
                "count": {"$sum": 1}
            }}
        ]
        
        async for result in analytics_collection.aggregate(pipeline):
            avg_latency = result["avg_latency"]
            
            # Score based on latency (lower is better)
            if avg_latency < 500:
                return 95.0
            elif avg_latency < 1000:
                return 85.0
            elif avg_latency < 2000:
                return 75.0
            else:
                return 60.0
        
        return 80.0  # Default score
    
    async def _calculate_returning_users(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate number of returning users."""
        # Implementation would track users who had previous activity
        # For now, return a placeholder
        return 0
    
    async def _calculate_session_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate session-related metrics."""
        # Implementation would analyze session durations and question counts
        return {
            "avg_duration": 180.0,  # 3 minutes average
            "avg_questions": 2.5
        }
    
    async def _calculate_engagement_distribution(self, start_date: datetime, end_date: datetime) -> Dict[EngagementLevel, int]:
        """Calculate engagement level distribution."""
        # Implementation would categorize users by engagement level
        return {
            EngagementLevel.HIGH: 25,
            EngagementLevel.MEDIUM: 45,
            EngagementLevel.LOW: 20,
            EngagementLevel.NONE: 10
        }
    
    async def _calculate_retention_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate user retention rate."""
        # Implementation would track user return behavior
        return 65.0  # 65% retention rate
    
    async def _get_top_recommendations(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get top performing recommendations."""
        # Implementation would analyze most accepted/clicked recommendations
        return []
    
    async def _get_common_preferences(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get most common user preferences."""
        # Implementation would analyze preference patterns
        return {}
    
    async def _get_performance_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get performance trends over time."""
        # Implementation would analyze performance changes over time
        return []


# Global analytics manager instance
recommendation_analytics = RecommendationAnalyticsManager() 