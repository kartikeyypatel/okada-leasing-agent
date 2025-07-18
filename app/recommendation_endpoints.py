# /app/recommendation_endpoints.py
"""
Smart Property Recommendations REST API Endpoints

This module provides comprehensive REST API endpoints for the Smart Property
Recommendations system including workflow management, preference handling,
and recommendation history.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.intent_detection import intent_detection_service
from app.recommendation_workflow_manager import recommendation_workflow_manager
from app.user_context_analyzer import user_context_analyzer
from app.conversation_state_manager import conversation_state_manager
from app.property_recommendation_engine import property_recommendation_engine
from app.database import get_database
from app.models import UserContext, PropertyRecommendation, RecommendationResult

logger = logging.getLogger(__name__)

# Create router for recommendation endpoints
recommendation_router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

# Request/Response Models
class StartRecommendationRequest(BaseModel):
    user_id: str
    message: str
    preferences: Optional[Dict[str, Any]] = None

class UpdatePreferencesRequest(BaseModel):
    user_id: str
    preferences: Dict[str, Any]
    merge_with_existing: bool = True

class RecommendationSessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: str
    current_step: str
    created_at: datetime
    updated_at: datetime
    progress_percentage: float
    next_action: Optional[str] = None

class RecommendationHistoryResponse(BaseModel):
    user_id: str
    total_sessions: int
    recent_sessions: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    last_recommendation_date: Optional[datetime]

@recommendation_router.post("/start", response_model=Dict[str, Any])
async def start_recommendation_workflow_endpoint(request: StartRecommendationRequest):
    """
    Start a new recommendation workflow for a user.
    
    Args:
        request: StartRecommendationRequest with user_id, message, and optional preferences
        
    Returns:
        Workflow session details and next step information
    """
    try:
        logger.info(f"Starting recommendation workflow via API for user {request.user_id}")
        
        # Start the workflow
        workflow_session = await recommendation_workflow_manager.start_recommendation_workflow(
            request.user_id,
            request.message
        )
        
        # If additional preferences provided, merge them
        if request.preferences:
            await user_context_analyzer.merge_new_preferences(
                request.user_id,
                request.preferences
            )
        
        # Get the next step
        next_step = await recommendation_workflow_manager.get_next_step(workflow_session.session_id)
        
        return {
            "success": True,
            "session_id": workflow_session.session_id,
            "user_id": workflow_session.user_id,
            "current_step": workflow_session.current_step,
            "message": next_step.response_message if next_step else "Workflow started successfully",
            "next_action": next_step.next_step if next_step else None,
            "workflow_complete": False,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting recommendation workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start recommendation workflow: {str(e)}")

@recommendation_router.get("/session/{session_id}", response_model=RecommendationSessionResponse)
async def get_recommendation_session_status(session_id: str):
    """
    Get the current status of a recommendation workflow session.
    
    Args:
        session_id: Workflow session identifier
        
    Returns:
        Current session status and progress information
    """
    try:
        db = get_database()
        
        # Get workflow session
        workflow_session = await db["workflow_sessions"].find_one({"_id": session_id})
        
        if not workflow_session:
            raise HTTPException(status_code=404, detail="Recommendation session not found")
        
        # Calculate progress percentage
        progress_map = {
            "initiated": 20,
            "gathering_preferences": 40,
            "clarifying_details": 60,
            "generating_recommendations": 80,
            "completed": 100,
            "failed": 0
        }
        
        progress = progress_map.get(workflow_session.get("current_step", "initiated"), 0)
        
        # Determine next action
        next_action = None
        if workflow_session.get("current_step") == "gathering_preferences":
            next_action = "respond_to_question"
        elif workflow_session.get("current_step") == "generating_recommendations":
            next_action = "view_recommendations"
        elif workflow_session.get("current_step") == "completed":
            next_action = "workflow_complete"
        
        return RecommendationSessionResponse(
            session_id=session_id,
            user_id=workflow_session["user_id"],
            status="active" if workflow_session.get("current_step") != "completed" else "completed",
            current_step=workflow_session.get("current_step", "initiated"),
            created_at=workflow_session.get("created_at", datetime.now()),
            updated_at=workflow_session.get("updated_at", datetime.now()),
            progress_percentage=progress,
            next_action=next_action
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@recommendation_router.post("/session/{session_id}/respond")
async def respond_to_recommendation_workflow(session_id: str, user_response: str = Body(..., embed=True)):
    """
    Respond to a clarifying question in an active recommendation workflow.
    
    Args:
        session_id: Workflow session identifier
        user_response: User's response to the clarifying question
        
    Returns:
        Next step in the workflow or final recommendations
    """
    try:
        logger.info(f"Processing workflow response for session {session_id}")
        
        # Process the user response
        workflow_step = await recommendation_workflow_manager.process_user_response(session_id, user_response)
        
        if not workflow_step.success:
            return {
                "success": False,
                "message": workflow_step.response_message,
                "error": "Workflow processing failed"
            }
        
        # Check if workflow is complete
        if workflow_step.step_name == "recommendations_generated":
            # Get final recommendations
            recommendation_result = await recommendation_workflow_manager.complete_workflow(session_id)
            
            return {
                "success": True,
                "workflow_complete": True,
                "message": workflow_step.response_message,
                "recommendations": [rec.model_dump() for rec in recommendation_result.recommendations],
                "session_summary": recommendation_result.conversation_summary
            }
        else:
            return {
                "success": True,
                "workflow_complete": False,
                "message": workflow_step.response_message,
                "next_step": workflow_step.next_step,
                "collected_data": workflow_step.collected_data
            }
        
    except Exception as e:
        logger.error(f"Error processing workflow response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}")

@recommendation_router.put("/preferences", response_model=Dict[str, Any])
async def update_user_preferences(request: UpdatePreferencesRequest):
    """
    Update user preferences for property recommendations.
    
    Args:
        request: UpdatePreferencesRequest with user_id, preferences, and merge option
        
    Returns:
        Updated user context and preferences
    """
    try:
        logger.info(f"Updating preferences for user {request.user_id}")
        
        if request.merge_with_existing:
            # Merge with existing preferences
            updated_context = await user_context_analyzer.merge_new_preferences(
                request.user_id,
                request.preferences
            )
        else:
            # Replace preferences
            from app.crm import get_user_by_email, create_or_update_user
            
            user_profile = await get_user_by_email(request.user_id)
            if user_profile:
                user_profile.recommendation_preferences = request.preferences
                await create_or_update_user(user_profile)
            
            updated_context = await user_context_analyzer.analyze_user_context(request.user_id)
        
        return {
            "success": True,
            "user_id": request.user_id,
            "updated_preferences": updated_context.historical_preferences,
            "budget_range": updated_context.budget_range,
            "preferred_locations": updated_context.preferred_locations,
            "required_features": updated_context.required_features,
            "excluded_features": updated_context.excluded_features,
            "last_updated": updated_context.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")

@recommendation_router.get("/preferences/{user_id}", response_model=Dict[str, Any])
async def get_user_preferences(user_id: str):
    """
    Get current user preferences and context for recommendations.
    
    Args:
        user_id: User identifier
        
    Returns:
        Current user context and preferences
    """
    try:
        user_context = await user_context_analyzer.analyze_user_context(user_id)
        
        return {
            "user_id": user_id,
            "preferences": user_context.historical_preferences,
            "budget_range": user_context.budget_range,
            "preferred_locations": user_context.preferred_locations,
            "required_features": user_context.required_features,
            "excluded_features": user_context.excluded_features,
            "last_updated": user_context.last_updated.isoformat(),
            "preference_completeness": await user_context_analyzer.calculate_preference_completeness(user_context)
        }
        
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")

@recommendation_router.get("/history/{user_id}", response_model=RecommendationHistoryResponse)
async def get_recommendation_history(user_id: str, limit: int = Query(10, ge=1, le=50)):
    """
    Get recommendation history for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of recent sessions to return
        
    Returns:
        User's recommendation history and statistics
    """
    try:
        db = get_database()
        
        # Get recent workflow sessions
        recent_sessions_cursor = db["workflow_sessions"].find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit)
        
        recent_sessions = []
        async for session in recent_sessions_cursor:
            session_data = {
                "session_id": session["_id"],
                "created_at": session.get("created_at"),
                "current_step": session.get("current_step"),
                "status": "completed" if session.get("current_step") == "completed" else "active",
                "recommendations_count": len(session.get("data", {}).get("recommendations", []))
            }
            recent_sessions.append(session_data)
        
        # Get total session count
        total_sessions = await db["workflow_sessions"].count_documents({"user_id": user_id})
        
        # Get user preferences
        user_context = await user_context_analyzer.analyze_user_context(user_id)
        
        # Get last recommendation date
        last_session = await db["workflow_sessions"].find_one(
            {"user_id": user_id, "current_step": "completed"},
            sort=[("created_at", -1)]
        )
        
        last_recommendation_date = None
        if last_session:
            last_recommendation_date = last_session.get("created_at")
        
        return RecommendationHistoryResponse(
            user_id=user_id,
            total_sessions=total_sessions,
            recent_sessions=recent_sessions,
            user_preferences=user_context.historical_preferences,
            last_recommendation_date=last_recommendation_date
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation history: {str(e)}")

@recommendation_router.post("/generate-direct", response_model=Dict[str, Any])
async def generate_direct_recommendations(user_id: str = Body(..., embed=True), max_results: int = Body(3, embed=True)):
    """
    Generate recommendations directly without going through the conversation workflow.
    
    Args:
        user_id: User identifier
        max_results: Maximum number of recommendations to return
        
    Returns:
        Direct property recommendations based on stored preferences
    """
    try:
        logger.info(f"Generating direct recommendations for user {user_id}")
        
        # Get user context
        user_context = await user_context_analyzer.analyze_user_context(user_id)
        
        # Check if user has sufficient preferences
        missing_prefs = await user_context_analyzer.identify_missing_preferences(user_context)
        
        if len(missing_prefs) > 2:
            return {
                "success": False,
                "message": "Insufficient user preferences for direct recommendations. Please use the conversation workflow to gather more information.",
                "missing_preferences": missing_prefs,
                "recommendation_workflow_suggested": True
            }
        
        # Generate recommendations
        recommendations = await property_recommendation_engine.generate_recommendations(
            user_context,
            max_results=max_results
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "recommendations": [rec.model_dump() for rec in recommendations],
            "user_context": {
                "budget_range": user_context.budget_range,
                "preferred_locations": user_context.preferred_locations,
                "required_features": user_context.required_features
            },
            "generated_at": datetime.now().isoformat(),
            "recommendation_count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error generating direct recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@recommendation_router.post("/test-intent", response_model=Dict[str, Any])
async def test_recommendation_intent(message: str = Body(..., embed=True)):
    """
    Test recommendation intent detection for a message.
    
    Args:
        message: Message to test for recommendation intent
        
    Returns:
        Intent detection results and confidence scores
    """
    try:
        intent = await intent_detection_service.detect_recommendation_intent(message)
        
        return {
            "message": message,
            "is_recommendation_request": intent.is_recommendation_request,
            "confidence": intent.confidence,
            "initial_preferences": intent.initial_preferences,
            "trigger_phrases": intent.trigger_phrases,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing intent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test intent: {str(e)}")

@recommendation_router.delete("/session/{session_id}")
async def cancel_recommendation_session(session_id: str):
    """
    Cancel an active recommendation workflow session.
    
    Args:
        session_id: Workflow session identifier
        
    Returns:
        Cancellation confirmation
    """
    try:
        db = get_database()
        
        # Update session status to cancelled
        result = await db["workflow_sessions"].update_one(
            {"_id": session_id},
            {"$set": {
                "current_step": "cancelled",
                "updated_at": datetime.now(),
                "cancelled_at": datetime.now()
            }}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Recommendation session not found")
        
        return {
            "success": True,
            "session_id": session_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel session: {str(e)}")

@recommendation_router.get("/analytics/summary")
async def get_recommendation_analytics_summary():
    """
    Get summary analytics for the recommendation system.
    
    Returns:
        System-wide recommendation analytics and statistics
    """
    try:
        from app.recommendation_analytics import recommendation_analytics
        
        # Get metrics for the last 30 days
        metrics = await recommendation_analytics.get_recommendation_metrics()
        engagement = await recommendation_analytics.get_user_engagement_metrics()
        
        return {
            "success": True,
            "analytics_summary": {
                "recommendation_metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_workflows": metrics.successful_workflows,
                    "completion_rate": (metrics.successful_workflows / metrics.total_requests * 100) if metrics.total_requests > 0 else 0,
                    "average_completion_time": metrics.average_completion_time,
                    "user_satisfaction_score": metrics.user_satisfaction_score,
                    "recommendation_acceptance_rate": metrics.recommendation_acceptance_rate,
                    "error_rate": metrics.error_rate,
                    "performance_score": metrics.performance_score
                },
                "engagement_metrics": {
                    "active_users": engagement.active_users,
                    "returning_users": engagement.returning_users,
                    "user_retention_rate": engagement.user_retention_rate,
                    "average_session_duration": engagement.average_session_duration,
                    "questions_per_session": engagement.questions_per_session,
                    "engagement_distribution": {level.value: count for level, count in engagement.engagement_level_distribution.items()}
                }
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@recommendation_router.get("/analytics/dashboard")
async def get_real_time_dashboard():
    """
    Get real-time dashboard data for monitoring.
    
    Returns:
        Real-time system status and metrics
    """
    try:
        from app.recommendation_analytics import recommendation_analytics
        
        dashboard_data = await recommendation_analytics.get_real_time_dashboard_data()
        
        return {
            "success": True,
            "dashboard_data": dashboard_data
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@recommendation_router.get("/analytics/report/{report_type}")
async def generate_analytics_report(report_type: str):
    """
    Generate comprehensive analytics report.
    
    Args:
        report_type: Type of report (daily, weekly, monthly)
        
    Returns:
        Comprehensive analytics report
    """
    try:
        if report_type not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="Invalid report type. Must be daily, weekly, or monthly.")
        
        from app.recommendation_analytics import recommendation_analytics
        
        report = await recommendation_analytics.generate_analytics_report(report_type)
        
        return {
            "success": True,
            "report": report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@recommendation_router.post("/analytics/track-interaction")
async def track_recommendation_interaction(
    user_id: str = Body(..., embed=True),
    session_id: str = Body(..., embed=True),
    property_id: str = Body(..., embed=True),
    interaction_type: str = Body(..., embed=True)
):
    """
    Track user interaction with recommendations.
    
    Args:
        user_id: User identifier
        session_id: Workflow session identifier
        property_id: Property that was interacted with
        interaction_type: Type of interaction (clicked, accepted, rejected)
        
    Returns:
        Tracking confirmation
    """
    try:
        if interaction_type not in ["clicked", "accepted", "rejected"]:
            raise HTTPException(status_code=400, detail="Invalid interaction type")
        
        from app.recommendation_analytics import recommendation_analytics
        
        await recommendation_analytics.track_recommendation_interaction(
            user_id=user_id,
            session_id=session_id,
            property_id=property_id,
            interaction_type=interaction_type
        )
        
        return {
            "success": True,
            "message": f"Tracked {interaction_type} interaction for property {property_id}",
            "tracked_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to track interaction: {str(e)}")

@recommendation_router.get("/analytics/user/{user_id}/metrics")
async def get_user_specific_metrics(user_id: str, days: int = Query(30, ge=1, le=365)):
    """
    Get analytics metrics for a specific user.
    
    Args:
        user_id: User identifier
        days: Number of days to look back
        
    Returns:
        User-specific analytics metrics
    """
    try:
        db = get_database()
        start_date = datetime.now() - timedelta(days=days)
        
        # Get user's workflow sessions
        workflow_sessions = []
        async for session in db["workflow_sessions"].find({
            "user_id": user_id,
            "created_at": {"$gte": start_date}
        }).sort("created_at", -1):
            workflow_sessions.append({
                "session_id": session["_id"],
                "created_at": session.get("created_at"),
                "current_step": session.get("current_step"),
                "status": "completed" if session.get("current_step") == "completed" else "active"
            })
        
        # Get user's analytics events
        analytics_events = []
        async for event in db["recommendation_analytics"].find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1).limit(50):
            analytics_events.append({
                "metric_type": event["metric_type"],
                "value": event["value"],
                "timestamp": event["timestamp"],
                "metadata": event.get("metadata", {})
            })
        
        # Calculate user-specific metrics
        total_sessions = len(workflow_sessions)
        completed_sessions = len([s for s in workflow_sessions if s["status"] == "completed"])
        completion_rate = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        return {
            "success": True,
            "user_id": user_id,
            "period_days": days,
            "metrics": {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "completion_rate": completion_rate,
                "recent_sessions": workflow_sessions[:10],
                "recent_events": analytics_events[:20]
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting user metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user metrics: {str(e)}") 