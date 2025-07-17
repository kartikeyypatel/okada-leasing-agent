# /app/recommendation_workflow_manager.py
"""
Recommendation Workflow Manager for Smart Property Recommendations

This service orchestrates the entire recommendation process from trigger to completion,
coordinating between intent detection, user context analysis, conversation management,
and property recommendation generation.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Import rag module to ensure LLM is properly configured
import app.rag
from app.models import (
    RecommendationResult, 
    WorkflowSession, 
    WorkflowStep, 
    ConversationState,
    UserContext
)
from app.intent_detection import intent_detection_service
from app.user_context_analyzer import user_context_analyzer
from app.conversation_state_manager import conversation_state_manager
from app.property_recommendation_engine import property_recommendation_engine
from app.database import get_database

logger = logging.getLogger(__name__)


class RecommendationWorkflowManager:
    """
    Service for orchestrating the complete recommendation workflow.
    
    Manages the flow from intent detection through recommendation generation,
    handling errors, state persistence, and fallback scenarios.
    """
    
    def __init__(self):
        pass
    
    async def start_recommendation_workflow(self, user_id: str, initial_message: str) -> WorkflowSession:
        """
        Start a new recommendation workflow for a user.
        
        Args:
            user_id: User's email address
            initial_message: The message that triggered the recommendation workflow
            
        Returns:
            WorkflowSession with initial state
        """
        session_id = str(uuid.uuid4())
        
        logger.info(f"Starting recommendation workflow {session_id} for user {user_id}")
        
        try:
            # Step 1: Detect intent and extract initial preferences
            intent = await intent_detection_service.detect_recommendation_intent(initial_message)
            
            # Step 2: Analyze user context
            user_context = await user_context_analyzer.analyze_user_context(user_id)
            
            # Step 3: Merge initial preferences from the message
            if intent.initial_preferences:
                user_context = await user_context_analyzer.merge_new_preferences(
                    user_id, 
                    intent.initial_preferences
                )
            
            # Step 4: Create conversation session
            conversation_session = await conversation_state_manager.create_session(
                user_id, 
                user_context
            )
            
            # Step 5: Create workflow session
            workflow_session = WorkflowSession(
                session_id=session_id,
                user_id=user_id,
                current_step="initiated",
                data={
                    "intent": intent.model_dump(),
                    "conversation_session_id": conversation_session.session_id,
                    "user_context": user_context.model_dump(),
                    "initial_message": initial_message
                }
            )
            
            # Store workflow session
            await self._store_workflow_session(workflow_session)
            
            logger.info(f"Successfully started workflow {session_id} for user {user_id}")
            return workflow_session
            
        except Exception as e:
            logger.error(f"Error starting recommendation workflow: {e}")
            # Create a minimal session to handle the error gracefully
            return WorkflowSession(
                session_id=session_id,
                user_id=user_id,
                current_step="failed",
                data={"error": str(e), "initial_message": initial_message}
            )
    
    async def process_user_response(self, session_id: str, response: str) -> WorkflowStep:
        """
        Process a user response within an active workflow.
        
        Args:
            session_id: Workflow session identifier
            response: User's response to a clarifying question
            
        Returns:
            WorkflowStep with the result of processing
        """
        logger.info(f"Processing user response for workflow {session_id}")
        
        try:
            # Get workflow session
            workflow_session = await self._get_workflow_session(session_id)
            if not workflow_session:
                return WorkflowStep(
                    step_name="error",
                    success=False,
                    response_message="Session not found. Let's start over with your property preferences.",
                    next_step=None
                )
            
            # Get conversation session ID
            conversation_session_id = workflow_session.data.get("conversation_session_id")
            if not conversation_session_id:
                return WorkflowStep(
                    step_name="error",
                    success=False,
                    response_message="Session error. Let's start fresh with your property search.",
                    next_step=None
                )
            
            # Update conversation session with user response
            conversation_session = await conversation_state_manager.update_session(
                conversation_session_id, 
                response
            )
            
            # Check if we have enough information to generate recommendations
            if await conversation_state_manager.is_conversation_complete(conversation_session):
                # Generate recommendations
                return await self._generate_recommendations_step(workflow_session)
            else:
                # Ask next clarifying question
                next_question = await conversation_state_manager.get_next_question(conversation_session)
                
                if next_question:
                    # Update workflow session
                    workflow_session.current_step = "gathering_preferences"
                    workflow_session.data["last_question"] = next_question
                    await self._store_workflow_session(workflow_session)
                    
                    return WorkflowStep(
                        step_name="clarifying_question",
                        success=True,
                        response_message=next_question,
                        next_step="gathering_preferences",
                        collected_data=conversation_session.collected_preferences
                    )
                else:
                    # No more questions, generate recommendations
                    return await self._generate_recommendations_step(workflow_session)
            
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
            return WorkflowStep(
                step_name="error",
                success=False,
                response_message="I encountered an issue processing your response. Could you please try again?",
                next_step=None
            )
    
    async def complete_workflow(self, session_id: str) -> RecommendationResult:
        """
        Complete the recommendation workflow and return final results.
        
        Args:
            session_id: Workflow session identifier
            
        Returns:
            RecommendationResult with final recommendations
        """
        logger.info(f"Completing recommendation workflow {session_id}")
        
        try:
            workflow_session = await self._get_workflow_session(session_id)
            if not workflow_session:
                raise ValueError(f"Workflow session {session_id} not found")
            
            # Get the final user context
            conversation_session_id = workflow_session.data.get("conversation_session_id")
            user_context = await user_context_analyzer.analyze_user_context(workflow_session.user_id)
            
            # Generate final recommendations
            recommendations = await property_recommendation_engine.generate_recommendations(
                user_context, 
                max_results=3
            )
            
            # Create conversation summary
            conversation_summary = await self._generate_conversation_summary(workflow_session)
            
            # Create final result
            result = RecommendationResult(
                session_id=session_id,
                recommendations=recommendations,
                user_context=user_context,
                conversation_summary=conversation_summary,
                total_properties_considered=len(recommendations),
                recommendations_generated=len(recommendations)
            )
            
            # Update workflow session as completed
            workflow_session.current_step = "completed"
            workflow_session.data["final_result"] = result.model_dump()
            await self._store_workflow_session(workflow_session)
            
            logger.info(f"Successfully completed workflow {session_id} with {len(recommendations)} recommendations")
            return result
            
        except Exception as e:
            logger.error(f"Error completing workflow: {e}")
            # Return empty result with error info
            return RecommendationResult(
                session_id=session_id,
                recommendations=[],
                user_context=UserContext(user_id=workflow_session.user_id if workflow_session else "unknown"),
                conversation_summary=f"Workflow failed: {str(e)}",
                total_properties_considered=0,
                recommendations_generated=0
            )
    
    async def get_next_step(self, session_id: str) -> Optional[WorkflowStep]:
        """
        Get the next step for an active workflow session.
        
        Args:
            session_id: Workflow session identifier
            
        Returns:
            WorkflowStep with next action or None if workflow is complete
        """
        try:
            workflow_session = await self._get_workflow_session(session_id)
            if not workflow_session:
                return None
            
            if workflow_session.current_step == "completed":
                return None
            
            # Get conversation session
            conversation_session_id = workflow_session.data.get("conversation_session_id")
            if not conversation_session_id:
                return None
            
            # Check if we need to ask a clarifying question
            user_context = await user_context_analyzer.analyze_user_context(workflow_session.user_id)
            missing_prefs = await user_context_analyzer.identify_missing_preferences(user_context)
            
            if missing_prefs and len(workflow_session.data.get("questions_asked", [])) < 3:
                # Generate next question
                next_question = await user_context_analyzer.generate_clarifying_question(
                    user_context, 
                    missing_prefs[0]
                )
                
                return WorkflowStep(
                    step_name="clarifying_question",
                    success=True,
                    response_message=next_question,
                    next_step="gathering_preferences"
                )
            else:
                # Ready for recommendations
                return WorkflowStep(
                    step_name="ready_for_recommendations",
                    success=True,
                    response_message="I have enough information to provide recommendations.",
                    next_step="generating_recommendations"
                )
                
        except Exception as e:
            logger.error(f"Error getting next step: {e}")
            return WorkflowStep(
                step_name="error",
                success=False,
                response_message="I encountered an issue. Let's try again.",
                next_step=None
            )
    
    async def _generate_recommendations_step(self, workflow_session: WorkflowSession) -> WorkflowStep:
        """Generate recommendations and return as a workflow step."""
        try:
            # Get updated user context
            user_context = await user_context_analyzer.analyze_user_context(workflow_session.user_id)
            
            # Generate recommendations
            recommendations = await property_recommendation_engine.generate_recommendations(
                user_context, 
                max_results=3
            )
            
            if recommendations:
                # Format recommendations for response
                response_parts = ["Based on your preferences, here are my top recommendations:"]
                
                for i, rec in enumerate(recommendations, 1):
                    address = rec.property_data.get('property_address', 'Property')
                    rent = rec.property_data.get('monthly_rent')
                    rent_str = f"${rent:,}/month" if rent else "Contact for pricing"
                    
                    response_parts.append(f"\n**{i}. {address}**")
                    response_parts.append(f"   • Rent: {rent_str}")
                    response_parts.append(f"   • {rec.explanation}")
                
                response_parts.append("\nWould you like more details about any of these properties or help with scheduling a viewing?")
                
                response_message = "\n".join(response_parts)
                
                # Update workflow session
                workflow_session.current_step = "completed"
                workflow_session.data["recommendations"] = [rec.model_dump() for rec in recommendations]
                await self._store_workflow_session(workflow_session)
                
                return WorkflowStep(
                    step_name="recommendations_generated",
                    success=True,
                    response_message=response_message,
                    next_step=None,
                    collected_data={"recommendations_count": len(recommendations)}
                )
            else:
                return WorkflowStep(
                    step_name="no_recommendations",
                    success=False,
                    response_message="I couldn't find any properties that match your criteria. Would you like to adjust your preferences or search requirements?",
                    next_step=None
                )
                
        except Exception as e:
            logger.error(f"Error generating recommendations step: {e}")
            return WorkflowStep(
                step_name="error",
                success=False,
                response_message="I had trouble generating recommendations. Let me try a different approach.",
                next_step=None
            )
    
    async def _generate_conversation_summary(self, workflow_session: WorkflowSession) -> str:
        """Generate a summary of the conversation and recommendations."""
        try:
            user_id = workflow_session.user_id
            initial_message = workflow_session.data.get("initial_message", "")
            recommendations_count = len(workflow_session.data.get("recommendations", []))
            
            summary = f"User {user_id} requested property recommendations with message: '{initial_message}'. "
            summary += f"After gathering preferences, generated {recommendations_count} personalized recommendations."
            
            return summary
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return "Recommendation workflow completed."
    
    async def _store_workflow_session(self, session: WorkflowSession) -> None:
        """Store workflow session in database."""
        try:
            db = get_database()
            sessions_collection = db["workflow_sessions"]
            
            session_dict = session.model_dump()
            session_dict['_id'] = session.session_id
            
            await sessions_collection.replace_one(
                {"_id": session.session_id},
                session_dict,
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error storing workflow session: {e}")
    
    async def _get_workflow_session(self, session_id: str) -> Optional[WorkflowSession]:
        """Retrieve workflow session from database."""
        try:
            db = get_database()
            sessions_collection = db["workflow_sessions"]
            
            session_data = await sessions_collection.find_one({"_id": session_id})
            if session_data:
                session_data.pop('_id', None)
                session_data['session_id'] = session_id
                return WorkflowSession(**session_data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving workflow session: {e}")
            return None
    
    async def cleanup_expired_workflows(self) -> int:
        """Clean up expired workflow sessions. Returns number of sessions cleaned."""
        try:
            db = get_database()
            sessions_collection = db["workflow_sessions"]
            
            # Delete sessions older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            result = await sessions_collection.delete_many({
                "created_at": {"$lt": cutoff_time}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} expired workflow sessions")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up expired workflows: {e}")
            return 0


# Global service instance
recommendation_workflow_manager = RecommendationWorkflowManager() 