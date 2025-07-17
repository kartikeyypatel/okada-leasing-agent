# /app/appointment_error_handler.py
"""
Appointment Error Handler

This service provides comprehensive error handling and recovery mechanisms
for the appointment booking system with graceful degradation strategies.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from app.models import AppointmentError, AppointmentData, WorkflowResponse

logger = logging.getLogger(__name__)


class AppointmentErrorType(str, Enum):
    """Types of appointment booking errors."""
    INTENT_DETECTION_ERROR = "intent_detection_error"
    WORKFLOW_START_ERROR = "workflow_start_error"
    SESSION_NOT_FOUND = "session_not_found"
    INFORMATION_EXTRACTION_ERROR = "information_extraction_error"
    VALIDATION_ERROR = "validation_error"
    CALENDAR_SERVICE_ERROR = "calendar_service_error"
    EMAIL_SERVICE_ERROR = "email_service_error"
    GOOGLE_MEET_ERROR = "google_meet_error"
    DATABASE_ERROR = "database_error"
    CONFIRMATION_ERROR = "confirmation_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class AppointmentErrorHandler:
    """
    Service for handling appointment booking errors with graceful degradation.
    
    Provides comprehensive error recovery strategies, fallback mechanisms,
    and user-friendly error messages for various failure scenarios.
    """
    
    def __init__(self):
        # Error recovery strategies
        self.recovery_strategies = {
            AppointmentErrorType.INTENT_DETECTION_ERROR: self._handle_intent_error,
            AppointmentErrorType.WORKFLOW_START_ERROR: self._handle_workflow_start_error,
            AppointmentErrorType.SESSION_NOT_FOUND: self._handle_session_error,
            AppointmentErrorType.INFORMATION_EXTRACTION_ERROR: self._handle_extraction_error,
            AppointmentErrorType.VALIDATION_ERROR: self._handle_validation_error,
            AppointmentErrorType.CALENDAR_SERVICE_ERROR: self._handle_calendar_error,
            AppointmentErrorType.EMAIL_SERVICE_ERROR: self._handle_email_error,
            AppointmentErrorType.GOOGLE_MEET_ERROR: self._handle_meet_error,
            AppointmentErrorType.DATABASE_ERROR: self._handle_database_error,
            AppointmentErrorType.CONFIRMATION_ERROR: self._handle_confirmation_error,
            AppointmentErrorType.NETWORK_ERROR: self._handle_network_error,
            AppointmentErrorType.UNKNOWN_ERROR: self._handle_unknown_error,
        }
    
    async def handle_error(self, error_type: str, error_message: str, 
                          context: Dict[str, Any] = None) -> WorkflowResponse:
        """
        Handle an appointment booking error with appropriate recovery strategy.
        
        Args:
            error_type: Type of error that occurred
            error_message: Error message details
            context: Additional context for error recovery
            
        Returns:
            WorkflowResponse with recovery action
        """
        logger.warning(f"Handling appointment error: {error_type} - {error_message}")
        
        try:
            # Normalize error type
            normalized_type = self._normalize_error_type(error_type)
            
            # Get recovery strategy
            recovery_strategy = self.recovery_strategies.get(
                normalized_type, 
                self._handle_unknown_error
            )
            
            # Execute recovery strategy
            response = await recovery_strategy(error_message, context or {})
            
            # Log recovery action
            logger.info(f"Error recovery executed: {normalized_type} -> {response.step_name}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return await self._handle_unknown_error(str(e), {})
    
    async def handle_partial_failure(self, appointment_data: AppointmentData, 
                                   failed_services: List[str]) -> WorkflowResponse:
        """
        Handle partial failures where some services succeed but others fail.
        
        Args:
            appointment_data: Appointment that was partially created
            failed_services: List of services that failed
            
        Returns:
            WorkflowResponse with partial success information
        """
        logger.warning(f"Handling partial failure: {failed_services}")
        
        success_services = []
        fallback_instructions = []
        
        # Determine what succeeded
        if "calendar" not in failed_services:
            success_services.append("Calendar event created")
        else:
            fallback_instructions.append("• Create calendar event manually")
        
        if "email" not in failed_services:
            success_services.append("Email notifications sent")
        else:
            fallback_instructions.append("• Send calendar invites manually")
        
        if "google_meet" not in failed_services:
            success_services.append("Google Meet link created")
        else:
            fallback_instructions.append("• Add Google Meet to calendar event")
        
        # Create partial success message
        message_parts = []
        
        if success_services:
            message_parts.append("✅ **Appointment Partially Confirmed**\n")
            message_parts.append("Successfully completed:")
            for service in success_services:
                message_parts.append(f"• {service}")
            message_parts.append("")
        
        if fallback_instructions:
            message_parts.append("⚠️ **Manual Steps Required:**")
            message_parts.extend(fallback_instructions)
            message_parts.append("")
        
        message_parts.extend([
            f"**Appointment Details:**",
            f"📋 {appointment_data.title}",
            f"📍 {appointment_data.location}",
            f"🕐 {appointment_data.date.strftime('%A, %B %d, %Y at %I:%M %p')}",
            "",
            "Sorry for the inconvenience! Most of your appointment was set up successfully."
        ])
        
        return WorkflowResponse(
            success=True,  # Partial success is still success
            message="\n".join(message_parts),
            step_name="partial_confirmation",
            appointment_data=appointment_data
        )
    
    def _normalize_error_type(self, error_type: str) -> AppointmentErrorType:
        """Normalize error type string to enum value."""
        try:
            return AppointmentErrorType(error_type.lower())
        except ValueError:
            # Check for partial matches
            error_lower = error_type.lower()
            
            if "intent" in error_lower or "detection" in error_lower:
                return AppointmentErrorType.INTENT_DETECTION_ERROR
            elif "session" in error_lower:
                return AppointmentErrorType.SESSION_NOT_FOUND
            elif "calendar" in error_lower:
                return AppointmentErrorType.CALENDAR_SERVICE_ERROR
            elif "email" in error_lower:
                return AppointmentErrorType.EMAIL_SERVICE_ERROR
            elif "meet" in error_lower:
                return AppointmentErrorType.GOOGLE_MEET_ERROR
            elif "database" in error_lower or "db" in error_lower:
                return AppointmentErrorType.DATABASE_ERROR
            elif "network" in error_lower or "connection" in error_lower:
                return AppointmentErrorType.NETWORK_ERROR
            else:
                return AppointmentErrorType.UNKNOWN_ERROR
    
    async def _handle_intent_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle intent detection errors."""
        return WorkflowResponse(
            success=False,
            message="""I'm having trouble understanding your appointment request. Could you please try rephrasing it?

Here are some examples that work well:
• "I want to book an appointment"
• "Can we schedule a meeting for tomorrow?"
• "I'd like to set up a call at 2pm"

What kind of appointment would you like to schedule?""",
            step_name="intent_recovery",
            error_details=AppointmentError(
                error_type="INTENT_DETECTION_ERROR",
                message=error_message,
                recovery_action="Prompt user to rephrase request",
                user_message="Please rephrase your appointment request"
            )
        )
    
    async def _handle_workflow_start_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle workflow start errors."""
        return WorkflowResponse(
            success=False,
            message="""I encountered an issue starting the appointment booking process. Let me try again.

Please tell me:
• What type of meeting you'd like to schedule
• When you'd prefer to meet
• Where the meeting should take place

I'm here to help you get this appointment set up!""",
            step_name="workflow_restart",
            error_details=AppointmentError(
                error_type="WORKFLOW_START_ERROR",
                message=error_message,
                recovery_action="Restart appointment booking process",
                user_message="Let's start the appointment booking again"
            )
        )
    
    async def _handle_session_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle session not found errors."""
        return WorkflowResponse(
            success=False,
            message="""It looks like your appointment booking session has expired or been lost. No worries - let's start fresh!

Please tell me what kind of appointment you'd like to schedule, and I'll help you set it up right away.""",
            step_name="session_recovery",
            error_details=AppointmentError(
                error_type="SESSION_NOT_FOUND",
                message=error_message,
                recovery_action="Start new appointment booking session",
                user_message="Start a new appointment booking"
            )
        )
    
    async def _handle_extraction_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle information extraction errors."""
        return WorkflowResponse(
            success=True,  # Continue workflow
            message="""I need a bit more information to set up your appointment. Could you please provide:

• The location or address for the meeting
• Your preferred date and time
• Any specific details about the meeting

What would work best for you?""",
            step_name="information_recovery",
            next_step="collect_location"
        )
    
    async def _handle_validation_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle validation errors."""
        return WorkflowResponse(
            success=True,  # Continue workflow
            message=f"""There's an issue with some of the appointment details. {error_message}

Please provide the correct information, and I'll update your appointment accordingly.""",
            step_name="validation_recovery",
            next_step="collect_missing_info"
        )
    
    async def _handle_calendar_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle calendar service errors."""
        appointment_data = context.get('appointment_data')
        if appointment_data:
            formatted_date = appointment_data.date.strftime('%A, %B %d, %Y at %I:%M %p')
            
            return WorkflowResponse(
                success=True,  # Partial success
                message=f"""Your appointment details have been confirmed, but I couldn't automatically create the calendar event.

**Please manually add to your calendar:**
📋 {appointment_data.title}
📍 {appointment_data.location}
🕐 {formatted_date}
⏱️ {appointment_data.duration_minutes} minutes

**To create Google Meet:**
1. Open Google Calendar
2. Create an event with the above details
3. Click "Add Google Meet video conferencing"
4. Invite: {', '.join(appointment_data.attendee_emails)}

Sorry for the inconvenience!""",
                step_name="calendar_fallback",
                appointment_data=appointment_data
            )
        else:
            return WorkflowResponse(
                success=False,
                message="I'm having trouble with the calendar service right now. Could you try booking your appointment again in a few minutes?",
                step_name="calendar_error"
            )
    
    async def _handle_email_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle email service errors."""
        return WorkflowResponse(
            success=True,  # Appointment still created
            message="""✅ Your appointment has been confirmed and added to the calendar!

⚠️ **Note:** I wasn't able to send email confirmations automatically. Please make sure to:
• Check your calendar for the event details
• Manually invite any attendees
• Share the Google Meet link with participants

The appointment is all set up - just the notifications need manual handling.""",
            step_name="email_fallback"
        )
    
    async def _handle_meet_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle Google Meet creation errors."""
        return WorkflowResponse(
            success=True,  # Appointment still created
            message="""✅ Your appointment has been confirmed and calendar invites sent!

⚠️ **Google Meet Setup:** I couldn't automatically create a Google Meet link. To add video conferencing:
1. Open the calendar event
2. Click "Add Google Meet video conferencing"
3. Save and send updates to attendees

Everything else is ready to go!""",
            step_name="meet_fallback"
        )
    
    async def _handle_database_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle database errors."""
        return WorkflowResponse(
            success=False,
            message="""I'm experiencing some technical difficulties saving your appointment information. 

Please try again in a moment, or if the issue persists, you can:
• Book the appointment manually in your calendar
• Contact support for assistance

Sorry for the inconvenience!""",
            step_name="database_error_recovery",
            error_details=AppointmentError(
                error_type="DATABASE_ERROR",
                message=error_message,
                recovery_action="Retry or manual booking",
                user_message="Technical issue - please try again"
            )
        )
    
    async def _handle_confirmation_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle confirmation errors."""
        return WorkflowResponse(
            success=False,
            message="""There was an issue confirming your appointment. Let me try to help you complete the booking process.

Please confirm you want to schedule:
• The meeting we discussed
• At the time and location specified
• With the attendees mentioned

Would you like me to try confirming again?""",
            step_name="confirmation_recovery"
        )
    
    async def _handle_network_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle network connectivity errors."""
        return WorkflowResponse(
            success=False,
            message="""I'm having trouble connecting to the appointment services right now. This might be a temporary network issue.

Please try again in a few minutes. If the problem continues, you can:
• Book the appointment manually in your calendar
• Try again later when connectivity is restored

Sorry for the inconvenience!""",
            step_name="network_error_recovery",
            error_details=AppointmentError(
                error_type="NETWORK_ERROR",
                message=error_message,
                recovery_action="Retry later or manual booking",
                user_message="Connectivity issue - please try again"
            )
        )
    
    async def _handle_unknown_error(self, error_message: str, context: Dict[str, Any]) -> WorkflowResponse:
        """Handle unknown or unexpected errors."""
        return WorkflowResponse(
            success=False,
            message="""I encountered an unexpected issue while processing your appointment request. 

Let's start fresh - please tell me:
• What type of appointment you'd like to schedule
• When you'd prefer to meet
• Where the meeting should take place

I'll do my best to help you get this appointment set up!""",
            step_name="unknown_error_recovery",
            error_details=AppointmentError(
                error_type="UNKNOWN_ERROR",
                message=error_message,
                recovery_action="Restart appointment booking",
                user_message="Unexpected error - starting fresh"
            )
        )


# Global service instance
appointment_error_handler = AppointmentErrorHandler() 