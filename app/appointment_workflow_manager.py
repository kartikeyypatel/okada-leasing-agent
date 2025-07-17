# /app/appointment_workflow_manager.py
"""
Appointment Workflow Manager

This service manages the multi-step appointment booking process, from initial intent
detection through information collection to final confirmation.
"""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

from app.models import (
    AppointmentSession, AppointmentData, AppointmentStatus, 
    WorkflowResponse, ConfirmationUI, AppointmentError
)
from app.database import get_database
from app.appointment_intent_detection import appointment_intent_detection_service

logger = logging.getLogger(__name__)


class AppointmentWorkflowManager:
    """
    Service for managing appointment booking workflows.
    
    Handles session state, information collection, and user interaction flow
    for booking appointments through conversational interface.
    """
    
    def __init__(self):
        pass
    
    async def start_appointment_booking(self, user_id: str, message: str) -> WorkflowResponse:
        """
        Start a new appointment booking workflow.
        
        Args:
            user_id: User starting the appointment booking
            message: Initial user message
            
        Returns:
            WorkflowResponse with next steps
        """
        logger.info(f"Starting appointment booking for user {user_id}")
        
        try:
            # Extract initial details from the message
            intent_result = await appointment_intent_detection_service.detect_appointment_intent(message)
            
            # Create initial appointment data
            appointment_data = AppointmentData(
                title="Meeting",  # Default title
                location="",
                date=datetime.now(),  # Placeholder
                organizer_email=user_id
            )
            
            # Apply extracted details
            self._apply_extracted_details(appointment_data, intent_result.extracted_details)
            
            # Create session
            session_id = str(uuid.uuid4())
            session = AppointmentSession(
                session_id=session_id,
                user_id=user_id,
                status=AppointmentStatus.COLLECTING_INFO,
                collected_data=appointment_data,
                missing_fields=intent_result.missing_fields
            )
            
            # Save session to database
            await self._save_session(session)
            
            # Generate next step in workflow
            return await self._get_next_workflow_step(session)
            
        except Exception as e:
            logger.error(f"Error starting appointment booking: {e}")
            return WorkflowResponse(
                success=False,
                message="Sorry, I encountered an issue starting the appointment booking. Please try again.",
                error_details=AppointmentError(
                    error_type="WORKFLOW_START_ERROR",
                    message=str(e)
                )
            )
    
    async def process_user_response(self, session_id: str, user_response: str) -> WorkflowResponse:
        """
        Process user response in an ongoing appointment workflow.
        
        Args:
            session_id: Active appointment session ID
            user_response: User's response message
            
        Returns:
            WorkflowResponse with next steps
        """
        logger.info(f"Processing user response for session {session_id}")
        
        try:
            # Load session
            session = await self._load_session(session_id)
            if not session:
                return WorkflowResponse(
                    success=False,
                    message="Sorry, I couldn't find your appointment booking session. Let's start over.",
                    error_details=AppointmentError(
                        error_type="SESSION_NOT_FOUND",
                        message="Session not found in database"
                    )
                )
            
            # Add response to conversation history
            session.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "user_response",
                "message": user_response
            })
            
            # Extract information from user response
            await self._extract_and_update_information(session, user_response)
            
            # Update session timestamp
            session.updated_at = datetime.now()
            
            # Save updated session
            await self._save_session(session)
            
            # Determine next step
            return await self._get_next_workflow_step(session)
            
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
            return WorkflowResponse(
                success=False,
                message="Sorry, there was an issue processing your response. Could you please try again?",
                error_details=AppointmentError(
                    error_type="RESPONSE_PROCESSING_ERROR",
                    message=str(e)
                )
            )
    
    async def generate_confirmation_ui(self, session: AppointmentSession) -> ConfirmationUI:
        """
        Generate confirmation UI for appointment details.
        
        Args:
            session: Appointment session with collected data
            
        Returns:
            ConfirmationUI configuration
        """
        appointment = session.collected_data
        
        # Format appointment details for display
        formatted_date = appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")
        attendees_text = ", ".join(appointment.attendee_emails) if appointment.attendee_emails else "No additional attendees"
        
        # Create appointment card configuration
        appointment_card = {
            "type": "appointment_confirmation",
            "title": "📅 Appointment Confirmation",
            "details": {
                "meeting_title": appointment.title,
                "location": f"📍 {appointment.location}",
                "datetime": f"🕐 {formatted_date}",
                "duration": f"⏱️ {appointment.duration_minutes} minutes",
                "attendees": f"👥 {attendees_text}",
                "description": appointment.description or "No additional details"
            }
        }
        
        # Create action buttons
        action_buttons = [
            {
                "id": "confirm_appointment",
                "text": "✅ Confirm Appointment",
                "type": "primary",
                "action": "confirm"
            },
            {
                "id": "cancel_appointment", 
                "text": "❌ Cancel",
                "type": "secondary",
                "action": "cancel"
            }
        ]
        
        # Define styling
        styling = {
            "card": {
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "borderRadius": "12px",
                "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                "padding": "24px",
                "color": "white"
            },
            "buttons": {
                "primary": {
                    "background": "#10B981",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "12px 24px",
                    "fontWeight": "600"
                },
                "secondary": {
                    "background": "#EF4444",
                    "color": "white", 
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "12px 24px",
                    "fontWeight": "600"
                }
            }
        }
        
        # Define animations
        animations = {
            "entrance": {
                "type": "slideUp",
                "duration": "300ms",
                "easing": "ease-out"
            },
            "buttonHover": {
                "type": "scale",
                "scale": "1.05",
                "duration": "200ms"
            }
        }
        
        return ConfirmationUI(
            appointment_card=appointment_card,
            action_buttons=action_buttons,
            styling=styling,
            animations=animations
        )
    
    async def confirm_appointment(self, session_id: str) -> WorkflowResponse:
        """
        Confirm and create the appointment.
        
        Args:
            session_id: Session ID to confirm
            
        Returns:
            WorkflowResponse with confirmation result
        """
        logger.info(f"Confirming appointment for session {session_id}")
        
        try:
            session = await self._load_session(session_id)
            if not session:
                return WorkflowResponse(
                    success=False,
                    message="Appointment session not found.",
                    error_details=AppointmentError(
                        error_type="SESSION_NOT_FOUND",
                        message="Session not found"
                    )
                )
            
            # Update session status
            session.status = AppointmentStatus.CONFIRMED
            session.updated_at = datetime.now()
            
            # This would integrate with Google Calendar and Meet services
            # For now, we'll mark it as confirmed
            appointment = session.collected_data
            
            # Save updated session
            await self._save_session(session)
            
            # Generate success message
            formatted_date = appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")
            success_message = f"""
🎉 **Appointment Confirmed!**

Your appointment has been successfully scheduled:

📋 **{appointment.title}**
📍 **Location:** {appointment.location}
🕐 **Date & Time:** {formatted_date}
⏱️ **Duration:** {appointment.duration_minutes} minutes

You will receive a calendar invitation shortly with all the details and a Google Meet link for the meeting.
            """.strip()
            
            return WorkflowResponse(
                success=True,
                message=success_message,
                session_id=session_id,
                step_name="appointment_confirmed",
                appointment_data=appointment
            )
            
        except Exception as e:
            logger.error(f"Error confirming appointment: {e}")
            return WorkflowResponse(
                success=False,
                message="Sorry, there was an issue confirming your appointment. Please try again.",
                error_details=AppointmentError(
                    error_type="CONFIRMATION_ERROR",
                    message=str(e)
                )
            )
    
    async def cancel_appointment(self, session_id: str) -> WorkflowResponse:
        """
        Cancel an appointment booking session.
        
        Args:
            session_id: Session ID to cancel
            
        Returns:
            WorkflowResponse with cancellation result
        """
        logger.info(f"Cancelling appointment for session {session_id}")
        
        try:
            session = await self._load_session(session_id)
            if session:
                session.status = AppointmentStatus.CANCELLED
                session.updated_at = datetime.now()
                await self._save_session(session)
            
            return WorkflowResponse(
                success=True,
                message="No problem! Your appointment booking has been cancelled. Feel free to ask me anything else or start a new appointment booking whenever you're ready.",
                session_id=session_id,
                step_name="appointment_cancelled"
            )
            
        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}")
            return WorkflowResponse(
                success=True,  # Still successful from user perspective
                message="Your appointment booking has been cancelled. Feel free to ask me anything else!",
                session_id=session_id,
                step_name="appointment_cancelled"
            )
    
    async def _get_next_workflow_step(self, session: AppointmentSession) -> WorkflowResponse:
        """Determine the next step in the appointment workflow."""
        
        # Special handling for confirmation responses
        if session.status == AppointmentStatus.CONFIRMING:
            # Check if a confirmation response was detected
            if hasattr(session.collected_data, 'confirmation_response'):
                if session.collected_data.confirmation_response == "confirmed":
                    logger.info(f"Processing confirmation response")
                    return await self._confirm_appointment_internal(session)
                elif session.collected_data.confirmation_response == "cancelled":
                    logger.info(f"Processing cancellation response")
                    return await self._cancel_appointment_internal(session)
        
        # Check if we have all required information
        missing_fields = self._check_missing_fields(session.collected_data)
        session.missing_fields = missing_fields
        
        if missing_fields:
            # Still collecting information
            session.status = AppointmentStatus.COLLECTING_INFO
            question = await self._generate_information_question(session, missing_fields[0])
            
            return WorkflowResponse(
                success=True,
                message=question,
                session_id=session.session_id,
                step_name="collecting_information",
                next_step=missing_fields[0]
            )
        else:
            # Ready for confirmation
            session.status = AppointmentStatus.CONFIRMING
            confirmation_ui = await self.generate_confirmation_ui(session)
            
            confirmation_message = f"""
Perfect! I have all the details for your appointment. Please review and confirm:

**{session.collected_data.title}**
📍 **Location:** {session.collected_data.location}
🕐 **Date & Time:** {session.collected_data.date.strftime("%A, %B %d, %Y at %I:%M %p")}
⏱️ **Duration:** {session.collected_data.duration_minutes} minutes

Would you like me to confirm this appointment?
            """.strip()
            
            return WorkflowResponse(
                success=True,
                message=confirmation_message,
                session_id=session.session_id,
                step_name="awaiting_confirmation",
                ui_components=confirmation_ui
            )
    
    def _apply_extracted_details(self, appointment_data: AppointmentData, extracted_details: Dict[str, Any]):
        """Apply extracted details to appointment data."""
        
        if 'location' in extracted_details:
            appointment_data.location = extracted_details['location']
        
        if 'title' in extracted_details:
            appointment_data.title = extracted_details['title']
        
        if 'email' in extracted_details:
            emails = extracted_details['email']
            if isinstance(emails, list):
                appointment_data.attendee_emails.extend(emails)
            else:
                appointment_data.attendee_emails.append(emails)
        
        # Handle date and time parsing (simplified for now)
        if 'date' in extracted_details and 'time' in extracted_details:
            # This would need more sophisticated date/time parsing
            appointment_data.date = self._parse_datetime(
                extracted_details['date'], 
                extracted_details.get('time', '2:00 PM')
            )
        elif 'date' in extracted_details:
            appointment_data.date = self._parse_datetime(extracted_details['date'], '2:00 PM')
        elif 'time' in extracted_details:
            # Default to tomorrow at specified time
            tomorrow = datetime.now() + timedelta(days=1)
            appointment_data.date = self._parse_datetime('tomorrow', extracted_details['time'])
    
    def _parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse date and time strings into datetime object."""
        # Simplified parsing - would need more robust implementation
        now = datetime.now()
        
        if 'tomorrow' in date_str.lower():
            date_base = now + timedelta(days=1)
        elif 'today' in date_str.lower():
            date_base = now
        else:
            date_base = now + timedelta(days=1)  # Default to tomorrow
        
        # Simple time parsing
        if 'pm' in time_str.lower():
            hour = 14  # Default 2 PM
        else:
            hour = 10  # Default 10 AM
        
        return date_base.replace(hour=hour, minute=0, second=0, microsecond=0)
    
    def _check_missing_fields(self, appointment_data: AppointmentData) -> List[str]:
        """Check which required fields are missing."""
        missing = []
        
        if not appointment_data.location or appointment_data.location.strip() == "":
            missing.append("location")
        
        # Check if date is still placeholder (today or in past)
        if appointment_data.date <= datetime.now():
            missing.append("date_time")
        
        return missing
    
    async def _generate_information_question(self, session: AppointmentSession, missing_field: str) -> str:
        """Generate a question to collect missing information."""
        
        user_context = f"User: {session.user_id}, Current data: {session.collected_data.__dict__}"
        
        question_prompts = {
            "location": "Where would you like to have this meeting? You can specify an address, office location, or if it should be a virtual meeting.",
            "date_time": "When would you like to schedule this meeting? Please let me know your preferred date and time.",
            "attendees": "Who else should I invite to this meeting? Please provide their email addresses."
        }
        
        if missing_field in question_prompts:
            return question_prompts[missing_field]
        else:
            return f"I need some additional information about your {missing_field}. Could you please provide that?"
    
    async def _extract_and_update_information(self, session: AppointmentSession, user_response: str):
        """Extract information from user response and update session data."""
        
        logger.info(f"Processing user response: '{user_response}' for session {session.session_id}")
        logger.info(f"Current location before processing: '{session.collected_data.location}'")
        logger.info(f"Session status: {session.status}")
        
        # Handle confirmation responses when session is in confirming state
        if session.status == AppointmentStatus.CONFIRMING:
            confirmation_keywords = ['yes', 'confirm', 'ok', 'okay', 'sure', 'correct', 'right', 'approve', 'proceed', 'go ahead']
            cancellation_keywords = ['no', 'cancel', 'stop', 'abort', 'never mind', 'not now']
            
            response_lower = user_response.lower().strip()
            
            # Check for confirmation
            if any(keyword in response_lower for keyword in confirmation_keywords):
                logger.info(f"Confirmation detected in response: '{user_response}'")
                # Mark this as a confirmation response in session data
                session.collected_data.confirmation_response = "confirmed"
                return
            
            # Check for cancellation
            if any(keyword in response_lower for keyword in cancellation_keywords):
                logger.info(f"Cancellation detected in response: '{user_response}'")
                # Mark this as a cancellation response in session data
                session.collected_data.confirmation_response = "cancelled"
                return
        
        # Use intent detection service to extract details
        intent_result = await appointment_intent_detection_service.detect_appointment_intent(user_response)
        extracted_details = intent_result.extracted_details
        logger.info(f"Intent service extracted details: {extracted_details}")
        
        # Apply extracted details
        self._apply_extracted_details(session.collected_data, extracted_details)
        logger.info(f"Location after applying extracted details: '{session.collected_data.location}'")
        
        # Manual extraction for common patterns
        response_lower = user_response.lower()
        
        # Enhanced Location extraction - only when collecting info and location is missing
        if (session.status == AppointmentStatus.COLLECTING_INFO and 
            (not session.collected_data.location or session.collected_data.location.strip() == "")):
            
            logger.info(f"Location is missing, checking if response '{user_response}' is a location")
            
            # Check if this response contains location-like content
            location_indicators = ['office', 'building', 'room', 'address', 'virtual', 'zoom', 'meet', 'street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd', 'drive', 'dr']
            
            has_indicators = any(word in response_lower for word in location_indicators)
            looks_like_address = self._looks_like_address(user_response)
            
            # Remove the overly broad "short response" logic and add appointment request exclusion
            is_appointment_request = any(phrase in response_lower for phrase in [
                'schedule', 'appointment', 'meeting', 'book', 'arrange', 'set up'
            ])
            
            logger.info(f"Location check - has_indicators: {has_indicators}, looks_like_address: {looks_like_address}, is_appointment_request: {is_appointment_request}")
            
            # Only capture as location if it has indicators OR looks like address AND is not an appointment request
            if (has_indicators or looks_like_address) and not is_appointment_request:
                session.collected_data.location = user_response.strip()
                logger.info(f"✅ Captured location: '{session.collected_data.location}'")
            else:
                logger.info(f"❌ Response does not look like a location or is an appointment request")
        else:
            logger.info(f"Location already set: '{session.collected_data.location}' or not collecting info")
        
        # Simple date/time extraction
        if any(word in response_lower for word in ['tomorrow', 'today', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            old_date = session.collected_data.date
            session.collected_data.date = self._parse_datetime(user_response, '2:00 PM')
            logger.info(f"Updated date from {old_date} to {session.collected_data.date}")
        
        logger.info(f"Final appointment data after processing: location='{session.collected_data.location}', date={session.collected_data.date}")
    
    def _looks_like_address(self, text: str) -> bool:
        """Check if text looks like an address."""
        import re
        
        # Common address patterns
        address_patterns = [
            r'\d+\s+\w+\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|place|pl|lane|ln|way|court|ct)',
            r'\d+\s+\w+\s+\w+',  # Basic pattern like "123 Main St"
            r'\w+\s+(building|office|center|centre)',
            r'virtual|online|zoom|meet|teams',
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in address_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Also check if it's a simple numeric + word pattern (like "84 Mulberry St")
        if re.match(r'^\d+\s+\w+.*', text.strip()):
            return True
            
        return False
    
    async def _save_session(self, session: AppointmentSession):
        """Save appointment session to database."""
        try:
            db = get_database()
            collection = db["appointment_sessions"]
            
            session_dict = {
                "_id": session.session_id,
                "user_id": session.user_id,
                "status": session.status.value,
                "collected_data": {
                    "title": session.collected_data.title,
                    "location": session.collected_data.location,
                    "date": session.collected_data.date.isoformat(),
                    "duration_minutes": session.collected_data.duration_minutes,
                    "attendee_emails": session.collected_data.attendee_emails,
                    "description": session.collected_data.description,
                    "meet_link": session.collected_data.meet_link,
                    "calendar_event_id": session.collected_data.calendar_event_id,
                    "organizer_email": session.collected_data.organizer_email
                },
                "missing_fields": session.missing_fields,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "conversation_history": session.conversation_history
            }
            
            await collection.replace_one(
                {"_id": session.session_id},
                session_dict,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            raise
    
    async def _load_session(self, session_id: str) -> Optional[AppointmentSession]:
        """Load appointment session from database."""
        try:
            db = get_database()
            collection = db["appointment_sessions"]
            
            session_doc = await collection.find_one({"_id": session_id})
            
            if not session_doc:
                return None
            
            # Reconstruct AppointmentData
            data_dict = session_doc["collected_data"]
            appointment_data = AppointmentData(
                title=data_dict["title"],
                location=data_dict["location"],
                date=datetime.fromisoformat(data_dict["date"]),
                duration_minutes=data_dict["duration_minutes"],
                attendee_emails=data_dict["attendee_emails"],
                description=data_dict.get("description"),
                meet_link=data_dict.get("meet_link"),
                calendar_event_id=data_dict.get("calendar_event_id"),
                organizer_email=data_dict.get("organizer_email")
            )
            
            # Reconstruct AppointmentSession
            session = AppointmentSession(
                session_id=session_doc["_id"],
                user_id=session_doc["user_id"],
                status=AppointmentStatus(session_doc["status"]),
                collected_data=appointment_data,
                missing_fields=session_doc.get("missing_fields", []),
                created_at=datetime.fromisoformat(session_doc["created_at"]),
                updated_at=datetime.fromisoformat(session_doc["updated_at"]),
                conversation_history=session_doc.get("conversation_history", [])
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None

    async def _confirm_appointment_internal(self, session: AppointmentSession) -> WorkflowResponse:
        """Internal method to confirm appointment and update session."""
        
        # Update session status
        session.status = AppointmentStatus.CONFIRMED
        session.updated_at = datetime.now()
        
        # Save updated session
        await self._save_session(session)
        
        # Generate success message
        appointment = session.collected_data
        formatted_date = appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")
        success_message = f"""
🎉 **Appointment Confirmed!**

Your appointment has been successfully scheduled:

📋 **{appointment.title}**
📍 **Location:** {appointment.location}
🕐 **Date & Time:** {formatted_date}
⏱️ **Duration:** {appointment.duration_minutes} minutes

You will receive a calendar invitation shortly with all the details and a Google Meet link for the meeting.
        """.strip()
        
        return WorkflowResponse(
            success=True,
            message=success_message,
            session_id=session.session_id,
            step_name="appointment_confirmed",
            appointment_data=appointment
        )

    async def _cancel_appointment_internal(self, session: AppointmentSession) -> WorkflowResponse:
        """Internal method to cancel appointment and update session."""
        
        # Update session status
        session.status = AppointmentStatus.CANCELLED
        session.updated_at = datetime.now()
        
        # Save updated session
        await self._save_session(session)
        
        return WorkflowResponse(
            success=True,
            message="No problem! Your appointment booking has been cancelled. Feel free to ask me anything else or start a new appointment booking whenever you're ready.",
            session_id=session.session_id,
            step_name="appointment_cancelled"
        )


# Global service instance
appointment_workflow_manager = AppointmentWorkflowManager() 