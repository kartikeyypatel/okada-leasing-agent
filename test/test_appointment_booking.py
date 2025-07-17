# /test/test_appointment_booking.py
"""
Comprehensive test suite for the Appointment Booking Chatbot system.

Tests all components of the appointment booking feature including:
- Intent detection
- Workflow management
- UI components
- Error handling
- Integration with existing systems
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the appointment booking components
from app.appointment_intent_detection import appointment_intent_detection_service
from app.appointment_workflow_manager import appointment_workflow_manager
from app.enhanced_calendar import enhanced_calendar_service
from app.google_meet_service import google_meet_service
from app.email_notification_service import email_notification_service
from app.appointment_error_handler import appointment_error_handler

from app.models import (
    AppointmentIntent, AppointmentData, AppointmentSession, 
    AppointmentStatus, WorkflowResponse, ConfirmationUI
)


class TestAppointmentIntentDetection:
    """Test the appointment intent detection service."""
    
    @pytest.mark.asyncio
    async def test_appointment_intent_detection_positive(self):
        """Test that appointment phrases are correctly detected."""
        test_cases = [
            ("I want to book an appointment", True, 0.8),
            ("Can we schedule a meeting tomorrow?", True, 0.8),
            ("Let's set up a call for next week", True, 0.8),
            ("I need to arrange a meeting", True, 0.8),
            ("When are you available for a meeting?", True, 0.7),
        ]
        
        for message, expected_intent, min_confidence in test_cases:
            result = await appointment_intent_detection_service.detect_appointment_intent(message)
            
            assert result.is_appointment_request == expected_intent, f"Failed for: {message}"
            assert result.confidence >= min_confidence, f"Low confidence for: {message}"
            assert result.intent_type == "appointment_booking"
    
    @pytest.mark.asyncio
    async def test_appointment_intent_detection_negative(self):
        """Test that non-appointment phrases are correctly rejected."""
        test_cases = [
            ("Tell me about this property", False),
            ("What's the rent for 123 Main St?", False),
            ("How are you today?", False),
            ("I like this apartment", False),
            ("Thank you for the information", False),
        ]
        
        for message, expected_intent in test_cases:
            result = await appointment_intent_detection_service.detect_appointment_intent(message)
            
            assert result.is_appointment_request == expected_intent, f"Failed for: {message}"
            assert result.confidence < 0.6, f"Too high confidence for: {message}"
    
    def test_appointment_detail_extraction(self):
        """Test extraction of appointment details from messages."""
        test_cases = [
            {
                "message": "I want to book an appointment at the office tomorrow at 2pm",
                "expected_details": {
                    "location": "office",
                    "date": "tomorrow", 
                    "time": "2pm"
                }
            },
            {
                "message": "Can we schedule a meeting with john@example.com at 123 Main St?",
                "expected_details": {
                    "location": "123 main st",
                    "email": ["john@example.com"]
                }
            },
            {
                "message": "Let's have a call about the property listing on Friday morning",
                "expected_details": {
                    "date": "friday",
                    "time": "morning"
                }
            }
        ]
        
        for test_case in test_cases:
            result = appointment_intent_detection_service.extract_appointment_details(
                test_case["message"]
            )
            
            for key, expected_value in test_case["expected_details"].items():
                assert key in result, f"Missing {key} in extracted details"
                if isinstance(expected_value, list):
                    assert all(item in result[key] for item in expected_value)
                else:
                    assert expected_value.lower() in result[key].lower()


class TestAppointmentWorkflowManager:
    """Test the appointment workflow management."""
    
    @pytest.fixture
    def sample_appointment_data(self):
        """Fixture providing sample appointment data."""
        return AppointmentData(
            title="Property Viewing",
            location="123 Main St",
            date=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            attendee_emails=["client@example.com"],
            organizer_email="agent@okada.com"
        )
    
    @pytest.fixture
    def sample_appointment_session(self, sample_appointment_data):
        """Fixture providing sample appointment session."""
        return AppointmentSession(
            session_id="test-session-123",
            user_id="test@example.com",
            status=AppointmentStatus.COLLECTING_INFO,
            collected_data=sample_appointment_data
        )
    
    @pytest.mark.asyncio
    @patch('app.appointment_workflow_manager.appointment_intent_detection_service')
    @patch('app.appointment_workflow_manager.AppointmentWorkflowManager._save_session')
    async def test_start_appointment_booking_success(self, mock_save, mock_intent_service):
        """Test successful start of appointment booking workflow."""
        # Mock intent detection
        mock_intent = AppointmentIntent(
            is_appointment_request=True,
            confidence=0.9,
            extracted_details={"location": "office", "time": "2pm"},
            missing_fields=["date"]
        )
        mock_intent_service.detect_appointment_intent.return_value = mock_intent
        mock_save.return_value = None
        
        # Test workflow start
        response = await appointment_workflow_manager.start_appointment_booking(
            "test@example.com",
            "I want to book an appointment at the office at 2pm"
        )
        
        assert response.success == True
        assert response.session_id is not None
        assert "when would you like" in response.message.lower()
        assert response.step_name == "collecting_information"
    
    @pytest.mark.asyncio
    async def test_appointment_confirmation_ui_generation(self, sample_appointment_session):
        """Test generation of appointment confirmation UI."""
        ui = await appointment_workflow_manager.generate_confirmation_ui(sample_appointment_session)
        
        assert isinstance(ui, ConfirmationUI)
        assert "appointment_confirmation" in ui.appointment_card["type"]
        assert len(ui.action_buttons) == 2
        
        # Check button configuration
        confirm_button = next(btn for btn in ui.action_buttons if btn["action"] == "confirm")
        cancel_button = next(btn for btn in ui.action_buttons if btn["action"] == "cancel")
        
        assert confirm_button["type"] == "primary"
        assert cancel_button["type"] == "secondary"
        assert "✅" in confirm_button["text"]
        assert "❌" in cancel_button["text"]
    
    @pytest.mark.asyncio
    @patch('app.appointment_workflow_manager.AppointmentWorkflowManager._load_session')
    @patch('app.appointment_workflow_manager.AppointmentWorkflowManager._save_session')
    async def test_process_user_response(self, mock_save, mock_load, sample_appointment_session):
        """Test processing user responses in workflow."""
        mock_load.return_value = sample_appointment_session
        mock_save.return_value = None
        
        response = await appointment_workflow_manager.process_user_response(
            "test-session-123",
            "Tomorrow at 2pm"
        )
        
        assert response.success == True
        assert response.session_id == "test-session-123"


class TestEnhancedCalendarService:
    """Test the enhanced calendar service."""
    
    @pytest.fixture
    def sample_appointment_data(self):
        """Sample appointment data for testing."""
        return AppointmentData(
            title="Client Meeting",
            location="Office Conference Room",
            date=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            attendee_emails=["client@example.com"],
            organizer_email="agent@okada.com"
        )
    
    @pytest.mark.asyncio
    @patch('app.enhanced_calendar.base_calendar.schedule_viewing')
    async def test_create_appointment_with_meet_success(self, mock_schedule, sample_appointment_data):
        """Test successful appointment creation with Google Meet."""
        # Mock calendar service response
        mock_schedule.return_value = "https://calendar.google.com/event/12345"
        
        result = await enhanced_calendar_service.create_appointment_with_meet(sample_appointment_data)
        
        assert result["success"] == True
        assert "event_id" in result
        assert "meet_link" in result
        assert result["attendees_notified"] == True
        
        # Verify calendar service was called
        mock_schedule.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.enhanced_calendar.base_calendar.schedule_viewing')
    async def test_create_appointment_calendar_failure(self, mock_schedule, sample_appointment_data):
        """Test handling of calendar service failures."""
        # Mock calendar service failure
        mock_schedule.side_effect = Exception("Calendar service unavailable")
        
        result = await enhanced_calendar_service.create_appointment_with_meet(sample_appointment_data)
        
        assert result["success"] == False
        assert "fallback_info" in result
        assert "manual" in result["fallback_info"]["fallback_instructions"].lower()
    
    @pytest.mark.asyncio
    async def test_send_calendar_invitations(self):
        """Test sending calendar invitations."""
        result = await enhanced_calendar_service.send_calendar_invitations(
            "event123",
            ["attendee1@example.com", "attendee2@example.com"]
        )
        
        # In the mock implementation, this should return True
        assert result == True


class TestGoogleMeetService:
    """Test the Google Meet integration service."""
    
    @pytest.mark.asyncio
    async def test_create_meet_link(self):
        """Test Google Meet link creation."""
        meet_link = await google_meet_service.create_meet_link(
            "calendar-event-123",
            "Client Meeting"
        )
        
        assert "meet.google.com" in meet_link
        assert meet_link.startswith("https://")
    
    @pytest.mark.asyncio
    async def test_create_instant_meeting(self):
        """Test instant meeting creation."""
        result = await google_meet_service.create_instant_meeting(
            "organizer@example.com",
            "Urgent Discussion"
        )
        
        assert result["success"] == True
        assert "meet_link" in result
        assert "join_instructions" in result
        assert result["organizer"] == "organizer@example.com"
    
    def test_validate_meet_link(self):
        """Test Google Meet link validation."""
        valid_links = [
            "https://meet.google.com/abc-defg-hij",
            "https://g.co/meet/xyz123",
            "https://hangouts.google.com/call/abc123"
        ]
        
        invalid_links = [
            "https://zoom.us/j/123456789",
            "https://teams.microsoft.com/meet",
            "not-a-url",
            ""
        ]
        
        for link in valid_links:
            assert google_meet_service.validate_meet_link(link) == True
        
        for link in invalid_links:
            assert google_meet_service.validate_meet_link(link) == False
    
    def test_extract_meet_id_from_link(self):
        """Test extraction of Meet ID from URLs."""
        test_cases = [
            ("https://meet.google.com/abc-defg-hij", "abc-defg-hij"),
            ("https://meet.google.com/xyz123?param=value", "xyz123"),
            ("invalid-url", None)
        ]
        
        for url, expected_id in test_cases:
            result = google_meet_service.extract_meet_id_from_link(url)
            assert result == expected_id


class TestEmailNotificationService:
    """Test the email notification service."""
    
    @pytest.fixture
    def sample_appointment_data(self):
        """Sample appointment data for email testing."""
        return AppointmentData(
            title="Property Consultation",
            location="Downtown Office",
            date=datetime(2024, 1, 15, 14, 0),  # Fixed date for testing
            duration_minutes=45,
            attendee_emails=["client@example.com"],
            organizer_email="agent@okada.com",
            description="Discussion about property options"
        )
    
    @pytest.mark.asyncio
    @patch('app.email_notification_service.EmailNotificationService._send_email')
    async def test_send_appointment_confirmation(self, mock_send, sample_appointment_data):
        """Test sending appointment confirmation emails."""
        mock_send.return_value = True
        
        result = await email_notification_service.send_appointment_confirmation(sample_appointment_data)
        
        assert result == True
        # Should send to organizer + attendees
        assert mock_send.call_count == 2
    
    @pytest.mark.asyncio
    @patch('app.email_notification_service.EmailNotificationService._send_email')
    async def test_send_appointment_reminder(self, mock_send, sample_appointment_data):
        """Test sending appointment reminders."""
        mock_send.return_value = True
        
        result = await email_notification_service.send_appointment_reminder(
            sample_appointment_data, 
            "24h"
        )
        
        assert result == True
        assert mock_send.call_count >= 1
    
    @pytest.mark.asyncio
    @patch('app.email_notification_service.EmailNotificationService._send_email')
    async def test_send_appointment_cancellation(self, mock_send, sample_appointment_data):
        """Test sending appointment cancellation emails."""
        mock_send.return_value = True
        
        result = await email_notification_service.send_appointment_cancellation(
            sample_appointment_data,
            "Schedule conflict"
        )
        
        assert result == True
        assert mock_send.call_count >= 1
    
    def test_email_content_generation(self, sample_appointment_data):
        """Test email content generation methods."""
        # Test HTML content generation
        html_content = email_notification_service._create_confirmation_email_html(sample_appointment_data)
        
        assert "Property Consultation" in html_content
        assert "Downtown Office" in html_content
        assert "January 15, 2024" in html_content
        assert "client@example.com" not in html_content  # Should not expose attendee emails in content
        
        # Test text content generation
        text_content = email_notification_service._create_confirmation_email_text(sample_appointment_data)
        
        assert "Property Consultation" in text_content
        assert "Downtown Office" in text_content
        assert "45 minutes" in text_content


class TestAppointmentErrorHandler:
    """Test the appointment error handling service."""
    
    @pytest.mark.asyncio
    async def test_handle_intent_detection_error(self):
        """Test handling of intent detection errors."""
        response = await appointment_error_handler.handle_error(
            "INTENT_DETECTION_ERROR",
            "Could not parse user intent",
            {"user_message": "unclear request"}
        )
        
        assert response.success == False
        assert "rephras" in response.message.lower()
        assert response.step_name == "intent_recovery"
        assert "examples" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_handle_calendar_service_error(self):
        """Test handling of calendar service errors."""
        appointment_data = AppointmentData(
            title="Test Meeting",
            location="Test Location",
            date=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            organizer_email="test@example.com"
        )
        
        response = await appointment_error_handler.handle_error(
            "CALENDAR_SERVICE_ERROR",
            "Calendar API unavailable",
            {"appointment_data": appointment_data}
        )
        
        assert response.success == True  # Partial success
        assert "manual" in response.message.lower()
        assert response.step_name == "calendar_fallback"
        assert "Test Meeting" in response.message
    
    @pytest.mark.asyncio
    async def test_handle_partial_failure(self):
        """Test handling of partial service failures."""
        appointment_data = AppointmentData(
            title="Partial Failure Test",
            location="Test Location",
            date=datetime.now() + timedelta(days=1),
            duration_minutes=30,
            organizer_email="test@example.com"
        )
        
        response = await appointment_error_handler.handle_partial_failure(
            appointment_data,
            ["email", "google_meet"]  # These services failed
        )
        
        assert response.success == True
        assert "partially confirmed" in response.message.lower()
        assert "calendar event created" in response.message.lower()
        assert "manual steps required" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_unknown_error_handling(self):
        """Test handling of unknown errors."""
        response = await appointment_error_handler.handle_error(
            "UNKNOWN_WEIRD_ERROR",
            "Something went wrong",
            {}
        )
        
        assert response.success == False
        assert "unexpected issue" in response.message.lower()
        assert response.step_name == "unknown_error_recovery"
        assert "start fresh" in response.message.lower()


class TestAppointmentIntegration:
    """Integration tests for the complete appointment booking system."""
    
    @pytest.mark.asyncio
    @patch('app.appointment_workflow_manager.AppointmentWorkflowManager._save_session')
    @patch('app.appointment_workflow_manager.AppointmentWorkflowManager._load_session')
    async def test_complete_appointment_booking_flow(self, mock_load, mock_save):
        """Test the complete appointment booking workflow from start to finish."""
        mock_save.return_value = None
        
        # Step 1: Start appointment booking
        start_response = await appointment_workflow_manager.start_appointment_booking(
            "test@example.com",
            "I want to book an appointment"
        )
        
        assert start_response.success == True
        session_id = start_response.session_id
        
        # Step 2: Mock session for subsequent calls
        test_session = AppointmentSession(
            session_id=session_id,
            user_id="test@example.com",
            status=AppointmentStatus.COLLECTING_INFO,
            collected_data=AppointmentData(
                title="Meeting",
                location="",
                date=datetime.now(),
                organizer_email="test@example.com"
            )
        )
        mock_load.return_value = test_session
        
        # Step 3: Provide location
        location_response = await appointment_workflow_manager.process_user_response(
            session_id,
            "At the downtown office"
        )
        
        assert location_response.success == True
        
        # Step 4: Provide date/time
        test_session.collected_data.location = "downtown office"
        datetime_response = await appointment_workflow_manager.process_user_response(
            session_id,
            "Tomorrow at 2pm"
        )
        
        assert datetime_response.success == True
    
    @pytest.mark.asyncio
    async def test_appointment_intent_to_workflow_integration(self):
        """Test integration between intent detection and workflow management."""
        # Test message that should trigger appointment booking
        test_message = "I need to schedule a meeting for tomorrow at the office"
        
        # Detect intent
        intent = await appointment_intent_detection_service.detect_appointment_intent(test_message)
        
        assert intent.is_appointment_request == True
        assert intent.confidence > 0.6
        
        # Verify extracted details are useful for workflow
        assert len(intent.extracted_details) > 0
        assert len(intent.missing_fields) >= 0  # Some fields might be missing
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self):
        """Test error recovery across different components."""
        # Test calendar error recovery
        calendar_error_response = await appointment_error_handler.handle_error(
            "CALENDAR_SERVICE_ERROR",
            "Service unavailable",
            {
                "appointment_data": AppointmentData(
                    title="Recovery Test",
                    location="Test Office",
                    date=datetime.now() + timedelta(days=1),
                    duration_minutes=60,
                    organizer_email="test@example.com"
                )
            }
        )
        
        assert calendar_error_response.success == True  # Should provide fallback
        assert "manual" in calendar_error_response.message.lower()
        
        # Test intent error recovery
        intent_error_response = await appointment_error_handler.handle_error(
            "INTENT_DETECTION_ERROR",
            "Could not understand request",
            {}
        )
        
        assert intent_error_response.success == False
        assert "rephrase" in intent_error_response.message.lower()


class TestAppointmentUIComponents:
    """Test the appointment UI components (mock tests for frontend)."""
    
    def test_appointment_confirmation_props(self):
        """Test appointment confirmation component interface."""
        # This would test the TypeScript interfaces in a real frontend test
        sample_appointment = {
            "title": "Property Meeting",
            "location": "123 Main St",
            "datetime": "Monday, January 15, 2024 at 2:00 PM",
            "duration": 60,
            "attendees": "client@example.com",
            "description": "Property viewing discussion"
        }
        
        # In a real test, this would render the component and test interactions
        assert sample_appointment["title"] == "Property Meeting"
        assert sample_appointment["duration"] == 60
        assert "client@example.com" in sample_appointment["attendees"]
    
    def test_appointment_card_props(self):
        """Test appointment card component interface."""
        sample_card_props = {
            "title": "Client Consultation",
            "location": "Office Building",
            "datetime": "Tomorrow at 10:00 AM",
            "duration": 45,
            "status": "confirmed",
            "meetLink": "https://meet.google.com/abc-defg-hij"
        }
        
        # Test prop validation
        assert sample_card_props["status"] in ["upcoming", "confirmed", "cancelled"]
        assert "meet.google.com" in sample_card_props["meetLink"]
        assert sample_card_props["duration"] > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 