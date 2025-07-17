# /app/google_meet_service.py
"""
Google Meet Integration Service

This service handles Google Meet link creation and integration with calendar events
for appointment booking functionality.
"""

import logging
import hashlib
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GoogleMeetService:
    """
    Service for Google Meet integration.
    
    Handles Meet link creation, event association, and meeting management
    for appointment booking workflows.
    """
    
    def __init__(self):
        pass
    
    async def create_meet_link(self, calendar_event_id: str, meeting_title: str = "Meeting") -> str:
        """
        Create a Google Meet link for a calendar event.
        
        Args:
            calendar_event_id: Associated calendar event ID
            meeting_title: Title for the meeting
            
        Returns:
            Google Meet link URL
        """
        logger.info(f"Creating Google Meet link for event {calendar_event_id}")
        
        try:
            # In production, this would use Google Calendar API to create Meet link
            # For demo purposes, we'll generate a realistic-looking Meet link
            
            # Generate a unique meet ID
            meet_id = self._generate_meet_id(calendar_event_id, meeting_title)
            meet_link = f"https://meet.google.com/{meet_id}"
            
            logger.info(f"Generated Meet link: {meet_link}")
            return meet_link
            
        except Exception as e:
            logger.error(f"Error creating Google Meet link: {e}")
            # Return a fallback link
            return f"https://meet.google.com/new"
    
    async def add_meet_to_event(self, event_id: str, meet_link: str) -> bool:
        """
        Add a Google Meet link to an existing calendar event.
        
        Args:
            event_id: Calendar event ID
            meet_link: Google Meet link to add
            
        Returns:
            Success status
        """
        logger.info(f"Adding Meet link to event {event_id}")
        
        try:
            # In production, this would use Google Calendar API to update the event
            # with conference data
            
            # For now, we'll simulate success
            logger.info(f"Successfully added Meet link to event {event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding Meet link to event: {e}")
            return False
    
    async def create_instant_meeting(self, organizer_email: str, meeting_title: str = "Instant Meeting") -> Dict[str, Any]:
        """
        Create an instant Google Meet for immediate use.
        
        Args:
            organizer_email: Email of meeting organizer
            meeting_title: Title for the meeting
            
        Returns:
            Dictionary with meeting details
        """
        logger.info(f"Creating instant meeting for {organizer_email}")
        
        try:
            # Generate instant meeting details
            meet_id = self._generate_meet_id(organizer_email, meeting_title)
            meet_link = f"https://meet.google.com/{meet_id}"
            
            return {
                'success': True,
                'meet_link': meet_link,
                'meet_id': meet_id,
                'organizer': organizer_email,
                'title': meeting_title,
                'created_at': datetime.now().isoformat(),
                'join_instructions': self._create_join_instructions(meet_link)
            }
            
        except Exception as e:
            logger.error(f"Error creating instant meeting: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_link': 'https://meet.google.com/new'
            }
    
    async def get_meeting_info(self, meet_id: str) -> Dict[str, Any]:
        """
        Get information about a Google Meet.
        
        Args:
            meet_id: Google Meet ID
            
        Returns:
            Dictionary with meeting information
        """
        logger.info(f"Getting meeting info for {meet_id}")
        
        try:
            # In production, this would query Google Meet API for meeting details
            # For now, we'll return basic info
            
            return {
                'meet_id': meet_id,
                'meet_link': f"https://meet.google.com/{meet_id}",
                'status': 'active',
                'participant_count': 0,  # Would be actual count in production
                'join_instructions': self._create_join_instructions(f"https://meet.google.com/{meet_id}")
            }
            
        except Exception as e:
            logger.error(f"Error getting meeting info: {e}")
            return {
                'error': str(e),
                'meet_id': meet_id
            }
    
    async def end_meeting(self, meet_id: str) -> bool:
        """
        End a Google Meet session.
        
        Args:
            meet_id: Google Meet ID to end
            
        Returns:
            Success status
        """
        logger.info(f"Ending meeting {meet_id}")
        
        try:
            # In production, this would use Google Meet API to end the meeting
            # For now, we'll simulate success
            
            logger.info(f"Meeting {meet_id} ended successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error ending meeting: {e}")
            return False
    
    def _generate_meet_id(self, seed_data: str, additional_data: str = "") -> str:
        """
        Generate a realistic Google Meet ID.
        
        Args:
            seed_data: Data to seed the ID generation
            additional_data: Additional data for uniqueness
            
        Returns:
            Generated Meet ID in Google Meet format
        """
        # Create a hash from the input data
        combined_data = f"{seed_data}{additional_data}{datetime.now().isoformat()}"
        hash_object = hashlib.md5(combined_data.encode())
        hash_hex = hash_object.hexdigest()
        
        # Format like a Google Meet ID: xxx-xxxx-xxx
        meet_id = f"{hash_hex[:3]}-{hash_hex[3:7]}-{hash_hex[7:10]}"
        
        return meet_id
    
    def _create_join_instructions(self, meet_link: str) -> str:
        """Create comprehensive join instructions for a Google Meet."""
        
        return f"""
🎥 **How to Join the Meeting:**

**Option 1: Click to Join**
• Click this link: {meet_link}
• Allow microphone and camera access when prompted

**Option 2: Join by Phone**
• Dial: +1 (US) or international number
• Enter meeting ID when prompted

**Option 3: Join from Calendar**
• Open your calendar event
• Click "Join with Google Meet"

**Tips for a Great Meeting:**
• Join from a quiet location
• Test your microphone and camera beforehand
• Use headphones for better audio quality
• Mute yourself when not speaking

**Trouble Joining?**
• Try refreshing your browser
• Use Google Chrome for best compatibility
• Check your internet connection
        """.strip()
    
    def validate_meet_link(self, meet_link: str) -> bool:
        """
        Validate if a link is a valid Google Meet link.
        
        Args:
            meet_link: Link to validate
            
        Returns:
            True if valid Google Meet link
        """
        if not meet_link:
            return False
        
        # Check for Google Meet URL patterns
        valid_patterns = [
            "meet.google.com/",
            "g.co/meet/",
            "hangouts.google.com/call/"
        ]
        
        return any(pattern in meet_link for pattern in valid_patterns)
    
    def extract_meet_id_from_link(self, meet_link: str) -> Optional[str]:
        """
        Extract Meet ID from a Google Meet link.
        
        Args:
            meet_link: Google Meet link
            
        Returns:
            Extracted Meet ID or None
        """
        try:
            if "meet.google.com/" in meet_link:
                # Extract ID after last slash
                meet_id = meet_link.split("/")[-1]
                # Remove any query parameters
                meet_id = meet_id.split("?")[0]
                return meet_id
            return None
        except Exception:
            return None


# Global service instance
google_meet_service = GoogleMeetService() 