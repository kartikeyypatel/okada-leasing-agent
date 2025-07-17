# /app/email_notification_service.py
"""
Email Notification Service for Appointment Booking

This service handles sending appointment confirmations, reminders, and 
updates via email for the appointment booking system.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from app.models import AppointmentData, AppointmentError

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """
    Service for sending appointment-related email notifications.
    
    Handles confirmation emails, reminders, updates, and cancellations
    with professional email templates.
    """
    
    def __init__(self):
        # Email configuration - in production, these would come from environment variables
        self.smtp_server = "smtp.gmail.com"  # Configure as needed
        self.smtp_port = 587
        self.sender_email = "noreply@okadaleasing.com"  # Configure as needed
        self.sender_password = None  # Would be set from environment variables
        
    async def send_appointment_confirmation(self, appointment: AppointmentData) -> bool:
        """
        Send appointment confirmation email to all attendees.
        
        Args:
            appointment: Appointment data with all details
            
        Returns:
            Success status of email sending
        """
        logger.info(f"Sending appointment confirmation for {appointment.title}")
        
        try:
            # Create confirmation email content
            subject = f"Appointment Confirmed: {appointment.title}"
            html_content = self._create_confirmation_email_html(appointment)
            text_content = self._create_confirmation_email_text(appointment)
            
            # Send to organizer
            success = True
            if appointment.organizer_email:
                organizer_success = await self._send_email(
                    to_email=appointment.organizer_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and organizer_success
            
            # Send to attendees
            for attendee_email in appointment.attendee_emails:
                attendee_success = await self._send_email(
                    to_email=attendee_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and attendee_success
            
            if success:
                logger.info(f"Appointment confirmation emails sent successfully")
            else:
                logger.warning(f"Some appointment confirmation emails failed to send")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending appointment confirmation: {e}")
            return False
    
    async def send_appointment_reminder(self, appointment: AppointmentData, reminder_type: str = "24h") -> bool:
        """
        Send appointment reminder email.
        
        Args:
            appointment: Appointment data
            reminder_type: Type of reminder (24h, 1h, etc.)
            
        Returns:
            Success status of email sending
        """
        logger.info(f"Sending {reminder_type} appointment reminder for {appointment.title}")
        
        try:
            subject = f"Reminder: {appointment.title} - {self._format_reminder_time(appointment, reminder_type)}"
            html_content = self._create_reminder_email_html(appointment, reminder_type)
            text_content = self._create_reminder_email_text(appointment, reminder_type)
            
            # Send to all participants
            success = True
            
            if appointment.organizer_email:
                organizer_success = await self._send_email(
                    to_email=appointment.organizer_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and organizer_success
            
            for attendee_email in appointment.attendee_emails:
                attendee_success = await self._send_email(
                    to_email=attendee_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and attendee_success
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending appointment reminder: {e}")
            return False
    
    async def send_appointment_update(self, appointment: AppointmentData, changes: Dict[str, Any]) -> bool:
        """
        Send appointment update notification.
        
        Args:
            appointment: Updated appointment data
            changes: Dictionary of what changed
            
        Returns:
            Success status of email sending
        """
        logger.info(f"Sending appointment update notification for {appointment.title}")
        
        try:
            subject = f"Updated: {appointment.title}"
            html_content = self._create_update_email_html(appointment, changes)
            text_content = self._create_update_email_text(appointment, changes)
            
            # Send to all participants
            success = True
            
            if appointment.organizer_email:
                organizer_success = await self._send_email(
                    to_email=appointment.organizer_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and organizer_success
            
            for attendee_email in appointment.attendee_emails:
                attendee_success = await self._send_email(
                    to_email=attendee_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and attendee_success
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending appointment update: {e}")
            return False
    
    async def send_appointment_cancellation(self, appointment: AppointmentData, reason: str = "") -> bool:
        """
        Send appointment cancellation notification.
        
        Args:
            appointment: Cancelled appointment data
            reason: Optional cancellation reason
            
        Returns:
            Success status of email sending
        """
        logger.info(f"Sending appointment cancellation for {appointment.title}")
        
        try:
            subject = f"Cancelled: {appointment.title}"
            html_content = self._create_cancellation_email_html(appointment, reason)
            text_content = self._create_cancellation_email_text(appointment, reason)
            
            # Send to all participants
            success = True
            
            if appointment.organizer_email:
                organizer_success = await self._send_email(
                    to_email=appointment.organizer_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and organizer_success
            
            for attendee_email in appointment.attendee_emails:
                attendee_success = await self._send_email(
                    to_email=attendee_email,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content
                )
                success = success and attendee_success
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending appointment cancellation: {e}")
            return False
    
    async def _send_email(self, to_email: str, subject: str, html_content: str, text_content: str) -> bool:
        """
        Send an individual email.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text email content
            
        Returns:
            Success status
        """
        try:
            # For demo purposes, we'll simulate email sending by logging
            # In production, this would use actual SMTP or email service
            
            logger.info(f"📧 Simulating email send to {to_email}")
            logger.info(f"Subject: {subject}")
            logger.debug(f"Content preview: {text_content[:100]}...")
            
            # Simulate email sending delay
            import asyncio
            await asyncio.sleep(0.1)
            
            # In production, uncomment and configure the following:
            # return await self._send_smtp_email(to_email, subject, html_content, text_content)
            
            return True  # Simulate success
            
        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            return False
    
    async def _send_smtp_email(self, to_email: str, subject: str, html_content: str, text_content: str) -> bool:
        """
        Send email using SMTP (production implementation).
        
        This method would be used in production environments.
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = to_email
            
            # Create text and HTML parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')
            
            # Add parts to message
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.sender_password:
                    server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP email sending failed: {e}")
            return False
    
    def _create_confirmation_email_html(self, appointment: AppointmentData) -> str:
        """Create HTML content for appointment confirmation email."""
        
        formatted_date = appointment.date.strftime("%A, %B %d, %Y")
        formatted_time = appointment.date.strftime("%I:%M %p")
        
        meet_section = ""
        if appointment.meet_link:
            meet_section = f"""
            <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0;">🎥 Join via Google Meet</h3>
                <a href="{appointment.meet_link}" style="color: #1e40af; text-decoration: none; font-weight: bold;">
                    Click here to join the meeting
                </a>
            </div>
            """
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; text-align: center;">
                <h1 style="margin: 0;">✅ Appointment Confirmed</h1>
                <p style="margin: 10px 0 0 0; font-size: 18px;">Your meeting has been successfully scheduled</p>
            </div>
            
            <div style="background-color: white; padding: 30px; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px;">
                <h2 style="color: #374151; margin-top: 0;">📅 {appointment.title}</h2>
                
                <div style="margin: 20px 0;">
                    <p style="margin: 8px 0;"><strong>📍 Location:</strong> {appointment.location}</p>
                    <p style="margin: 8px 0;"><strong>📅 Date:</strong> {formatted_date}</p>
                    <p style="margin: 8px 0;"><strong>🕐 Time:</strong> {formatted_time}</p>
                    <p style="margin: 8px 0;"><strong>⏱️ Duration:</strong> {appointment.duration_minutes} minutes</p>
                </div>
                
                {meet_section}
                
                <div style="background-color: #f9fafb; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="color: #374151; margin: 0 0 10px 0;">📝 Meeting Details</h3>
                    <p style="margin: 0;">{appointment.description or "No additional details provided."}</p>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <p style="color: #6b7280; font-size: 14px;">
                        This appointment was scheduled through Okada Leasing Agent.<br>
                        You will receive a calendar invitation shortly.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_confirmation_email_text(self, appointment: AppointmentData) -> str:
        """Create plain text content for appointment confirmation email."""
        
        formatted_date = appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")
        
        meet_section = ""
        if appointment.meet_link:
            meet_section = f"""
Google Meet Link: {appointment.meet_link}
            """
        
        return f"""
✅ APPOINTMENT CONFIRMED

Your meeting has been successfully scheduled:

📋 {appointment.title}
📍 Location: {appointment.location}
🕐 Date & Time: {formatted_date}
⏱️ Duration: {appointment.duration_minutes} minutes

{meet_section}

📝 Meeting Details:
{appointment.description or "No additional details provided."}

This appointment was scheduled through Okada Leasing Agent.
You will receive a calendar invitation shortly.
        """.strip()
    
    def _create_reminder_email_html(self, appointment: AppointmentData, reminder_type: str) -> str:
        """Create HTML content for appointment reminder email."""
        
        time_until = self._format_reminder_time(appointment, reminder_type)
        formatted_date = appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")
        
        meet_section = ""
        if appointment.meet_link:
            meet_section = f"""
            <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0;">🎥 Ready to Join?</h3>
                <a href="{appointment.meet_link}" style="display: inline-block; background-color: #1e40af; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                    Join Google Meet
                </a>
            </div>
            """
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 30px; border-radius: 12px; text-align: center;">
                <h1 style="margin: 0;">⏰ Meeting Reminder</h1>
                <p style="margin: 10px 0 0 0; font-size: 18px;">{time_until}</p>
            </div>
            
            <div style="background-color: white; padding: 30px; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px;">
                <h2 style="color: #374151; margin-top: 0;">📅 {appointment.title}</h2>
                
                <div style="margin: 20px 0;">
                    <p style="margin: 8px 0;"><strong>📍 Location:</strong> {appointment.location}</p>
                    <p style="margin: 8px 0;"><strong>🕐 Date & Time:</strong> {formatted_date}</p>
                    <p style="margin: 8px 0;"><strong>⏱️ Duration:</strong> {appointment.duration_minutes} minutes</p>
                </div>
                
                {meet_section}
            </div>
        </body>
        </html>
        """
    
    def _create_reminder_email_text(self, appointment: AppointmentData, reminder_type: str) -> str:
        """Create plain text content for appointment reminder email."""
        
        time_until = self._format_reminder_time(appointment, reminder_type)
        formatted_date = appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")
        
        meet_section = ""
        if appointment.meet_link:
            meet_section = f"""
Google Meet Link: {appointment.meet_link}
            """
        
        return f"""
⏰ MEETING REMINDER

{time_until}

📋 {appointment.title}
📍 Location: {appointment.location}
🕐 Date & Time: {formatted_date}
⏱️ Duration: {appointment.duration_minutes} minutes

{meet_section}

See you at the meeting!
        """.strip()
    
    def _create_update_email_html(self, appointment: AppointmentData, changes: Dict[str, Any]) -> str:
        """Create HTML content for appointment update email."""
        
        changes_list = "\n".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in changes.items()])
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 30px; border-radius: 12px; text-align: center;">
                <h1 style="margin: 0;">📝 Meeting Updated</h1>
                <p style="margin: 10px 0 0 0; font-size: 18px;">Changes have been made to your appointment</p>
            </div>
            
            <div style="background-color: white; padding: 30px; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px;">
                <h2 style="color: #374151; margin-top: 0;">What Changed:</h2>
                <ul style="color: #374151;">{changes_list}</ul>
                
                <h3 style="color: #374151;">Updated Meeting Details:</h3>
                <p><strong>📋 Title:</strong> {appointment.title}</p>
                <p><strong>📍 Location:</strong> {appointment.location}</p>
                <p><strong>🕐 Date & Time:</strong> {appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")}</p>
            </div>
        </body>
        </html>
        """
    
    def _create_update_email_text(self, appointment: AppointmentData, changes: Dict[str, Any]) -> str:
        """Create plain text content for appointment update email."""
        
        changes_list = "\n".join([f"• {key}: {value}" for key, value in changes.items()])
        
        return f"""
📝 MEETING UPDATED

Changes have been made to your appointment:

{changes_list}

Updated Meeting Details:
📋 Title: {appointment.title}
📍 Location: {appointment.location}
🕐 Date & Time: {appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")}
        """.strip()
    
    def _create_cancellation_email_html(self, appointment: AppointmentData, reason: str) -> str:
        """Create HTML content for appointment cancellation email."""
        
        reason_section = ""
        if reason:
            reason_section = f"""
            <div style="background-color: #fef3c7; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #92400e; margin: 0 0 10px 0;">Reason for Cancellation:</h3>
                <p style="margin: 0; color: #92400e;">{reason}</p>
            </div>
            """
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 30px; border-radius: 12px; text-align: center;">
                <h1 style="margin: 0;">❌ Meeting Cancelled</h1>
                <p style="margin: 10px 0 0 0; font-size: 18px;">Your appointment has been cancelled</p>
            </div>
            
            <div style="background-color: white; padding: 30px; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px;">
                <h2 style="color: #374151; margin-top: 0;">Cancelled Meeting:</h2>
                <p><strong>📋 Title:</strong> {appointment.title}</p>
                <p><strong>📍 Location:</strong> {appointment.location}</p>
                <p><strong>🕐 Was scheduled for:</strong> {appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")}</p>
                
                {reason_section}
                
                <p style="color: #6b7280; margin-top: 20px;">
                    If you need to reschedule, please feel free to book a new appointment.
                </p>
            </div>
        </body>
        </html>
        """
    
    def _create_cancellation_email_text(self, appointment: AppointmentData, reason: str) -> str:
        """Create plain text content for appointment cancellation email."""
        
        reason_section = ""
        if reason:
            reason_section = f"""
Reason for Cancellation:
{reason}
            """
        
        return f"""
❌ MEETING CANCELLED

Your appointment has been cancelled:

📋 Title: {appointment.title}
📍 Location: {appointment.location}
🕐 Was scheduled for: {appointment.date.strftime("%A, %B %d, %Y at %I:%M %p")}

{reason_section}

If you need to reschedule, please feel free to book a new appointment.
        """.strip()
    
    def _format_reminder_time(self, appointment: AppointmentData, reminder_type: str) -> str:
        """Format the reminder time description."""
        
        if reminder_type == "24h":
            return "Your meeting is tomorrow"
        elif reminder_type == "1h":
            return "Your meeting starts in 1 hour"
        elif reminder_type == "30m":
            return "Your meeting starts in 30 minutes"
        else:
            return "You have an upcoming meeting"


# Global service instance
email_notification_service = EmailNotificationService() 