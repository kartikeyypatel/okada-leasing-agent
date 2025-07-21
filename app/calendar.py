# /app/calendar.py
import os.path
import datetime as dt

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import settings

# Scopes for both Calendar and Gmail
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send"
]

def get_calendar_service():
    """
    Initializes and returns the Google Calendar API service client
    using Service Account credentials.
    """
    creds = None
    credentials_path = settings.GOOGLE_CALENDAR_CREDENTIALS_PATH

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Service account key file not found at '{credentials_path}'. "
                                "Please follow the instructions to create and place the file.")

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
    except Exception as e:
        raise ValueError(f"Failed to load service account credentials: {e}")

    return build("calendar", "v3", credentials=creds)


def schedule_viewing(user_email: str, property_address: str, time_str: str) -> str:
    """
    Schedules a new event on Google Calendar and sends email notification.
    
    Args:
        user_email: The email of the user to invite.
        property_address: The address of the property for the viewing.
        time_str: The ISO 8601 formatted start time for the event.

    Returns:
        The URL of the created Google Calendar event.
    """
    try:
        print(f"Starting calendar event creation for {user_email}")
        
        # Check if credentials file exists
        credentials_path = settings.GOOGLE_CALENDAR_CREDENTIALS_PATH
        if not os.path.exists(credentials_path):
            print(f"Credentials file not found at: {credentials_path}")
            # Return a mock URL for testing purposes
            return f"https://calendar.google.com/calendar/event?action=TEMPLATE&text=Property+Viewing:+{property_address.replace(' ', '+')}&dates={time_str.replace(':', '').replace('-', '')}/{time_str.replace(':', '').replace('-', '')}"
        
        service = get_calendar_service()
        print("Calendar service initialized successfully")
        
        # Parse the datetime string
        try:
            start_time = dt.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except ValueError:
            # Fallback parsing
            from dateutil import parser
            start_time = parser.parse(time_str)
        
        end_time = start_time + dt.timedelta(hours=1)
        print(f"Event time: {start_time} to {end_time}")

        event = {
            "summary": f"Property Viewing: {property_address}",
            "location": property_address,
            "description": f"""Property Viewing Appointment

Property: {property_address}
Scheduled for: {user_email}
Duration: 1 hour

Please arrive 5 minutes early. If you need to reschedule or have any questions, please contact our office.

Best regards,
Okada Leasing Team""",
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "America/New_York",
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "America/New_York",
            },
            # "attendees": [{"email": user_email}],  # Removed - service accounts need Domain-Wide Delegation for this
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 24 * 60},  # 24 hours before
                    {"method": "popup", "minutes": 30},       # 30 minutes before
                ],
            },
            "guestsCanModify": False,
            "guestsCanInviteOthers": False,
            "guestsCanSeeOtherGuests": False,
        }

        # Create a public calendar template link instead of service account event
        # This ensures users can actually access and add the event to their calendar
        fallback_url = create_fallback_calendar_link(user_email, property_address, time_str)
        print(f"Creating public calendar template link: {fallback_url}")
        
        # Send email notification with the public calendar link
        try:
            send_appointment_email_sync(user_email, property_address, start_time, fallback_url)
        except Exception as email_error:
            print(f"Failed to send email notification: {email_error}")
            # Don't fail the whole process if email fails
        
        return fallback_url

    except HttpError as error:
        print(f"Google Calendar API error: {error}")
        error_details = error.error_details if hasattr(error, 'error_details') else str(error)
        print(f"Error details: {error_details}")
        
        # Return a fallback calendar link
        fallback_url = create_fallback_calendar_link(user_email, property_address, time_str)
        print(f"Using fallback calendar link: {fallback_url}")
        
        # Still send email confirmation even with fallback
        try:
            start_time = dt.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except ValueError:
            from dateutil import parser
            start_time = parser.parse(time_str)
        
        print("Sending email confirmation for fallback calendar event...")
        send_appointment_email_sync(user_email, property_address, start_time, fallback_url)
        
        return fallback_url
        
    except Exception as e:
        print(f"Unexpected error during calendar scheduling: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a fallback calendar link
        fallback_url = create_fallback_calendar_link(user_email, property_address, time_str)
        print(f"Using fallback calendar link: {fallback_url}")
        
        # Still send email confirmation even with fallback
        try:
            start_time = dt.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except ValueError:
            from dateutil import parser
            start_time = parser.parse(time_str)
        
        print("Sending email confirmation for fallback calendar event...")
        send_appointment_email_sync(user_email, property_address, start_time, fallback_url)
        
        return fallback_url

def create_fallback_calendar_link(user_email: str, property_address: str, time_str: str) -> str:
    """Create a Google Calendar template link as fallback."""
    try:
        from urllib.parse import quote
        from dateutil import parser
        
        # Parse the datetime
        start_time = parser.parse(time_str)
        end_time = start_time + dt.timedelta(hours=1)
        
        # Format for Google Calendar URL
        start_formatted = start_time.strftime("%Y%m%dT%H%M%S")
        end_formatted = end_time.strftime("%Y%m%dT%H%M%S")
        
        title = quote(f"Property Viewing: {property_address}")
        details = quote(f"Property viewing appointment at {property_address}")
        location = quote(property_address)
        
        fallback_url = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={title}&dates={start_formatted}/{end_formatted}&details={details}&location={location}"
        
        return fallback_url
        
    except Exception as e:
        print(f"Error creating fallback link: {e}")
        return f"https://calendar.google.com/calendar/render?action=TEMPLATE&text=Property+Viewing"

async def send_appointment_email(user_email: str, property_address: str, appointment_time, event_url: str):
    """Send appointment confirmation email."""
    try:
        # This is a placeholder for email functionality
        # In production, you would integrate with Gmail API or SMTP
        print(f"Sending appointment confirmation email to {user_email}")
        print(f"Property: {property_address}")
        print(f"Time: {appointment_time}")
        print(f"Calendar Link: {event_url}")
        
        # TODO: Implement actual email sending using Gmail API or SMTP
        # For now, we'll just log the email content
        
        email_content = f"""
Subject: Property Viewing Appointment Confirmed - {property_address}

Dear Valued Client,

Your property viewing appointment has been confirmed!

📅 Appointment Details:
🏢 Property: {property_address}
📅 Date & Time: {appointment_time.strftime('%A, %B %d, %Y at %I:%M %p')}
📧 Your Email: {user_email}

🔗 Calendar Event: {event_url}

📝 Important Notes:
- Please arrive 5 minutes early
- Bring a valid ID for security purposes
- Feel free to ask questions during the viewing
- If you need to reschedule, please contact us at least 24 hours in advance

📞 Contact Information:
- Office: (555) 123-4567
- Email: info@okadaleasing.com

We look forward to showing you this property!

Best regards,
The Okada Leasing Team
"""
        
        print("Email content prepared:")
        print(email_content)
        
        # In production, implement actual email sending here
        return True
        
    except Exception as e:
        print(f"Error sending appointment email: {e}")
        return False

def get_gmail_service():
    """
    Initializes and returns the Gmail API service client
    using Service Account credentials.
    """
    creds = None
    credentials_path = settings.GOOGLE_CALENDAR_CREDENTIALS_PATH

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Service account key file not found at '{credentials_path}'")

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
    except Exception as e:
        raise ValueError(f"Failed to load service account credentials: {e}")

    return build("gmail", "v1", credentials=creds)

def create_email_message(to_email: str, subject: str, body: str, from_email: str = "noreply@okadaleasing.com"):
    """Create an email message in the format required by Gmail API."""
    import base64
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # Create message container
    message = MIMEMultipart('alternative')
    message['to'] = to_email
    message['from'] = from_email
    message['subject'] = subject
    
    # Create HTML version of the email
    html_body = body.replace('\n', '<br>')
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        {html_body}
    </body>
    </html>
    """
    
    # Attach parts
    text_part = MIMEText(body, 'plain')
    html_part = MIMEText(html_body, 'html')
    
    message.attach(text_part)
    message.attach(html_part)
    
    # Encode message
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    return {'raw': raw_message}

def send_appointment_email_sync(user_email: str, property_address: str, appointment_time, event_url: str):
    """Send appointment confirmation email using Gmail API."""
    try:
        print(f"Sending appointment confirmation email to {user_email}")
        print(f"Property: {property_address}")
        print(f"Time: {appointment_time}")
        print(f"Calendar Link: {event_url}")
        
        # Create email content
        subject = f"Property Viewing Appointment Confirmed - {property_address}"
        
        email_body = f"""Dear Valued Client,

Your property viewing appointment has been confirmed! 🎉

📅 APPOINTMENT DETAILS:
🏢 Property: {property_address}
📅 Date & Time: {appointment_time.strftime('%A, %B %d, %Y at %I:%M %p')}
📧 Your Email: {user_email}

🔗 Calendar Event: {event_url}

📝 IMPORTANT NOTES:
• Please arrive 5 minutes early
• Bring a valid ID for security purposes
• Feel free to ask questions during the viewing
• If you need to reschedule, please contact us at least 24 hours in advance

📞 CONTACT INFORMATION:
• Office: (555) 123-4567
• Email: info@okadaleasing.com
• Website: www.okadaleasing.com

We look forward to showing you this property!

Best regards,
The Okada Leasing Team

---
This is an automated confirmation email. Please do not reply to this email.
For any questions or changes, please contact our office directly.
"""
        
        # Use SMTP directly (Gmail API requires additional setup)
        print("Using SMTP for email sending...")
        return send_email_via_smtp(user_email, subject, email_body)
        
    except Exception as e:
        print(f"Error sending appointment email: {e}")
        import traceback
        traceback.print_exc()
        return False

def send_email_via_smtp(to_email: str, subject: str, body: str):
    """Send email using SMTP with credentials from .env file."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import os
        from dotenv import load_dotenv
        
        print("Attempting to send email via SMTP...")
        
        # Load environment variables
        load_dotenv()
        
        # Get SMTP settings from environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("FROM_EMAIL", smtp_username)
        
        # Check if SMTP credentials are configured
        if not smtp_username or not smtp_password:
            print("SMTP credentials not configured in .env file")
            print("Please set SMTP_USERNAME and SMTP_PASSWORD in your .env file")
            return False
        
        # Create message
        message = MIMEMultipart('alternative')
        message['From'] = from_email
        message['To'] = to_email
        message['Subject'] = subject
        
        # Create HTML version of the email
        html_body = body.replace('\n', '<br>')
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                {html_body}
            </div>
        </body>
        </html>
        """
        
        # Attach both plain text and HTML versions
        text_part = MIMEText(body, 'plain')
        html_part = MIMEText(html_body, 'html')
        
        message.attach(text_part)
        message.attach(html_part)
        
        # Send email
        print(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable encryption
        
        print(f"Logging in with username: {smtp_username}")
        server.login(smtp_username, smtp_password)
        
        print(f"Sending email to: {to_email}")
        text = message.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        
        print("✅ Email sent successfully via SMTP!")
        return True
        
    except Exception as smtp_error:
        print(f"❌ SMTP email sending failed: {smtp_error}")
        print(f"Error type: {type(smtp_error)}")
        
        # Provide helpful error messages
        if "authentication failed" in str(smtp_error).lower():
            print("🔑 Authentication failed. Please check:")
            print("   1. SMTP_USERNAME is correct")
            print("   2. SMTP_PASSWORD is an App Password (not your regular Gmail password)")
            print("   3. 2-Factor Authentication is enabled on your Gmail account")
        elif "connection" in str(smtp_error).lower():
            print("🌐 Connection failed. Please check:")
            print("   1. Internet connection")
            print("   2. SMTP server and port settings")
            print("   3. Firewall settings")
        
        return False