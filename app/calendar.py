# /app/calendar.py
import os.path
import datetime as dt

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import settings

# Scopes remain the same
SCOPES = ["https://www.googleapis.com/auth/calendar"]

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
    Schedules a new event on Google Calendar.
    
    Args:
        user_email: The email of the user to invite.
        property_address: The address of the property for the viewing.
        time_str: The ISO 8601 formatted start time for the event.

    Returns:
        The URL of the created Google Calendar event.
    """
    try:
        service = get_calendar_service()
        
        start_time = dt.datetime.fromisoformat(time_str)
        end_time = start_time + dt.timedelta(hours=1)

        event = {
            "summary": f"Property Viewing: {property_address}",
            "location": property_address,
            "description": f"Viewing scheduled for {property_address} with {user_email}.",
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "America/New_York", # Consider making this configurable
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "America/New_York",
            },
            # "attendees": [{"email": user_email}], # This line is removed
            # The above line is removed because standard service accounts cannot invite attendees
            # without special domain-wide delegation permissions from a Google Workspace admin.
            # By removing this, the event is created on the calendar, and the user can be
            # sent the link to the event.
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 24 * 60},
                    {"method": "popup", "minutes": 30},
                ],
            },
        }

        # Note: With service accounts, you must specify the calendarId of the calendar
        # you want to add events to. 'primary' refers to the primary calendar of
        # the service account itself, which is likely not what you want.
        # You need to share your own primary Google Calendar with the service account's email address
        # and grant it "Make changes to events" permission.
        # Then, replace 'primary' with your own email address.
        # For now, we will leave it as 'primary' but this is a critical next step for the user.
        
        # A placeholder for the user's actual calendar ID (their email)
        # You should share your main calendar with the service account's email.
        calendar_id = 'primary' 

        created_event = service.events().insert(calendarId=calendar_id, body=event).execute()
        return created_event.get("htmlLink")

    except HttpError as error:
        print(f"An error occurred: {error}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during scheduling: {e}")
        raise 