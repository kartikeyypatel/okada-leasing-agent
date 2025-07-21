import os
import asyncio
import re
from fastapi import FastAPI, HTTPException, Body, Request, UploadFile, File, BackgroundTasks, Form, Query, Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import datetime as dt

from app.models import ChatRequest, ChatResponse, User, DocumentUploadRequest, ScheduleRequest, AppointmentState, PropertySuggestionState
import app.rag as rag_module
import app.crm as crm_module
import app.history as history_module
from app.crm import create_or_update_user, get_user_by_email
from app.database import connect_to_mongo, close_mongo_connection, get_database
from app.config import settings
import app.calendar as calendar
from llama_index.llms.gemini import Gemini

app = FastAPI(title="Okada Leasing Agent API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    close_mongo_connection()

# In-memory storage for conversation states (in production, use Redis or database)
appointment_states = {}
property_suggestion_states = {}

async def get_appointment_state(user_id: str) -> Optional[AppointmentState]:
    """Get current appointment state for user."""
    return appointment_states.get(user_id)

async def set_appointment_state(user_id: str, state: AppointmentState):
    """Set appointment state for user."""
    appointment_states[user_id] = state

async def get_property_suggestion_state(user_id: str) -> Optional[PropertySuggestionState]:
    """Get current property suggestion state for user."""
    return property_suggestion_states.get(user_id)

async def set_property_suggestion_state(user_id: str, state: PropertySuggestionState):
    """Set property suggestion state for user."""
    property_suggestion_states[user_id] = state

async def clear_property_suggestion_state(user_id: str):
    """Clear property suggestion state for user."""
    if user_id in property_suggestion_states:
        del property_suggestion_states[user_id]

async def clear_appointment_state(user_id: str):
    """Clear appointment state for user."""
    if user_id in appointment_states:
        del appointment_states[user_id]

async def handle_appointment_scheduling(user_id: str, message: str, user_index) -> str:
    """Handle the multi-step appointment scheduling conversation."""
    current_state = await get_appointment_state(user_id)
    
    # Check if user wants to schedule an appointment
    schedule_keywords = ['schedule', 'appointment', 'book', 'viewing', 'visit', 'meet', 'tour']
    if not current_state and any(keyword in message.lower() for keyword in schedule_keywords):
        # Start the scheduling process
        state = AppointmentState(user_id=user_id, step="property_number")
        await set_appointment_state(user_id, state)
        
        return """📅 I'd be happy to help you schedule a property viewing appointment!

Let's get started:

🏢 Please provide the property number or address you'd like to visit.

You can say something like:
- "Property #1" 
- "15 W 38th St, Floor P7, Suite 702"
- "The property at 1412 Broadway"

What property would you like to schedule a viewing for?"""

    if not current_state:
        return None  # Not in scheduling flow
    
    # Handle each step of the scheduling process
    if current_state.step == "property_number":
        # Extract property information from user input
        property_info = await extract_property_info(message, user_index)
        if property_info:
            current_state.property_number = message
            current_state.property_address = property_info
            current_state.step = "purpose"
            await set_appointment_state(user_id, current_state)
            
            return f"""✅ Great! I found the property: {property_info}

🎯 What's the purpose of your visit?

Please choose one:
1. 📋 Property viewing/tour
2. 💼 Business meeting
3. 📋 Property inspection
4. 🤝 Lease discussion
5. 📞 Other (please specify)

Just type the number or describe your purpose."""
        else:
            return """❌ I couldn't find that property in our database.

Please provide a valid property number or address. You can:
- Browse available properties first by asking "show me available properties"
- Provide a specific address like "15 W 38th St"
- Use a property number if you have one

What property would you like to schedule a viewing for?"""

    elif current_state.step == "purpose":
        current_state.purpose = message
        current_state.step = "datetime"
        await set_appointment_state(user_id, current_state)
        
        return f"""✅ Purpose noted: {message}

📅 When would you like to schedule the appointment?

Please provide your preferred date and time in one of these formats:
- "Tomorrow at 2 PM"
- "January 25, 2025 at 10:30 AM"
- "Next Monday at 3 PM"
- "2025-01-25 14:30"

When would work best for you?"""

    elif current_state.step == "datetime":
        # Parse and validate the datetime
        parsed_datetime = await parse_datetime(message)
        if parsed_datetime:
            current_state.datetime = parsed_datetime
            current_state.step = "email"
            await set_appointment_state(user_id, current_state)
            
            return f"""✅ Date and time confirmed: {parsed_datetime}

📧 Please provide your email address for the calendar invitation and confirmation.

This email will be used to:
- Send you a Google Calendar invitation
- Send appointment confirmation
- Contact you if needed

What's your email address?"""
        else:
            return """❌ I couldn't understand that date and time format.

Please provide the date and time in a clear format like:
- "Tomorrow at 2 PM"
- "January 25, 2025 at 10:30 AM"
- "Next Monday at 3 PM"

When would you like to schedule the appointment?"""

    elif current_state.step == "email":
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, message.strip()):
            current_state.email = message.strip()
            current_state.step = "confirmation"
            await set_appointment_state(user_id, current_state)
            
            return f"""✅ Email confirmed: {message.strip()}

📋 Please review your appointment details:

🏢 Property: {current_state.property_address}
🎯 Purpose: {current_state.purpose}
📅 Date & Time: {current_state.datetime}
📧 Email: {current_state.email}

Type "CONFIRM" to schedule the appointment or "CANCEL" to start over.
You can also type "EDIT" to modify any details."""
        else:
            return """❌ Please provide a valid email address.

Examples of valid email formats:
- john.doe@example.com
- user@company.co
- name123@gmail.com

What's your email address?"""

    elif current_state.step == "confirmation":
        if message.upper() == "CONFIRM":
            # Create the calendar event
            try:
                print(f"Creating calendar event with:")
                print(f"  Email: {current_state.email}")
                print(f"  Property: {current_state.property_address}")
                print(f"  DateTime: {current_state.datetime}")
                
                event_url = calendar.schedule_viewing(
                    current_state.email,
                    current_state.property_address,
                    current_state.datetime
                )
                
                print(f"Calendar event created successfully: {event_url}")
                
                # Update user with scheduled event
                user = await crm_module.get_user_by_email(current_state.email)
                if user:
                    # Get existing scheduled events or create empty list
                    existing_events = user.scheduled_events if user.scheduled_events else []
                    existing_events.append(event_url)
                    
                    # Update user with new scheduled event
                    await crm_module.create_or_update_user(
                        email=current_state.email,
                        full_name=user.full_name,
                        preferences=user.preferences,
                        scheduled_events=existing_events
                    )
                else:
                    # Create new user if doesn't exist
                    await crm_module.create_or_update_user(
                        email=current_state.email,
                        full_name="Unknown User",  # Will be updated when user provides info
                        scheduled_events=[event_url]
                    )
                
                # Clear the appointment state
                await clear_appointment_state(user_id)
                
                return f"""🎉 Appointment scheduled successfully!

📅 Your appointment details:
🏢 Property: {current_state.property_address}
🎯 Purpose: {current_state.purpose}
📅 Date & Time: {current_state.datetime}
📧 Email: {current_state.email}

✅ A Google Calendar invitation has been sent to {current_state.email}
🔗 Calendar Event: {event_url}

📞 If you need to reschedule or have questions, please contact our office.

Is there anything else I can help you with?"""
                
            except Exception as e:
                print(f"Error creating calendar event: {e}")
                await clear_appointment_state(user_id)
                return """❌ Sorry, there was an error scheduling your appointment. Please try again or contact our office directly.

You can start over by saying "I want to schedule an appointment" """
                
        elif message.upper() == "CANCEL":
            await clear_appointment_state(user_id)
            return """❌ Appointment scheduling cancelled.

No worries! You can start a new appointment request anytime by saying "I want to schedule an appointment"

Is there anything else I can help you with?"""
            
        elif message.upper() == "EDIT":
            current_state.step = "property_number"
            await set_appointment_state(user_id, current_state)
            return """✏️ Let's start over with your appointment details.

🏢 Please provide the property number or address you'd like to visit."""
        else:
            return """Please respond with one of the following:
- "CONFIRM" to schedule the appointment
- "CANCEL" to cancel the scheduling
- "EDIT" to modify the details

What would you like to do?"""

    return None

async def extract_property_info(message: str, user_index) -> Optional[str]:
    """Extract property information from user message using RAG."""
    if not user_index:
        return None
    
    try:
        # Use retriever to find matching property
        retriever = user_index.as_retriever(
            similarity_top_k=2,  # Further reduced from 3 to 2 for faster property extraction
            response_mode="compact"
        )
        retrieved_nodes = await retriever.aretrieve(message)
        
        if retrieved_nodes:
            # Get the best matching property
            best_match = retrieved_nodes[0]
            # Extract property address from the node text
            node_text = best_match.text
            if "Property Address:" in node_text:
                address_part = node_text.split("Property Address:")[1].split(",")[0].strip()
                return address_part
            else:
                # Fallback: use first part of node text
                return node_text.split(",")[0].strip()
    except Exception as e:
        print(f"Error extracting property info: {e}")
    
    return None

async def parse_datetime(datetime_str: str) -> Optional[str]:
    """Parse datetime string into ISO format."""
    try:
        from dateutil import parser
        import datetime as dt
        
        # Try to parse the datetime string
        parsed_dt = parser.parse(datetime_str, fuzzy=True)
        
        # If no year specified, assume current year
        if parsed_dt.year == 1900:
            current_year = dt.datetime.now().year
            parsed_dt = parsed_dt.replace(year=current_year)
        
        # If the date is in the past, assume next year
        if parsed_dt < dt.datetime.now():
            parsed_dt = parsed_dt.replace(year=parsed_dt.year + 1)
        
        return parsed_dt.isoformat()
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        return None

async def extract_preferences_from_history(user_id: str) -> dict:
    """Extract property preferences from user's chat history."""
    preferences = {
        "budget": None,
        "size": None,
        "location": None,
        "property_type": None,
        "amenities": []
    }
    
    try:
        # Get user's chat history
        history = await history_module.get_user_history(user_id)
        if not history:
            return preferences
        
        # Combine all messages for analysis
        all_messages = ""
        for msg in history:
            if msg.get('role') == 'user':
                all_messages += msg.get('content', '') + " "
        
        if not all_messages.strip():
            return preferences
        
        # Use LLM to extract preferences
        llm = Gemini(model="models/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)
        
        extraction_prompt = f"""
        Analyze the following chat history and extract any property preferences mentioned by the user.
        Pay careful attention to any mentions of budget, size requirements, locations, property types, or specific amenities.
        
        Return ONLY a JSON object with the following structure (use null for missing information):
        {{
            "budget": "extracted budget range or null",
            "size": "extracted size preference (sq ft) or null",
            "location": "extracted location preference or null",
            "property_type": "extracted property type or null",
            "amenities": ["list", "of", "amenities", "mentioned"]
        }}
        
        Examples of what to look for:
        - Budget: "around $100,000 per month", "between $70K and $90K", "under $150,000 monthly"
        - Size: "about 10,000 square feet", "at least 15,000 SF", "between 8,000 and 12,000 SF"
        - Location: "Midtown", "near Times Square", "downtown area", "Manhattan"
        - Property Type: "office space", "retail storefront", "mixed-use", "open floor plan"
        - Amenities: "natural light", "conference rooms", "parking", "kitchen", "elevator access"
        
        Chat history: "{all_messages}"
        
        Important: Return ONLY the JSON object, no other text.
        """
        
        response = await llm.acomplete(extraction_prompt)
        extraction_result = str(response).strip()
        
        # Parse JSON response
        import json
        try:
            # Clean up the response - remove markdown code blocks if present
            if extraction_result.startswith("```json"):
                extraction_result = extraction_result.replace("```json", "").replace("```", "").strip()
            elif extraction_result.startswith("```"):
                extraction_result = extraction_result.replace("```", "").strip()
                
            extracted_info = json.loads(extraction_result)
            
            # Update preferences with extracted info
            for key in preferences.keys():
                if key in extracted_info and extracted_info[key] is not None and extracted_info[key] != "null":
                    preferences[key] = extracted_info[key]
                    
            print(f"Extracted preferences from history: {preferences}")
            
        except json.JSONDecodeError:
            print(f"Failed to parse extraction result: {extraction_result}")
        
    except Exception as e:
        print(f"Error extracting preferences from history: {e}")
        
    return preferences

async def handle_property_suggestion(user_id: str, message: str, user_index) -> Optional[str]:
    """Handle the multi-step property suggestion conversation."""
    current_state = await get_property_suggestion_state(user_id)
    
    # Check if user wants property suggestions
    suggestion_keywords = ['suggest', 'recommendation', 'recommend', 'property for me', 'find me', 'looking for', 'help me find', 'show me', 'need a property', 'want a property', 'search for', 'ideal property']
    
    # More comprehensive detection of property suggestion requests
    is_suggestion_request = False
    if not current_state:
        # Check for direct keywords
        if any(keyword in message.lower() for keyword in suggestion_keywords):
            is_suggestion_request = True
        # Check for question patterns about properties
        elif any(q in message.lower() for q in ['what property', 'which property', 'best property', 'good property']):
            is_suggestion_request = True
    
    if not current_state and is_suggestion_request:
        # Extract preferences from chat history
        history_preferences = await extract_preferences_from_history(user_id)
        
        # Start the suggestion process
        state = PropertySuggestionState(
            user_id=user_id, 
            step="budget",
            chat_history_preferences=history_preferences
        )
        await set_property_suggestion_state(user_id, state)
        
        # Prepare initial message based on history
        budget_context = ""
        if history_preferences["budget"]:
            budget_context = f"\n\nI noticed you previously mentioned a budget around {history_preferences['budget']}. Is this still your budget range?"
        
        return f"""🏢 I'd be happy to suggest some properties that match your needs!

Let's find your perfect property by understanding your requirements.

💰 First, what's your monthly budget range for rent?{budget_context}

For example:
- "Between $70,000 and $100,000 per month"
- "Around $120,000 monthly"
- "Maximum $150,000\""""

    if not current_state:
        return None  # Not in property suggestion flow
    
    # Handle each step of the suggestion process
    if current_state.step == "budget":
        current_state.budget = message
        current_state.step = "size"
        await set_property_suggestion_state(user_id, current_state)
        
        # Prepare size question based on history
        size_context = ""
        if current_state.chat_history_preferences["size"]:
            size_context = f"\n\nI noticed you previously mentioned a size preference around {current_state.chat_history_preferences['size']}. Is this still your preference?"
        
        return f"""✅ Budget noted: {message}

📏 What size space are you looking for (in square feet)?{size_context}

For example:
- "Around 10,000 SF"
- "Between 15,000 and 20,000 square feet"
- "At least 8,000 SF\""""
        
    elif current_state.step == "size":
        current_state.size_preference = message
        current_state.step = "location"
        await set_property_suggestion_state(user_id, current_state)
        
        # Prepare location question based on history
        location_context = ""
        if current_state.chat_history_preferences["location"]:
            location_context = f"\n\nI noticed you previously mentioned interest in {current_state.chat_history_preferences['location']}. Is this still your preferred location?"
        
        return f"""✅ Size preference noted: {message}

📍 Do you have a preferred location or neighborhood?{location_context}

For example:
- "Midtown Manhattan"
- "Near Times Square"
- "Downtown area"
- "No preference\""""
        
    elif current_state.step == "location":
        current_state.location_preference = message
        current_state.step = "type"
        await set_property_suggestion_state(user_id, current_state)
        
        # Prepare property type question based on history
        type_context = ""
        if current_state.chat_history_preferences["property_type"]:
            type_context = f"\n\nI noticed you previously mentioned interest in {current_state.chat_history_preferences['property_type']} properties. Is this still your preference?"
        
        return f"""✅ Location preference noted: {message}

🏗️ What type of property are you looking for?{type_context}

For example:
- "Office space"
- "Retail storefront"
- "Mixed-use property"
- "Open floor plan\""""
        
    elif current_state.step == "type":
        current_state.property_type = message
        current_state.step = "amenities"
        await set_property_suggestion_state(user_id, current_state)
        
        # Prepare amenities question based on history
        amenities_context = ""
        if current_state.chat_history_preferences["amenities"] and len(current_state.chat_history_preferences["amenities"]) > 0:
            amenities_list = ", ".join(current_state.chat_history_preferences["amenities"])
            amenities_context = f"\n\nI noticed you previously mentioned interest in properties with {amenities_list}. Are these still important to you?"
        
        return f"""✅ Property type noted: {message}

✨ Are there any specific amenities or features you're looking for?{amenities_context}

For example:
- "Elevator access"
- "Natural lighting"
- "Conference rooms"
- "Kitchen facilities"
- "None in particular\""""
        
    elif current_state.step == "amenities":
        current_state.amenities = message
        current_state.step = "additional"
        await set_property_suggestion_state(user_id, current_state)
        
        return f"""✅ Amenities noted: {message}

📝 Any additional requirements or preferences you'd like to mention?

For example:
- "Need parking spaces"
- "Prefer recently renovated"
- "Must be available immediately"
- "None\""""
        
    elif current_state.step == "additional":
        current_state.additional_requirements = message
        current_state.step = "analysis"
        await set_property_suggestion_state(user_id, current_state)
        
        # Prepare analysis message
        return f"""✅ Additional requirements noted: {message}

🔍 Thank you for providing all this information! I'm analyzing our property database to find the best matches for you..."""
        
    elif current_state.step == "analysis":
        # This is where we generate property suggestions based on all collected preferences
        if not user_index:
            await clear_property_suggestion_state(user_id)
            return "I'm sorry, but I don't have access to the property database at the moment. Please upload property documents first so I can provide personalized recommendations."
        
        # Combine all preferences into a comprehensive query
        preferences = {
            "budget": current_state.budget,
            "size": current_state.size_preference,
            "location": current_state.location_preference,
            "property_type": current_state.property_type,
            "amenities": current_state.amenities,
            "additional": current_state.additional_requirements
        }
        
        # Merge with history preferences where current preferences are empty
        for key, value in current_state.chat_history_preferences.items():
            if key in preferences and (preferences[key] is None or preferences[key] == "None" or preferences[key] == ""):
                preferences[key] = value
        
        # Generate property suggestions
        suggestions = await generate_property_suggestions(user_id, preferences, user_index)
        
        # Clear the state after generating suggestions
        await clear_property_suggestion_state(user_id)
        
        return suggestions
    
    return None

async def generate_property_suggestions(user_id: str, preferences: dict, user_index) -> str:
    """Generate property suggestions based on user preferences."""
    try:
        # Create a comprehensive query from preferences
        query_parts = []
        if preferences["budget"]:
            query_parts.append(f"Budget: {preferences['budget']}")
        if preferences["size"]:
            query_parts.append(f"Size: {preferences['size']}")
        if preferences["location"]:
            query_parts.append(f"Location: {preferences['location']}")
        if preferences["property_type"]:
            query_parts.append(f"Property type: {preferences['property_type']}")
        if preferences["amenities"]:
            query_parts.append(f"Amenities: {preferences['amenities']}")
        if preferences["additional"]:
            query_parts.append(f"Additional requirements: {preferences['additional']}")
            
        query = " ".join(query_parts)
        print(f"Property suggestion query: {query}")
        
        # Use direct retrieval to get relevant properties (optimized for speed)
        retriever = user_index.as_retriever(
            similarity_top_k=2,  # Further reduced from 3 to 2 for maximum speed
            response_mode="compact"  # Use compact mode for faster responses
        )
        retrieved_nodes = await retriever.aretrieve(query)
        
        if not retrieved_nodes or len(retrieved_nodes) == 0:
            return "I'm sorry, but I couldn't find any properties matching your criteria. Would you like to try with different preferences?"
        
        # Create context with retrieved properties
        context_parts = []
        for i, node in enumerate(retrieved_nodes):
            context_parts.append(f"Property {i+1}: {node.text}")
        
        full_context = "\n\n".join(context_parts)
        
        # Use LLM to analyze and recommend top 3 properties
        llm = Gemini(model="models/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)
        
        analysis_prompt = f"""
You are a professional real estate assistant. Based on the user's preferences and the available properties, recommend the TOP 3 best matches.

USER PREFERENCES:
- Budget: {preferences['budget']}
- Size: {preferences['size']}
- Location: {preferences['location']}
- Property Type: {preferences['property_type']}
- Amenities: {preferences['amenities']}
- Additional Requirements: {preferences['additional']}

AVAILABLE PROPERTIES:
{full_context}

ANALYSIS INSTRUCTIONS:
1. Carefully analyze each property against ALL user preferences
2. Prioritize properties that match the most important criteria (budget, size, location)
3. Consider the value proposition (price per square foot)
4. Look for properties that have the requested amenities
5. Select the TOP 3 properties that best match ALL requirements

FORMATTING REQUIREMENTS:
- Provide a brief introduction explaining your recommendations
- For each property, use this EXACT format:

🏢 [Property Address]
   📍 Location: Floor [Floor], Suite [Suite]  
   📐 Size: [Size] SF  
   💰 Monthly Rent: $[Amount]/month  
   ✨ Highlights: [2-3 key features that match user preferences]
   💯 Match Score: Explain briefly why this property is a good match

- Number the properties 1, 2, 3
- After listing the properties, provide a brief conclusion
- DO NOT use ** or any markdown formatting - use plain text only
- Use emojis and clean spacing to make it visually appealing

Provide your TOP 3 recommendations that best match the user's preferences:
"""

        response = await llm.acomplete(analysis_prompt)
        recommendations = str(response)
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating property suggestions: {e}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, but I encountered an error while generating property suggestions. Please try again or contact support if the issue persists."

async def extract_and_store_user_info(user_id: str, message: str):
    """Extract user information from conversation content and store it in CRM."""
    try:
        # Initialize LLM for information extraction
        llm = Gemini(model="models/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)
        
        # Create extraction prompt
        extraction_prompt = f"""
        Analyze the following user message and extract any personal information that can be used for CRM purposes.
        Return ONLY a JSON object with the following structure (use null for missing information):
        {{
            "full_name": "extracted full name or null",
            "company_name": "extracted company name or null", 
            "preferences": {{
                "property_type": "extracted property preference or null",
                "budget": "extracted budget information or null",
                "location": "extracted location preference or null",
                "bedrooms": "extracted bedroom preference or null",
                "other": "any other relevant preferences or null"
            }}
        }}
        
        User message: "{message}"
        
        Important: Return ONLY the JSON object, no other text.
        """
        
        # Get extraction result
        response = await llm.acomplete(extraction_prompt)
        extraction_result = str(response).strip()
        
        # Parse JSON response
        import json
        try:
            # Clean up the response - remove markdown code blocks if present
            if extraction_result.startswith("```json"):
                extraction_result = extraction_result.replace("```json", "").replace("```", "").strip()
            elif extraction_result.startswith("```"):
                extraction_result = extraction_result.replace("```", "").strip()
            
            extracted_info = json.loads(extraction_result)
        except json.JSONDecodeError:
            print(f"Failed to parse extraction result: {extraction_result}")
            return
        
        # Check if we have any useful information
        full_name = extracted_info.get("full_name")
        company_name = extracted_info.get("company_name")
        preferences = extracted_info.get("preferences", {})
        
        # Clean up preferences - remove null values
        clean_preferences = {k: v for k, v in preferences.items() if v is not None and v != "null"}
        
        # Only update if we have meaningful information
        if full_name or company_name or clean_preferences:
            print(f"Extracted info for {user_id}: name={full_name}, company={company_name}, preferences={clean_preferences}")
            
            # Update user information
            await crm_module.create_or_update_user(
                email=user_id,
                full_name=full_name,
                company_name=company_name,
                preferences=clean_preferences if clean_preferences else None
            )
            
    except Exception as e:
        print(f"Error extracting user info from message: {e}")
        # Don't raise the error - this is a background enhancement, shouldn't break chat

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for property inquiries."""
    import time
    from app.models import ProcessingMetadata
    
    start_time = time.time()
    processing_steps = []
    documents_retrieved = 0
    user_info_extracted = False
    rag_enabled = False
    tokens_used = None
    
    try:
        print(f"Chat request received - User: {request.user_id}, Message: '{request.message}'")
        processing_steps.append("request_received")
        
        # Extract and store user information from conversation
        if request.user_id:
            extraction_start = time.time()
            await extract_and_store_user_info(request.user_id, request.message)
            user_info_extracted = True
            processing_steps.append("user_info_extraction")
        
        # Get user index if available
        user_index = None
        if request.user_id:
            processing_steps.append("index_retrieval")
            user_index = await rag_module.get_user_index(request.user_id)
            print(f"Cached index found for {request.user_id}: {user_index is not None}")
            
            if not user_index:
                # Try to build index from user documents
                user_doc_dir = os.path.join("user_documents", request.user_id)
                print(f"Checking user document directory: {user_doc_dir}")
                
                if os.path.exists(user_doc_dir):
                    # Get all supported file types
                    all_files = []
                    for f in os.listdir(user_doc_dir):
                        if f.endswith(('.csv', '.pdf', '.txt', '.json')):
                            all_files.append(os.path.join(user_doc_dir, f))
                    
                    print(f"Found {len(all_files)} supported files: {[os.path.basename(f) for f in all_files]}")
                    
                    if all_files:
                        print(f"Building index for user {request.user_id} with {len(all_files)} files")
                        processing_steps.append("index_building")
                        user_index = await rag_module.build_user_index(request.user_id, all_files)
                        if user_index:
                            print(f"Successfully built index for user {request.user_id}")
                        else:
                            print(f"Failed to build index for user {request.user_id}")
                    else:
                        print(f"No supported files found in {user_doc_dir}")
                else:
                    print(f"User document directory does not exist: {user_doc_dir}")
            else:
                print(f"Using cached index for user {request.user_id}")

        # Check for appointment scheduling first
        if request.user_id:
            appointment_response = await handle_appointment_scheduling(request.user_id, request.message, user_index)
            if appointment_response:
                processing_steps.append("appointment_scheduling")
                
                # Calculate response time
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Estimate token usage
                tokens_used = len(request.message.split()) + len(appointment_response.split())
                
                # Create metadata
                metadata = ProcessingMetadata(
                    response_time_ms=round(response_time_ms, 2),
                    tokens_used=tokens_used,
                    documents_retrieved=0,
                    processing_steps=processing_steps,
                    rag_enabled=False,
                    user_info_extracted=user_info_extracted
                )
                
                # Save conversation history
                asyncio.create_task(history_module.add_message_to_history(
                    request.user_id, 
                    request.message, 
                    appointment_response
                ))
                
                return ChatResponse(
                    answer=appointment_response, 
                    schedule_details=None,
                    metadata=metadata
                )
            
            # Check for property suggestion
            print(f"Checking for property suggestion request: '{request.message}'")
            suggestion_response = await handle_property_suggestion(request.user_id, request.message, user_index)
            if suggestion_response:
                processing_steps.append("property_suggestion")
                
                # Calculate response time
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Estimate token usage
                tokens_used = len(request.message.split()) + len(suggestion_response.split())
                
                # Create metadata
                metadata = ProcessingMetadata(
                    response_time_ms=round(response_time_ms, 2),
                    tokens_used=tokens_used,
                    documents_retrieved=10,  # We retrieve 10 properties for suggestions
                    processing_steps=processing_steps,
                    rag_enabled=True,
                    user_info_extracted=user_info_extracted
                )
                
                # Save conversation history
                asyncio.create_task(history_module.add_message_to_history(
                    request.user_id, 
                    request.message, 
                    suggestion_response
                ))
                
                return ChatResponse(
                    answer=suggestion_response, 
                    schedule_details=None,
                    metadata=metadata
                )

        # Generate response
        processing_steps.append("response_generation")
        if user_index:
            rag_enabled = True
            processing_steps.append("rag_processing")
            
            print(f"User index found for {request.user_id}, creating chat engine...")
            
            # For property ranking queries, use a direct retrieval approach
            if any(keyword in request.message.lower() for keyword in ['top', 'lowest', 'highest', 'cheapest', 'most expensive', 'best', 'rank']):
                print("Detected ranking query, using direct retrieval approach...")
                
                # Get all documents directly from the index
                retriever = user_index.as_retriever(similarity_top_k=50)  # Get many documents
                retrieved_nodes = await retriever.aretrieve(request.message)
                
                # Create a comprehensive context with all property data
                context_parts = []
                for i, node in enumerate(retrieved_nodes):
                    context_parts.append(f"Property {i+1}: {node.text}")
                
                full_context = "\n\n".join(context_parts)
                
                # Use LLM directly with full context
                llm = Gemini(model="models/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)
                
                direct_prompt = f"""You are a professional real estate assistant. Based on the property data provided, answer the user's question with clean, easy-to-read responses.

PROPERTY DATA:
{full_context}

USER QUESTION: {request.message}

FORMATTING REQUIREMENTS:
- Use clean, professional formatting with proper spacing
- For property listings, use this EXACT format:

🏢 [Property Address]
   📍 Location: Floor [Floor], Suite [Suite]  
   📐 Size: [Size] SF  
   💰 Monthly Rent: $[Amount]/month  
   📧 Contact: [BROKER Email ID]

- For rankings/lists, number them clearly (1., 2., 3., etc.)
- Use emojis and clean spacing to make it visually appealing
- Add helpful context and summary information
- Include a brief conclusion or next steps when appropriate
- DO NOT use ** or any markdown formatting - use plain text only

INSTRUCTIONS:
- If asked for "top 5 low rent" or similar, find the 5 properties with the LOWEST Monthly Rent values
- Always provide the EXACT number requested
- Sort properly based on the criteria requested
- Make the response visually appealing and easy to scan
- Use only plain text formatting with emojis and proper spacing

Answer with cleanly formatted property details:"""

                response = await llm.acomplete(direct_prompt)
                response_text = str(response)
                documents_retrieved = len(retrieved_nodes)
                
                print(f"Direct retrieval response generated with {documents_retrieved} documents")
                
            else:
                # Use regular chat engine for conversational queries
                chat_engine = user_index.as_chat_engine(
                    chat_mode="context",
                    similarity_top_k=2,  # Further reduced from 3 to 2 for even faster retrieval
                    response_mode="compact",  # Use compact mode for faster responses
                    streaming=False,  # Disable streaming for faster batch responses
                    system_prompt="""You are a professional real estate assistant with access to a comprehensive property database containing 225+ properties. 

FORMATTING REQUIREMENTS - Always format your responses cleanly:

For property information, use this EXACT format:
🏢 [Property Address]
   📍 Location: Floor [Floor], Suite [Suite]  
   📐 Size: [Size] SF  
   💰 Monthly Rent: $[Amount]/month  
   📧 Contact: [BROKER Email ID]

For multiple properties:
- Number them clearly (1., 2., 3., etc.)
- Use emojis and clean spacing to make it visually appealing
- Add helpful context and summary information
- Include brief conclusions or next steps when appropriate
- DO NOT use ** or any markdown formatting - use plain text only

IMPORTANT GUIDELINES:
1. For rent queries: Use the "Monthly Rent" column to sort properties
2. For size queries: Use the "Size (SF)" column  
3. For location queries: Use the "Property Address" column
4. Always provide the EXACT number requested (e.g., if asked for top 5, provide exactly 5)
5. Include specific details: Address, suite, monthly rent, size
6. Make responses visually appealing and easy to scan
7. Use only plain text formatting with emojis and proper spacing

You have access to properties across multiple buildings including:
- 36 W 36th St, 15 W 38th St, 1412 Broadway, 314 E 34th St, 221-223 W 37th St
- 14 E 44th St, 202 W 40th St, 345 Seventh Avenue, and many more

Always provide cleanly formatted, professional responses that are easy to read and visually appealing using plain text only."""
                )
                
                print(f"Chat engine created, processing message: {request.message}")
                
                # Get response with RAG
                response = await chat_engine.achat(request.message)
                response_text = str(response)
                
                # Try to get document count from response metadata if available
                if hasattr(response, 'source_nodes'):
                    documents_retrieved = len(response.source_nodes)
                    print(f"Retrieved {documents_retrieved} source nodes")
                elif hasattr(response, 'sources'):
                    documents_retrieved = len(response.sources)
                    print(f"Retrieved {documents_retrieved} sources")
                else:
                    # Estimate based on typical RAG retrieval
                    documents_retrieved = 20
                    print("No source information available, estimating 20 documents")
                
        else:
            print(f"No user index found for {request.user_id}")
            response_text = "Please upload property documents first so I can help you with property information."
        
        # Save conversation history
        if request.user_id:
            processing_steps.append("history_saving")
            asyncio.create_task(history_module.add_message_to_history(
                request.user_id, 
                request.message, 
                response_text
            ))
        
        # Calculate response time
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Estimate token usage (rough approximation)
        tokens_used = len(request.message.split()) + len(response_text.split())
        
        # Create metadata
        metadata = ProcessingMetadata(
            response_time_ms=round(response_time_ms, 2),
            tokens_used=tokens_used,
            documents_retrieved=documents_retrieved,
            processing_steps=processing_steps,
            rag_enabled=rag_enabled,
            user_info_extracted=user_info_extracted
        )
        
        return ChatResponse(
            answer=response_text, 
            schedule_details=None,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        error_message = "I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists."
        
        # Save error to conversation history
        if request.user_id:
            asyncio.create_task(history_module.add_message_to_history(
                request.user_id, 
                request.message, 
                error_message
            ))
        
        # Calculate response time even for errors
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        processing_steps.append("error_handling")
        
        # Create error metadata
        metadata = ProcessingMetadata(
            response_time_ms=round(response_time_ms, 2),
            tokens_used=len(request.message.split()) + len(error_message.split()),
            documents_retrieved=documents_retrieved,
            processing_steps=processing_steps,
            rag_enabled=rag_enabled,
            user_info_extracted=user_info_extracted
        )
        
        return ChatResponse(
            answer=error_message, 
            schedule_details=None,
            metadata=metadata
        )

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and process a document for a specific user."""
    import time
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_exts = ['.csv', '.pdf', '.txt', '.json']
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="Only CSV, PDF, TXT, and JSON files are supported")
        
        # Create user directory
        user_dir = os.path.join("user_documents", user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(user_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Start background indexing
        background_tasks.add_task(index_user_document, user_id, file_path)
        
        # Calculate processing time
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return {
            "message": f"File {file.filename} uploaded successfully for user {user_id}",
            "filename": file.filename,
            "status": "processing",
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "file_size_bytes": len(content),
                "file_type": ext,
                "processing_steps": ["validation", "directory_creation", "file_save", "indexing_queued"]
            }
        }
        
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def index_user_document(user_id: str, file_path: str):
    """Background task to index a user's document."""
    try:
        # Build index for this user
        await rag_module.build_user_index(user_id, [file_path])
        print(f"Successfully indexed document for user {user_id}")
        
    except Exception as e:
        print(f"Error indexing document for user {user_id}: {e}")

@app.post("/api/documents/load")
async def load_documents(request: DocumentUploadRequest):
    """Load specific documents for a user."""
    try:
        user_dir = os.path.join("user_documents", request.user_id)
        
        if not os.path.exists(user_dir):
            raise HTTPException(status_code=404, detail=f"No documents found for user {request.user_id}")
        
        # Get file paths
        file_paths = []
        if request.filenames:
            # Load specific files
            for filename in request.filenames:
                file_path = os.path.join(user_dir, filename)
                if os.path.exists(file_path):
                    file_paths.append(file_path)
        else:
            # Load all CSV files
            file_paths = [
                os.path.join(user_dir, f) 
                for f in os.listdir(user_dir) 
                if f.endswith('.csv')
            ]
        
        if not file_paths:
            raise HTTPException(status_code=404, detail="No CSV files found")
        
        # Build index
        index = await rag_module.build_user_index(request.user_id, file_paths)
        
        if index:
            return {
                "message": f"Successfully loaded {len(file_paths)} documents for user {request.user_id}",
                "files_loaded": [os.path.basename(path) for path in file_paths]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to build index")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users")
async def create_user(user: User):
    """Create a new user."""
    import time
    start_time = time.time()
    
    try:
        result = await crm_module.create_or_update_user(user.email, user.full_name)
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        if result:
            return {
                "message": "User created successfully", 
                "user": user,
                "metadata": {
                    "processing_time_ms": round(processing_time_ms, 2),
                    "operation": "create_user",
                    "processing_steps": ["validation", "database_insert", "company_association"]
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create user")
    except Exception as e:
        print(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{email}")
async def get_user(email: str):
    """Get user by email."""
    try:
        user = await crm_module.get_user_by_email(email)
        if user:
            return user
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        print(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/users/{email}")
async def update_user(email: str, user_data: dict):
    """Update user information."""
    try:
        # Check if user exists
        existing_user = await crm_module.get_user_by_email(email)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Extract update fields
        full_name = user_data.get("full_name")
        company_name = user_data.get("company_name")
        preferences = user_data.get("preferences")
        
        # Update user
        updated_user = await crm_module.create_or_update_user(
            email=email,
            full_name=full_name,
            company_name=company_name,
            preferences=preferences
        )
        
        return {"message": "User updated successfully", "user": updated_user}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/users/{email}")
async def delete_user(email: str):
    """Delete user by email."""
    try:
        success = await crm_module.delete_user_by_email(email)
        if success:
            return {"message": f"User {email} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def list_users(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    """List all users with pagination."""
    try:
        db = get_database()
        user_collection = db["users"]
        
        # Get total count
        total = await user_collection.count_documents({})
        
        # Get users with pagination
        cursor = user_collection.find({}).skip(skip).limit(limit)
        users_data = await cursor.to_list(length=limit)
        
        # Convert to User models
        users = [User(**user_data) for user_data in users_data]
        
        return {
            "users": users,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        print(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def list_companies(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    """List all companies with pagination."""
    try:
        db = get_database()
        company_collection = db["companies"]
        
        # Get total count
        total = await company_collection.count_documents({})
        
        # Get companies with pagination
        cursor = company_collection.find({}).skip(skip).limit(limit)
        companies_data = await cursor.to_list(length=limit)
        
        # Convert to Company models
        from app.models import Company
        companies = [Company(**company_data) for company_data in companies_data]
        
        return {
            "companies": companies,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        print(f"Error listing companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/schedule")
async def schedule_viewing(request: ScheduleRequest):
    """Schedule a property viewing."""
    try:
        # Create calendar event
        event_url = calendar.schedule_viewing(
            request.email,
            request.address,
            request.time
        )
        
        if event_url:
            # Update user with scheduled event
            user = await crm_module.get_user_by_email(request.email)
            if user:
                user.scheduled_events.append(event_url)
                await crm_module.create_or_update_user(user.email, user.full_name, user.preferences, user.scheduled_events)
            
            return {
                "message": "Viewing scheduled successfully",
                "event_url": event_url,
                "address": request.address,
                "time": request.time
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create calendar event")
            
    except Exception as e:
        print(f"Error scheduling viewing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_docs")
async def upload_docs(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload docs (PDF/TXT/CSV/JSON) to populate the RAG base."""
    return await upload_document(background_tasks, user_id, file)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Accepts user message, returns LLM response with optional RAG-enhanced context."""
    return await chat(request)

@app.post("/crm/create_user")
async def crm_create_user(user: User):
    """Creates a new user profile with provided details."""
    return await create_user(user)

@app.put("/crm/update_user")
async def crm_update_user(user_data: dict):
    """Updates user information by user ID."""
    email = user_data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    return await update_user(email, user_data)

@app.get("/crm/conversations/{user_id}")
async def crm_get_conversation_history(user_id: str, tag: Optional[str] = Query(None), category: Optional[str] = Query(None)):
    """Fetch full conversation history for a user."""
    return await get_conversation_history(user_id, tag, category)

@app.post("/reset")
async def reset_endpoint(user_id: Optional[str] = None):
    """Clears conversation memory (optional: per user)."""
    return await reset_data(user_id)

@app.post("/api/reset")
async def reset_data(user_id: Optional[str] = None):
    """Reset user data or all data."""
    try:
        if user_id:
            # Reset specific user
            success = await rag_module.clear_user_index(user_id)
            if success:
                return {"message": f"Data reset successfully for user {user_id}"}
            else:
                raise HTTPException(status_code=500, detail="Failed to reset user data")
        else:
            # Reset all data
            rag_module.clear_index()
            return {"message": "All data reset successfully"}
            
    except Exception as e:
        print(f"Error resetting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/list/{user_id}")
async def list_user_documents(user_id: str):
    """List all CSV documents for a user."""
    user_dir = os.path.join("user_documents", user_id)
    if not os.path.exists(user_dir):
        return {"documents": []}
    docs = [f for f in os.listdir(user_dir) if f.endswith('.csv')]
    return {"documents": docs}

@app.post("/api/conversations/{user_id}/messages/{message_index}/tag")
async def tag_message(user_id: str = Path(...), message_index: int = Path(...), tags: Optional[List[str]] = None, category: Optional[str] = None):
    """Update tags and/or category for a specific message in the user's most recent conversation."""
    success = await history_module.update_message_tags_and_category(user_id, message_index, tags, category)
    if success:
        return {"status": "success"}
    else:
        return {"status": "error", "detail": "Message not found or update failed."}

@app.get("/api/conversations/{user_id}")
async def get_conversation_history(user_id: str, tag: Optional[str] = Query(None), category: Optional[str] = Query(None)):
    """Return the user's chat history, optionally filtered by tag or category."""
    try:
        history = await history_module.get_user_history(user_id)
        if tag:
            history = [msg for msg in history if tag in (msg.get('tags') or [])]
        if category:
            history = [msg for msg in history if msg.get('category') == category]
        return {"history": history}
    except Exception as e:
        print(f"Error fetching conversation history for {user_id}: {e}")
        return {"history": []}

@app.get("/api/documents/status")
async def document_status(user_id: str = Query(...)):
    """
    Return the status of document indexing for a user.
    """
    import time
    start_time = time.time()
    
    try:
        index_exists = await rag_module.user_index_exists(user_id)
        status = "ready" if index_exists else "processing"
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return {
            "status": status,
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "user_id": user_id,
                "index_exists": index_exists,
                "processing_steps": ["index_check", "status_determination"]
            }
        }
    except Exception as e:
        print(f"Error checking document status for {user_id}: {e}")
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return {
            "status": "error", 
            "detail": str(e),
            "metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "user_id": user_id,
                "processing_steps": ["index_check", "error_handling"]
            }
        }

# Static file serving
class SPAStaticFiles(StaticFiles):
    """Serve static files with SPA support (fallback to index.html)."""
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex

# Mount the static directory at the root
app.mount("/", SPAStaticFiles(directory="static", html=True), name="static-files")