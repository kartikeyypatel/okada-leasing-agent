# /app/main.py
import contextlib
import base64
import os
import asyncio
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.schema import MetadataMode
import json
import re
from typing import Optional, Literal
import datetime as dt

# --- NEW: Background Task Imports ---
from fastapi import UploadFile, File, BackgroundTasks, Form

from app.models import ChatRequest, ChatResponse, User, DocumentUploadRequest, ScheduleRequest
import app.rag as rag_module # Import the module itself
import app.crm as crm_module # Import the crm module itself
import app.history as history_module # Import the history module
from app.crm import create_or_update_user, get_user_by_email
from app.database import connect_to_mongo, close_mongo_connection, get_database
import app.calendar as calendar
import time

# --- NEW: Global state for indexing status ---
indexing_status = {"status": "idle", "message": "No active indexing jobs."}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    await connect_to_mongo()
    # Document indexing will now only happen on user action (e.g., upload)
    yield
    # On shutdown
    close_mongo_connection()

app = FastAPI(title="Okada Leasing Agent API", lifespan=lifespan)

# --- Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Adds a custom X-Process-Time header to all API responses,
    containing the time taken to process the request.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f} sec"
    return response

# --- CORS Middleware ---
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://localhost:5173", # Default Vite Port
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not rag_module.rag_index:
             raise HTTPException(status_code=503, detail="RAG index is not ready. Please upload documents first.")

        # --- NEW: Load persistent history from DB ---
        if request.user_id:
            persistent_history = await history_module.get_user_history(request.user_id)
            # --- NEW: Load user's CRM profile for context ---
            user_profile = await crm_module.get_user_profile(request.user_id)
        else:
            persistent_history = []
            user_profile = None

        # Combine the persistent history with the current session's history from the client
        # The client history is considered the most up-to-date for the current interaction.
        # We assume the client sends the most recent messages.
        combined_history = persistent_history + request.history

        # Deduplicate while preserving order, in case of overlap
        seen = set()
        unique_history = []
        for msg in combined_history:
            msg_tuple = tuple(sorted(msg.items()))
            if msg_tuple not in seen:
                seen.add(msg_tuple)
                unique_history.append(msg)
        # Sort by timestamp if available, but for now just use it as is.
        # This simple combination assumes client sends a continuation.

        # --- REVISED: Step 1 - Classify intent and route logic ---
        query_intent = await classify_query(request.message)
        intent_type = query_intent.get("type")

        # --- Intent: Scheduling ---
        if intent_type == "scheduling":
            schedule_details = await extract_schedule_details(request.message)
            
            # Check if we have complete scheduling information
            if schedule_details and schedule_details.get("address") and schedule_details.get("time"):
                # Format the response with clear details
                address = schedule_details.get("address")
                time_str = schedule_details.get("time")
                
                try:
                    # Parse and format the time for better display
                    import datetime as dt
                    parsed_time = dt.datetime.fromisoformat(time_str)
                    formatted_time = parsed_time.strftime("%A, %B %d, %Y at %I:%M %p")
                    
                    response_text = f"""Perfect! I have all the details for your appointment. Please review and confirm:

**Property Viewing**
📍 **Location:** {address}
🕐 **Date & Time:** {formatted_time}
⏱️ **Duration:** 60 minutes

Would you like me to confirm this appointment?"""
                except (ValueError, AttributeError):
                    response_text = f"""I can help schedule your appointment. Here are the details:

**Property Viewing**
📍 **Location:** {address}
🕐 **Time:** {time_str}

Would you like me to confirm this appointment?"""
                
                return ChatResponse(answer=response_text, schedule_details=schedule_details)
            else:
                # Ask for missing information
                missing_info = []
                if not schedule_details or not schedule_details.get("address"):
                    missing_info.append("property address")
                if not schedule_details or not schedule_details.get("time"):
                    missing_info.append("preferred date and time")
                
                if missing_info:
                    missing_str = " and ".join(missing_info)
                    response_text = f"I'd be happy to help you schedule a property viewing! To book your appointment, I'll need the {missing_str}. Could you please provide these details?"
                else:
                    response_text = "I can certainly help you book an appointment. Which property are you interested in viewing, and when would you like to see it?"
                
                return ChatResponse(answer=response_text, schedule_details=None)

        # --- Build System Prompt (for all other intents) ---
        system_prompt_parts = [
            "You are a helpful and highly intelligent leasing agent assistant for Okada.",
            "Your goal is to answer questions accurately and conversationally.",
            "Format responses for clarity and readability using markdown.",
            "When displaying details for a single property, use a clear 'Key: Value' format with each pair on a new line. Use markdown bold for the key. Example:\n**Address**: 123 Main St\n**Rent**: $5000/month",
            "When displaying a list of multiple properties, use a bulleted list. Each item should just be the property address and one or two key features (e.g., rent or size)."
        ]

        if user_profile:
            system_prompt_parts.append(f"You are speaking with {user_profile.full_name or 'a client'}.")
            if user_profile.preferences:
                # Convert preferences dict to a readable string
                prefs_str = ", ".join([f"{k}: {v}" for k, v in user_profile.preferences.items()])
                system_prompt_parts.append(f"This user has previously expressed these preferences: {prefs_str}. Keep these in mind and be proactive.")

        system_prompt_parts.append(
            "If the answer to a question is not in the provided document context, do NOT just say 'I don't have that information.' "
            "Instead, do the following: "
            "1. Acknowledge what the user is asking for (e.g., 'I understand you're looking for information on kitchen sizes.'). "
            "2. State that the specific detail is not available in the provided listings. "
            "3. Reassure them that you have noted their preference for future reference. "
            "4. Proactively ask another question to guide the conversation forward (e.g., 'Is there another feature I can look for, such as the number of bathrooms or proximity to a park?')."
        )
        
        system_prompt = "\n".join(system_prompt_parts)

        # Convert the incoming list of dicts into LlamaIndex ChatMessage objects
        chat_history = [ChatMessage(role=msg['role'], content=msg['content']) for msg in unique_history]

        # --- REVISED: Step 2 - Handle remaining intents ---

        # --- Intent: Top N Property Inquiry ---
        if intent_type == "top_n_inquiry":
            # Pass the original user message to the function
            top_n_data = await find_top_n_properties(request.message)
            if not top_n_data:
                response_text = "I couldn't retrieve the property data to determine the top listings. Please try again later."
            else:
                response_text = top_n_data
            
            response = response_text # Bypass chat engine for direct data response

        # --- Intent: Standard Property Inquiry (RAG) ---
        elif intent_type == "property_inquiry":
            system_prompt_parts.append("To answer the user's question, you MUST use the provided context from the document search. Do not use any other knowledge.")
            
            fusion_retriever = rag_module.get_fusion_retriever()
            if not fusion_retriever:
                raise HTTPException(status_code=503, detail="RAG retriever is not ready. Please upload documents first.")

            chat_engine = ContextChatEngine.from_defaults(
                retriever=fusion_retriever,
                chat_history=chat_history,
                system_prompt="\n".join(system_prompt_parts),
            )
            response = await chat_engine.achat(request.message)

        # --- Intent: Default / General Chat ---
        else:
            system_prompt_parts.append("To answer, rely on the user's details and the conversation history. Do not ask for a document search.")
            
            llm = Settings.llm
            messages_for_llm = [ChatMessage(role="system", content="\n".join(system_prompt_parts))] + chat_history + [ChatMessage(role="user", content=request.message)]
            response = await llm.achat(messages_for_llm)

        # Uniformly extract the string content from the response object
        answer_content: str
        if isinstance(response, str):
            answer_content = response
        elif hasattr(response, "response") and isinstance(getattr(response, "response", None), str):  # For AgentChatResponse
            answer_content = getattr(response, "response", "")
        elif hasattr(response, "message") and hasattr(response.message, "content"):  # For ChatResponse
            answer_content = response.message.content or ""
        else:
            answer_content = str(response)

        # --- Update CRM & History in the background ---
        if request.user_id:  # Only run if there is a known user
            # Save the conversation turn to history
            asyncio.create_task(
                history_module.add_message_to_history(
                    user_email=request.user_id,
                    user_message=request.message,
                    assistant_message=answer_content,
                )
            )
            # Extract user details from message
            asyncio.create_task(extract_and_update_crm_details(request.user_id, request.message))
            # --- NEW: Add tagging in background ---
            asyncio.create_task(tag_conversation(request.user_id, request.message, answer_content))

        return ChatResponse(answer=answer_content, schedule_details=None)

    except Exception as e:
        print(f"Chat API error: {e}")
        # Check for specific Google API errors (like rate limiting)
        if "429" in str(e) or "quota" in str(e).lower():
            raise HTTPException(status_code=429, detail=f"LLM API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user", response_model=User)
async def handle_create_or_update_user(user_data: dict):
    try:
        email = user_data.get("email")
        full_name = user_data.get("full_name")
        company_name = user_data.get("company_name")

        if not email:
            raise HTTPException(status_code=400, detail="Email is required.")
            
        user = await create_or_update_user(
            full_name=full_name or "Unknown User", # Provide a default if name is missing on update
            email=email,
            company_name=company_name
        )
        return user
    except Exception as e:
        print(f"Error in handle_create_or_update_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user", response_model=User)
async def handle_get_user(email: str):
    try:
        user = await get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        return user
    except Exception as e:
        print(f"Error in handle_get_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{email}")
async def handle_get_user_history(email: str):
    """
    Retrieves the full conversation history for a given user.
    """
    try:
        history = await history_module.get_user_history(email)
        return {"history": history}
    except Exception as e:
        print(f"Error fetching conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history.")


@app.delete("/api/user")
async def handle_delete_user(email: str):
    try:
        success = await crm_module.delete_user_by_email(email)
        if not success:
            raise HTTPException(status_code=404, detail=f"User with email '{email}' not found.")
        return {"message": f"User '{email}' deleted successfully."}
    except Exception as e:
        print(f"Error in handle_delete_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def tag_conversation(user_email: str, user_message: str, assistant_message: str):
    """
    Analyzes the conversation and applies relevant tags.
    """
    tagging_prompt = f"""
    Analyze the following exchange and categorize it with one or more relevant tags from this list: 
    ["Inquiry", "Scheduling", "Follow-up", "Complaint", "Resolved", "Pricing", "Amenities", "Location"].

    Respond with ONLY a raw JSON array of strings. Do not include any other text or markdown.
    
    Example:
    If the user asks "How much is the 2-bedroom?", your response should be:
    ["Inquiry", "Pricing"]

    If the user says "I want to see the apartment at 123 Main St tomorrow at 2pm", your response should be:
    ["Scheduling", "Inquiry"]

    User message: "{user_message}"
    Assistant response: "{assistant_message}"
    """
    try:
        chat_message = ChatMessage(role="user", content=tagging_prompt)
        response = await Settings.llm.achat([chat_message])
        response_content = response.message.content
        
        if not response_content:
            return

        tags = json.loads(response_content)
        if not isinstance(tags, list):
            return

        # Add tags to the latest conversation history entry for the user
        history_collection = await history_module.get_history_collection()
        
        # Find the most recent conversation to get its ID
        latest_conversation = await history_collection.find_one(
            {"user_email": user_email},
            sort=[("updated_at", -1)]
        )

        if latest_conversation:
            await history_collection.update_one(
                {"_id": latest_conversation["_id"]},
                {"$addToSet": {"tags": {"$each": tags}}},
            )
            print(f"Tagged conversation for {user_email} with: {tags}")

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Could not extract or apply tags: {e}")


def clean_json_from_response(response_text: str) -> str:
    """
    Extracts a JSON object from a string that might include markdown code blocks.
    """
    # Find JSON string within markdown code blocks ```json ... ```
    match = re.search(r"```(json)?\s*({.*})", response_text, re.DOTALL)
    if match:
        return match.group(2)
    # If no markdown, assume the whole string is a JSON object
    return response_text


async def extract_and_update_crm_details(user_email: str, message: str):
    """
    Extracts user details (name, company, preferences) from a message
    and updates the CRM in the background.
    """
    extraction_prompt = f"""
    Analyze the user's message to extract their full name, company name, and any specific leasing preferences mentioned (e.g., budget, desired size like '2 bedroom', location, specific amenities like 'gym' or 'large kitchen').

    The user's name will usually be preceded by a phrase like "my name is", "I'm", or "I am". Only extract a name if it's clearly stated. Do not guess a name from other words.

    Respond with ONLY a raw JSON object. Do not include any other text or markdown.
    The JSON object should contain one or more of the following keys if found: "full_name", "company_name", "preferences".
    The "preferences" key should itself be an object containing key-value pairs of the preferences.

    Example 1:
    If the user says "Hi, my name is John Doe and I'm looking for a 2-bedroom apartment in Brooklyn with a budget of $3000.", your response should be:
    {{
        "full_name": "John Doe",
        "preferences": {{
            "size": "2-bedroom",
            "location": "Brooklyn",
            "budget": "$3000"
        }}
    }}

    Example 2:
    If the user says "My name is Walter White, and I am looking for a property with a large kitchen", your response should be:
    {{
        "full_name": "Walter White",
        "preferences": {{
            "amenities": "large kitchen"
        }}
    }}

    If no details are found, return an empty JSON object: {{}}

    User Message: "{message}"
    """
    try:
        chat_message = ChatMessage(role="user", content=extraction_prompt)
        response = await Settings.llm.achat([chat_message])
        
        response_content = response.message.content
        if not response_content:
            return

        details = json.loads(response_content)
        if not details:
            return

        # Prepare details for the CRM update function
        full_name = details.get("full_name")
        company_name = details.get("company_name")
        preferences = details.get("preferences")

        # Call the update function if any details were found
        if full_name or company_name or preferences:
            await crm_module.create_or_update_user(
                email=user_email,
                full_name=full_name,
                company_name=company_name,
                preferences=preferences
            )
            print(f"Updated CRM for {user_email} with details: {details}")

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Could not extract or parse CRM details: {e}")


async def classify_query(user_message: str) -> dict:
    """
    Uses the LLM to classify the user's message to determine the correct response strategy.
    """
    prompt = f"""
    Classify the user's message into ONE of the following categories: "property_inquiry", "top_n_inquiry", "scheduling", or "general_chat".
    Respond with ONLY a single JSON object with a "type" key.

    - "property_inquiry": User is asking about property features, details, or qualitative descriptions (e.g., rent, size, address, amenities, "smaller kitchen").
    - "top_n_inquiry": User is asking for a ranked list using superlatives (e.g., "show me the cheapest", "what are the biggest?", "top 5").
    - "scheduling": User wants to book, schedule, view, see, or confirm an appointment or viewing. Also includes confirmation responses like "yes", "confirm", "book it".
    - "general_chat": User is asking about themselves, their preferences, or having general conversation.

    Examples:
    - "what is the price of 84 mulberry st?" -> {{"type": "property_inquiry"}}
    - "i want to book an appointment" -> {{"type": "scheduling"}}
    - "yes" (in context of scheduling) -> {{"type": "scheduling"}}
    - "confirm the appointment" -> {{"type": "scheduling"}}
    - "i would like properties with smaller kitchen" -> {{"type": "property_inquiry"}}
    - "show me the properties with the lowest rent" -> {{"type": "top_n_inquiry"}}
    - "can I see a property tomorrow?" -> {{"type": "scheduling"}}
    - "what are the 3 biggest spaces?" -> {{"type": "top_n_inquiry"}}
    - "can you book a viewing?" -> {{"type": "scheduling"}}
    - "what am i looking for?" -> {{"type": "general_chat"}}

    User Message: "{user_message}"
    """
    try:
        response = await Settings.llm.achat([ChatMessage(role="user", content=prompt)])
        response_content = response.message.content
        if not response_content:
            print("[Query-Classifier] Error: LLM returned an empty response.")
            return {"type": "property_inquiry"}
        
        data = json.loads(response_content)
        # --- NEW: Add a check to ensure the type is valid ---
        valid_types = {"property_inquiry", "top_n_inquiry", "scheduling", "general_chat"}
        if data.get("type") in valid_types:
            print(f"[Query-Classifier] Classified as: {data}")
            return data
        else:
            print(f"[Query-Classifier] Invalid type '{data.get('type')}', defaulting to property_inquiry")
            return {"type": "property_inquiry"}

    except (json.JSONDecodeError, AttributeError, KeyError):
        print("[Query-Classifier] Error, defaulting to property_inquiry")
        return {"type": "property_inquiry"}


async def find_top_n_properties(user_message: str) -> Optional[str]:
    """
    Scans all documents for a user, extracts the specified metric from the user's message,
    sorts them, and returns a formatted string of the top N properties.
    """
    # --- NEW: LLM call to extract parameters now happens inside this function ---
    extraction_prompt = f"""
    Analyze the user's message to extract parameters for a property search.
    Respond with ONLY a single JSON object.

    Extract:
    - "metric": The column name to sort by (e.g., "Monthly Rent", "Size (SF)").
    - "ascending": boolean, true for lowest/cheapest, false for highest/biggest.
    - "n": The number of items to list (default to 5 if not specified).

    Examples:
    - "show me the properties with the lowest rent" -> {{"metric": "Monthly Rent", "ascending": true, "n": 5}}
    - "what are the 3 biggest spaces?" -> {{"metric": "Size (SF)", "ascending": false, "n": 3}}
    - "top 10 cheapest" -> {{"metric": "Monthly Rent", "ascending": true, "n": 10}}

    User Message: "{user_message}"
    """
    try:
        response = await Settings.llm.achat([ChatMessage(role="user", content=extraction_prompt)])
        response_content = response.message.content or "{}"
        params = json.loads(response_content)
        
        # --- NEW: Check if a valid metric was extracted ---
        if "metric" not in params:
             return "I can create a list of properties for you, but I need to know how to rank them. For example, you can ask for the 'cheapest', 'most expensive', 'largest', or 'smallest' properties."

        metric = params.get("metric")
        top_n = params.get("n", 5)
        ascending = params.get("ascending", True)

    except (json.JSONDecodeError, AttributeError, KeyError):
        # Default to a helpful message if LLM fails or parsing is incorrect
        return "I can create a list of properties for you, but I need to know how to rank them. For example, you can ask for the 'cheapest', 'most expensive', 'largest', or 'smallest' properties."

    index = rag_module.rag_index
    if not index:
        return None

    # Get the document store from the index, which should always exist
    docstore = index.docstore
    if not docstore:
        print("[Error] find_top_n_properties: docstore not found on index.")
        return None  # Safeguard

    # This assumes the full document text is stored in the nodes, which it is.
    # We access .docs directly, which is specific to SimpleDocumentStore, using getattr for safety.
    all_nodes = getattr(docstore, "docs", None)

    # Add an explicit check to ensure all_nodes is a dictionary.
    if not isinstance(all_nodes, dict):
        print(f"[Warning] Could not retrieve document nodes; docstore might not be a SimpleDocumentStore or is empty.")
        return "Could not find any properties to analyze."

    properties = []
    # all_nodes is a Dict[str, BaseNode], so we iterate over its values
    for node in all_nodes.values():
        # The text is a "key: value, key: value" string. We can parse it.
        content = node.get_content()
        prop_dict = {}
        for part in content.split(', '):
            try:
                key, value = part.split(':', 1)
                prop_dict[key.strip()] = value.strip()
            except ValueError:
                continue
        properties.append(prop_dict)

    # Filter out properties that don't have the requested metric
    properties = [p for p in properties if metric in p]

    # Convert metric to a numeric type for sorting, handling errors
    for p in properties:
        try:
            # Remove characters like '$' or ',' for proper conversion
            p[metric] = float(re.sub(r'[^\d.]', '', p[metric]))
        except (ValueError, TypeError):
            p[metric] = float('-inf') if not ascending else float('inf') # Put invalid values at the end

    # Sort the properties
    sorted_properties = sorted(properties, key=lambda x: x.get(metric, 0), reverse=not ascending)

    # Get the top N
    top_properties = sorted_properties[:top_n]
    
    if not top_properties:
        return "Could not find any properties with that metric."

    # Format the output for the LLM to present
    output_lines = [f"Here are the top {top_n} properties based on {metric} ({'lowest' if ascending else 'highest'} first):"]
    for prop in top_properties:
        address = prop.get("Property Address", "N/A")
        value = prop.get(metric, "N/A")
        output_lines.append(f"- **{address}**: {metric} of ${value:,.2f}")

    return "\n".join(output_lines)


async def extract_schedule_details(user_message: str) -> dict:
    """
    Analyzes the user's message to extract scheduling details.
    This function assumes the message is already classified as a scheduling request.
    """
    details_prompt = f"""
    Analyze the user's message to extract scheduling details for a property viewing.
    The current date is {dt.date.today().isoformat()}.

    Look for:
    1. Property address (street address, building name, or property identifier)
    2. Desired date and time for the viewing
    3. Any confirmation words like "yes", "confirm", "book it", etc.

    Respond with ONLY a raw JSON object with these possible keys:
    - "address": Full property address if mentioned
    - "time": Desired time as ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS) if mentioned
    - "confirmation": true if the user is confirming a previously discussed appointment

    Examples:
    - "book 123 Main St for tomorrow at 2pm" -> {{"address": "123 Main St", "time": "2025-01-XX T14:00:00"}}
    - "yes" or "confirm" -> {{"confirmation": true}}
    - "I want to see a property" -> {{}}

    If no specific details are found, return an empty JSON object: {{}}

    User Message: "{user_message}"
    """
    try:
        details_response_raw = await Settings.llm.achat([ChatMessage(role="user", content=details_prompt)])
        details_response = details_response_raw.message.content
        if not details_response:
            print("[Scheduler-Details] The LLM returned an empty response for the details check.")
            return {}

        print(f"[Scheduler-Details] Raw LLM response: {details_response}")
        cleaned_response = clean_json_from_response(details_response)
        details_data = json.loads(cleaned_response)

        # Ensure we return a dictionary
        if isinstance(details_data, dict):
            return details_data
        else:
            print(f"[Scheduler-Details] LLM did not return a dict. Got: {type(details_data)}")
            return {}
        
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"[Scheduler-Details ERROR] Could not extract details. Raw Response: '{details_response or ''}'. Error: {e}")
        return {}


def run_user_indexing(user_id: str):
    """
    Looks for a user's document directory, finds all their files,
    and builds the RAG index from them.
    """
    global indexing_status
    try:
        user_doc_dir = os.path.join("user_documents", user_id)
        if not os.path.exists(user_doc_dir) or not os.listdir(user_doc_dir):
            indexing_status = {"status": "idle", "message": "No documents found for this user."}
            print(f"No documents found for user {user_id}. Index not built.")
            rag_module.clear_index() # Clear the index if the user has no docs
            return

        # The 'in_progress' status is now set in the main endpoint.
        
        file_paths = [os.path.join(user_doc_dir, f) for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
        
        if not file_paths:
            indexing_status = {"status": "idle", "message": "No CSV documents found for this user."}
            return

        rag_module.build_index_from_paths(file_paths)
        
        indexing_status = {"status": "success", "message": f"Documents for {user_id} loaded successfully."}
        print(f"Successfully built index for user {user_id} from {len(file_paths)} file(s).")
        
    except Exception as e:
        error_message = f"Failed to build index for {user_id}: {str(e)}"
        indexing_status = {"status": "error", "message": error_message}
        print(error_message)


@app.post("/api/documents/load")
async def load_user_documents(background_tasks: BackgroundTasks, user_id: str = Body(..., embed=True)):
    """
    Called on user login. Checks for existing documents and starts
    a background task to index them if found.
    IMMEDIATELY sets the status to in-progress to prevent race conditions.
    """
    global indexing_status
    indexing_status = {"status": "in_progress", "message": "Indexing started..."}
    background_tasks.add_task(run_user_indexing, user_id)
    return {"message": "Checking for existing documents. Indexing will start if documents are found."}


@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks, 
    user_id: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Accepts a file and a user_id, saves the file to a user-specific directory,
    and starts a background task to re-index ALL of that user's documents.
    """
    global indexing_status
    if indexing_status["status"] == "in_progress":
        raise HTTPException(status_code=409, detail="An indexing job is already in progress. Please wait.")

    try:
        user_doc_dir = os.path.join("user_documents", user_id)
        os.makedirs(user_doc_dir, exist_ok=True)
        
        # Ensure filename is not None before joining path
        filename = file.filename if file.filename else "default_uploaded_file.csv"
        file_path = os.path.join(user_doc_dir, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Start the background task to re-index all docs for the user
        background_tasks.add_task(run_user_indexing, user_id)
        
        return {"message": f"File '{file.filename}' uploaded successfully for user {user_id}. Re-indexing all documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.get("/api/documents/status")
async def get_indexing_status():
    """Returns the current status of the document indexing job."""
    return indexing_status


@app.post("/api/reset")
async def reset_application_data(user_id: Optional[str] = None):
    """
    Deletes data. If a user_id (email) is provided, only that user's
    conversations and CRM entry are deleted. Otherwise, all application
    data (RAG index, all users, all companies, all conversations) is wiped.
    """
    try:
        db = get_database()
        if user_id:
            # Per-user reset
            user = await get_user_by_email(user_id)
            if not user:
                raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

            # Delete conversations and user
            await db["conversation_history"].delete_many({"user_email": user_id})
            await db["users"].delete_one({"email": user_id})
            
            # Note: This does not delete the company, as it might be shared.
            return {"message": f"All conversation and CRM data for user '{user_id}' has been reset."}

        else:
            # Global reset
            rag_module.clear_index()
            await db["users"].delete_many({})
            await db["companies"].delete_many({})
            await db["conversation_history"].delete_many({})
            return {"message": "All application data has been reset."}

    except Exception as e:
        print(f"Error resetting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset application data: {str(e)}")

@app.post("/api/schedule")
async def schedule_event(request: ScheduleRequest):
    """
    Schedules a viewing event in Google Calendar and links it to the user.
    """
    try:
        event_url = calendar.schedule_viewing(
            user_email=request.email,
            property_address=request.address,
            time_str=request.time
        )
        
        # --- NEW: Link event to user in CRM ---
        db = get_database()
        user_collection = db["users"]
        await user_collection.update_one(
            {"email": request.email},
            {"$push": {"scheduled_events": event_url}}
        )
        print(f"Linked new event to user {request.email}")

        return {"message": "Event scheduled successfully and linked to user.", "event_url": event_url}
    except Exception as e:
        print(f"Error scheduling event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule event: {str(e)}")


# Serve frontend
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            # Try to serve the requested file
            return await super().get_response(path, scope)
        except HTTPException as ex:
            # If the file is not found, serve index.html
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex

# Mount the static directory at the root
app.mount("/", SPAStaticFiles(directory="static", html=True), name="static-files")
