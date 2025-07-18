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
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.schema import MetadataMode
import json
import re
from typing import Optional, Literal, List, Dict, Any
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
from app.user_context_validator import user_context_validator
from app.strict_response_generator import strict_response_generator
from app.error_handler import rag_error_handler, error_handling_context, ErrorContext, ErrorCategory, ErrorSeverity
from app.health_endpoints import health_router
from app.mongodb_health_endpoints import mongodb_health_router
import time
from pydantic import BaseModel, Field

# Add imports for Smart Property Recommendations
from app.intent_detection import intent_detection_service
from app.recommendation_workflow_manager import recommendation_workflow_manager
from app.recommendation_endpoints import recommendation_router
# Add imports for appointment booking at the top with other imports
from app.appointment_intent_detection import appointment_intent_detection_service
from app.appointment_workflow_manager import appointment_workflow_manager

# --- NEW: Global state for indexing status ---
indexing_status = {"status": "idle", "message": "No active indexing jobs."}

class PropertyRankingTool(BaseModel):
    """A structured tool for extracting property ranking parameters from a user query."""
    metric: Literal["monthly rent", "size (sf)"] = Field(..., description="The metric to sort properties by. Use 'monthly rent' for any cost-related queries and 'size (sf)' for any space-related queries.")
    order: Literal["ascending", "descending"] = Field(..., description="The sort order. Use 'ascending' for 'lowest', 'cheapest', 'smallest', 'low', or 'cheap'. Use 'descending' for 'highest', 'most expensive', 'biggest', or 'largest'.")
    n: int = Field(default=5, description="The number of properties to return. Defaults to 5 if not specified by the user.")

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    await connect_to_mongo()
    
    # Initialize ChromaDB client
    try:
        from app.chroma_client import chroma_manager
        await asyncio.to_thread(chroma_manager.get_client)
        print("ChromaDB client initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize ChromaDB client: {e}")
        print("Application will continue with fallback to in-memory storage")
    
    # Document indexing will now only happen on user action (e.g., upload)
    yield
    
    # On shutdown
    try:
        from app.chroma_client import chroma_manager
        chroma_manager.close_client()
        print("ChromaDB client closed")
    except Exception as e:
        print(f"Warning: Error closing ChromaDB client: {e}")
    
    close_mongo_connection()

app = FastAPI(title="Okada Leasing Agent API", lifespan=lifespan)

# Include health monitoring router
app.include_router(health_router)
app.include_router(mongodb_health_router)

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
    start_time = time.time()
    print(f"🔍 Chat request received - User: {request.user_id}, Message: '{request.message}'")
    
    async with error_handling_context("chat", user_id=request.user_id, query=request.message):
        try:
            # --- STEP 0: FAST MESSAGE CLASSIFICATION FOR PERFORMANCE OPTIMIZATION ---
            # Quick classification to handle simple messages without expensive operations
            from app.fast_message_classifier import FastMessageClassifier
            from app.conversational_response_handler import ConversationalResponseHandler
            from app.enhanced_intent_detection import EnhancedIntentDetectionService
            from app.models import MessageType, ProcessingStrategy
            
            fast_classifier = FastMessageClassifier()
            conversational_handler = ConversationalResponseHandler()
            enhanced_intent_service = EnhancedIntentDetectionService()
            
            # Get user context for personalization
            user_context = None
            if request.user_id:
                user_context = await conversational_handler.get_user_context(request.user_id)
            
            # Fast classification
            classification = fast_classifier.classify_message(request.message, user_context)
            
            print(f"⚡ Fast classification: {classification.message_type} (confidence: {classification.confidence:.2f}, strategy: {classification.processing_strategy})")
            
            # Handle quick responses for conversational messages
            if (classification.processing_strategy == ProcessingStrategy.QUICK_RESPONSE and 
                classification.confidence > 0.7):
                
                response_time = time.time() - start_time
                print(f"🚀 Quick response path taken ({response_time*1000:.1f}ms)")
                
                # Generate quick response
                response_message = conversational_handler.get_response_for_type(
                    classification.message_type, 
                    request.message, 
                    user_context
                )
                
                # Track performance
                try:
                    from app.performance_monitor import performance_monitor
                    performance_monitor.record_message_response(
                        message_type=classification.message_type.value,
                        duration_ms=(time.time() - start_time) * 1000,
                        success=True,
                        user_id=request.user_id,
                        additional_data={
                            "processing_strategy": classification.processing_strategy.value,
                            "confidence": classification.confidence,
                            "quick_response": True
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Performance monitoring error: {e}")
                
                # Save conversation history
                if request.user_id:
                    asyncio.create_task(history_module.add_message_to_history(
                        request.user_id, 
                        request.message, 
                        response_message
                    ))
                
                return ChatResponse(
                    answer=response_message,
                    schedule_details=None
                )
            
            # Handle direct property queries (bypass recommendation workflow)
            elif (classification.message_type == MessageType.DIRECT_PROPERTY_QUERY and 
                  classification.processing_strategy == ProcessingStrategy.DIRECT_SEARCH):
                
                print(f"🎯 Direct property query detected, using standard search functions")
                
                try:
                    # Use the existing query classification system
                    classification_result = await classify_query(request.message)
                    query_type = classification_result.get("type", "property_inquiry")
                    
                    print(f"Query classified as: {query_type}")
                    
                    # Route to appropriate search function
                    if query_type == "top_n_inquiry":
                        response_text = await find_top_n_properties(request.message, request.user_id)
                        if response_text:
                            # Save conversation history
                            if request.user_id:
                                asyncio.create_task(history_module.add_message_to_history(
                                    request.user_id, 
                                    request.message, 
                                    response_text
                                ))
                            
                            return ChatResponse(
                                answer=response_text,
                                schedule_details=None
                            )
                    
                    elif query_type == "entity_lookup":
                        response_text = await find_properties_by_metadata(request.message, request.user_id)
                        if response_text:
                            # Save conversation history
                            if request.user_id:
                                asyncio.create_task(history_module.add_message_to_history(
                                    request.user_id, 
                                    request.message, 
                                    response_text
                                ))
                            
                            return ChatResponse(
                                answer=response_text,
                                schedule_details=None
                            )
                    
                    # For other direct queries, continue with standard RAG search below
                    print(f"Direct query will use standard RAG search")
                    
                except Exception as e:
                    print(f"Error in direct query processing: {e}")
                    # Fall through to standard RAG search
            
            # For property searches, continue with enhanced intent detection
            elif classification.message_type == MessageType.PROPERTY_SEARCH:
                print(f"🏠 Property search detected, using enhanced intent detection")
                # Continue to existing property search logic below
                
            # For appointments, continue with appointment workflow
            elif classification.message_type == MessageType.APPOINTMENT_REQUEST:
                print(f"📅 Appointment request detected, using appointment workflow")
                # Continue to existing appointment logic below
                
            # For unknown/low confidence, use enhanced intent detection
            else:
                print(f"❓ Unclear intent or low confidence, using enhanced intent detection")
                enhanced_classification = await enhanced_intent_service.detect_intent_with_fallback(
                    request.message, request.user_id
                )
                
                # If enhanced detection gives us a quick response, use it
                if (enhanced_classification.processing_strategy == ProcessingStrategy.QUICK_RESPONSE and
                    enhanced_classification.confidence > 0.6):
                    
                    response_message = conversational_handler.get_response_for_type(
                        enhanced_classification.message_type, 
                        request.message, 
                        user_context
                    )
                    
                    # Track performance
                    try:
                        from app.performance_monitor import performance_monitor
                        performance_monitor.record_message_response(
                            message_type=enhanced_classification.message_type.value,
                            duration_ms=(time.time() - start_time) * 1000,
                            success=True,
                            user_id=request.user_id,
                            additional_data={
                                "processing_strategy": enhanced_classification.processing_strategy.value,
                                "confidence": enhanced_classification.confidence,
                                "enhanced_detection": True
                            }
                        )
                    except Exception as e:
                        print(f"⚠️ Performance monitoring error: {e}")
                    
                    # Save conversation history
                    if request.user_id:
                        asyncio.create_task(history_module.add_message_to_history(
                            request.user_id, 
                            request.message, 
                            response_message
                        ))
                    
                    return ChatResponse(
                        answer=response_message,
                        schedule_details=None
                    )

            # --- NEW: STEP 0A: APPOINTMENT BOOKING SESSION CHECK ---
            # First check if user has an existing appointment booking session
            existing_session_processed = False
            session_expiry_minutes = 30
            if request.user_id:
                try:
                    from app.database import get_database
                    import datetime
                    db = get_database()
                    appointment_collection = db["appointment_sessions"]
                    now = datetime.datetime.now()

                    # Check for existing appointment sessions
                    existing_session = await appointment_collection.find_one({
                        "user_id": request.user_id,
                        "status": {"$in": ["collecting_info", "confirming"]}
                    })

                    session_is_expired = False
                    is_new_appointment_intent = False
                    if existing_session:
                        # Check if session is expired
                        updated_at = existing_session.get("updated_at")
                        if updated_at:
                            if isinstance(updated_at, str):
                                updated_at = datetime.datetime.fromisoformat(updated_at)
                            session_is_expired = (now - updated_at).total_seconds() > session_expiry_minutes * 60
                        # Check if this message is a new appointment intent
                        try:
                            appointment_intent = await appointment_intent_detection_service.detect_appointment_intent(request.message)
                            is_new_appointment_intent = appointment_intent.is_appointment_request and appointment_intent.confidence > 0.6
                        except Exception:
                            is_new_appointment_intent = False
                        # If expired or new intent, cancel old session
                        if session_is_expired or is_new_appointment_intent:
                            print(f"🗑️ Expiring/cancelling old appointment session {existing_session['_id']} for user {request.user_id}")
                            try:
                                await appointment_workflow_manager.cancel_appointment(existing_session["_id"])
                            except Exception as e:
                                print(f"⚠️ Error cancelling old session: {e}")
                            existing_session = None
                    if existing_session:
                        session_id = existing_session["_id"]
                        print(f"📅 Found existing appointment session {session_id} for user {request.user_id}")
                        # Process this message as part of the existing workflow
                        workflow_response = await appointment_workflow_manager.process_user_response(
                            session_id,
                            request.message
                        )
                        if workflow_response.success:
                            print(f"✅ Processed appointment response: {workflow_response.step_name}")
                            existing_session_processed = True
                            # Save the conversation
                            if request.user_id:
                                asyncio.create_task(history_module.add_message_to_history(
                                    request.user_id, 
                                    request.message, 
                                    workflow_response.message
                                ))
                            # --- PATCH: Return appointment_details for frontend UI ---
                            appointment_details = None
                            ad = getattr(workflow_response, 'appointment_data', None)
                            if ad is not None:
                                appointment_details = {
                                    'title': ad.title,
                                    'location': ad.location,
                                    'datetime': ad.date.strftime('%A, %B %d, %Y at %I:%M %p'),
                                    'duration': ad.duration_minutes,
                                    'attendees': ', '.join(ad.attendee_emails) if ad.attendee_emails else '',
                                    'description': ad.description or ''
                                }
                            if appointment_details:
                                return {
                                    'answer': workflow_response.message,
                                    'schedule_details': None,
                                    'appointment_details': appointment_details
                                }
                            else:
                                return ChatResponse(
                                    answer=workflow_response.message,
                                    schedule_details=None
                                )
                        else:
                            print(f"⚠️ Failed to process appointment response: {workflow_response.message}")
                            existing_session_processed = True
                            # Still return the response to avoid creating a new session
                            return ChatResponse(
                                answer=workflow_response.message,
                                schedule_details=None
                            )
                except Exception as e:
                    print(f"❌ Error checking existing appointment sessions: {e}")
                    # Fall through to new appointment detection

            # --- NEW: STEP 0B: APPOINTMENT BOOKING DETECTION ---
            # Check if this is an appointment booking request before other intent detection
            # Only run this if we haven't already processed an existing session
            if request.user_id and not existing_session_processed:
                try:
                    print(f"🗓️ Checking for appointment booking intent in message: '{request.message}'")
                    appointment_intent = await appointment_intent_detection_service.detect_appointment_intent(request.message)
                    
                    if appointment_intent.is_appointment_request and appointment_intent.confidence > 0.6:
                        print(f"📅 Appointment booking intent detected (confidence: {appointment_intent.confidence:.2f})")
                        
                        # Start appointment booking workflow
                        workflow_response = await appointment_workflow_manager.start_appointment_booking(
                            request.user_id,
                            request.message
                        )
                        
                        if workflow_response.success:
                            print(f"✅ Appointment workflow started: {workflow_response.step_name}")
                            
                            # Save the conversation
                            if request.user_id:
                                asyncio.create_task(history_module.add_message_to_history(
                                    request.user_id, 
                                    request.message, 
                                    workflow_response.message
                                ))
                            # --- PATCH: Return appointment_details for frontend UI ---
                            appointment_details = None
                            ad = getattr(workflow_response, 'appointment_data', None)
                            if ad is not None:
                                appointment_details = {
                                    'title': ad.title,
                                    'location': ad.location,
                                    'datetime': ad.date.strftime('%A, %B %d, %Y at %I:%M %p'),
                                    'duration': ad.duration_minutes,
                                    'attendees': ', '.join(ad.attendee_emails) if ad.attendee_emails else '',
                                    'description': ad.description or ''
                                }
                            if appointment_details:
                                return {
                                    'answer': workflow_response.message,
                                    'schedule_details': None,
                                    'appointment_details': appointment_details
                                }
                            else:
                                return ChatResponse(
                                    answer=workflow_response.message,
                                    schedule_details=None
                                )
                        else:
                            print("⚠️ Appointment workflow failed to start properly, falling back to standard chat")
                    else:
                        print(f"ℹ️ No appointment booking intent detected (confidence: {appointment_intent.confidence:.2f})")
                        
                except Exception as e:
                    print(f"❌ Error in appointment booking detection: {e}")
                    # Fall through to standard chat flow

            # --- STEP 0C: SMART PROPERTY RECOMMENDATIONS DETECTION ---
            # Check if this is a recommendation request before proceeding with standard RAG
            if request.user_id:
                try:
                    print(f"🤖 Checking for recommendation intent in message: '{request.message}'")
                    recommendation_intent = await intent_detection_service.detect_recommendation_intent(request.message)
                    
                    if recommendation_intent.is_recommendation_request and recommendation_intent.confidence > 0.6:
                        print(f"✨ Recommendation intent detected (confidence: {recommendation_intent.confidence:.2f})")
                        
                        # Start recommendation workflow
                        workflow_session = await recommendation_workflow_manager.start_recommendation_workflow(
                            request.user_id,
                            request.message
                        )
                        
                        # Get the next step in the workflow
                        next_step = await recommendation_workflow_manager.get_next_step(workflow_session.session_id)
                        
                        if next_step and next_step.success:
                            print(f"🔄 Recommendation workflow step: {next_step.step_name}")
                            
                            # Save the conversation
                            if request.user_id:
                                asyncio.create_task(history_module.add_message_to_history(
                                    request.user_id, 
                                    request.message, 
                                    next_step.response_message
                                ))
                            
                            return ChatResponse(
                                answer=next_step.response_message,
                                schedule_details=None
                            )
                        else:
                            print("⚠️ Recommendation workflow failed to start properly, falling back to standard chat")
                    else:
                        print(f"ℹ️ No recommendation intent detected (confidence: {recommendation_intent.confidence:.2f})")
                        
                except Exception as e:
                    print(f"❌ Error in recommendation workflow detection: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall through to standard chat flow

            # --- STEP 1: ENHANCED INDEX HEALTH VALIDATION AND ASYNC MANAGEMENT ---
            # Check if user has a healthy index with async rebuilding support
            user_index = None
            if request.user_id and classification.requires_index:
                try:
                    from app.index_health_validator import IndexHealthValidator
                    from app.async_index_manager import AsyncIndexManager
                    
                    health_validator = IndexHealthValidator()
                    async_index_manager = AsyncIndexManager()
                    
                    print(f"🔍 Performing health check for user: {request.user_id}")
                    
                    # Fast health check (with caching)
                    health_result = await health_validator.validate_user_index_health(request.user_id)
                    
                    if health_result.is_healthy:
                        print(f"✅ Index is healthy for user {request.user_id}")
                        user_index = await rag_module.get_user_index(request.user_id)
                        
                    elif async_index_manager.is_user_rebuilding(request.user_id):
                        print(f"🔄 Index rebuild in progress for user {request.user_id}")
                        
                        # Provide fallback response while rebuilding
                        fallback_response = async_index_manager.get_fallback_response(request.user_id)
                        
                        # Track performance
                        try:
                            from app.performance_monitor import performance_monitor
                            performance_monitor.record_message_response(
                                message_type="property_search_rebuilding",
                                duration_ms=(time.time() - start_time) * 1000,
                                success=True,
                                user_id=request.user_id,
                                additional_data={
                                    "fallback_reason": "index_rebuilding",
                                    "health_issues": len(health_result.issues_found)
                                }
                            )
                        except Exception as e:
                            print(f"⚠️ Performance monitoring error: {e}")
                        
                        # Save conversation history
                        if request.user_id:
                            asyncio.create_task(history_module.add_message_to_history(
                                request.user_id, 
                                request.message, 
                                fallback_response
                            ))
                        
                        return ChatResponse(
                            answer=fallback_response,
                            schedule_details=None
                        )
                        
                    else:
                        print(f"⚠️ Index unhealthy for user {request.user_id}: {health_result.issues_found}")
                        
                        # Try to start async rebuild if we have documents
                        user_doc_dir = os.path.join("user_documents", request.user_id)
                        if os.path.exists(user_doc_dir):
                            csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
                            if csv_files:
                                print(f"🔄 Starting async rebuild for {request.user_id} with {len(csv_files)} files")
                                
                                # Start async rebuild
                                rebuild_op = await async_index_manager.rebuild_user_index_async(
                                    request.user_id, 
                                    priority="high"  # High priority for immediate user requests
                                )
                                
                                # Provide immediate fallback response
                                fallback_response = async_index_manager.get_fallback_response(request.user_id)
                                
                                # Track performance
                                try:
                                    from app.performance_monitor import performance_monitor
                                    performance_monitor.record_message_response(
                                        message_type="property_search_rebuild_started",
                                        duration_ms=(time.time() - start_time) * 1000,
                                        success=True,
                                        user_id=request.user_id,
                                        additional_data={
                                            "fallback_reason": "index_rebuild_started",
                                            "files_found": len(csv_files),
                                            "health_issues": len(health_result.issues_found)
                                        }
                                    )
                                except Exception as e:
                                    print(f"⚠️ Performance monitoring error: {e}")
                                
                                # Save conversation history
                                if request.user_id:
                                    asyncio.create_task(history_module.add_message_to_history(
                                        request.user_id, 
                                        request.message, 
                                        fallback_response
                                    ))
                                
                                return ChatResponse(
                                    answer=fallback_response,
                                    schedule_details=None
                                )
                            else:
                                print(f"❌ No documents found for user {request.user_id}")
                                # Continue without index - will trigger general chat
                        else:
                            print(f"❌ No document directory for user {request.user_id}")
                            # Continue without index - will trigger general chat
                            
                except Exception as e:
                    print(f"❌ Error in index health validation: {e}")
                    # Continue without index for general chat
                    
            elif request.user_id and not classification.requires_index:
                print(f"ℹ️ Request doesn't require index - skipping validation")

            # Legacy index checking for backward compatibility
            # This will be reached if health validation was skipped or failed
            if not user_index and classification.requires_index and request.user_id:
                # Try to build index if user has documents but no index
                if request.user_id:
                    user_doc_dir = os.path.join("user_documents", request.user_id)
                    if os.path.exists(user_doc_dir):
                        csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
                        if csv_files:
                            print(f"Found {len(csv_files)} documents for {request.user_id}, attempting to build index...")
                            file_paths = [os.path.join(user_doc_dir, f) for f in csv_files]
                            try:
                                # Build the index
                                user_index = await rag_module.build_user_index(request.user_id, file_paths)
                                
                                # Validate the index was created successfully
                                if user_index:
                                    # Verify we can create a retriever
                                    test_retriever = rag_module.get_fusion_retriever(request.user_id)
                                    if test_retriever:
                                        print(f"✅ Successfully built and validated index for {request.user_id}")
                                    else:
                                        print(f"⚠️ Index built but retriever creation failed for {request.user_id}")
                                        user_index = None
                                else:
                                    print(f"❌ Index building returned None for {request.user_id}")
                                    
                            except Exception as e:
                                print(f"❌ Failed to build index for {request.user_id}: {e}")
                                user_index = None
                        else:
                            print(f"No CSV files found in {user_doc_dir}")
                    else:
                        print(f"User document directory not found: {user_doc_dir}")
                
                if not user_index:
                    raise HTTPException(status_code=503, detail="RAG index is not ready. Please upload documents first or try again later.")

            # --- STEP 1: LOAD USER CONTEXT ---
            # Always start by getting the user's profile and saving their latest message.
            user_profile = None
            if request.user_id:
                # Run CRM extraction in the background to not delay the response
                asyncio.create_task(extract_and_update_crm_details(request.user_id, request.message))
                user_profile = await crm_module.get_user_by_email(request.user_id)

            # --- STEP 2: BUILD A SMARTER SYSTEM PROMPT ---
            # This prompt now includes the user's name and known preferences.
            system_prompt_parts = [
                "You are Okada IntelliAgent, a professional, highly intelligent, and proactive leasing agent assistant for Okada & Company.",
                "Your tone is always helpful, clear, and concise. You are a partner in finding the perfect property.",
                "Always format your responses for maximum readability using markdown.",
            ]
            if user_profile:
                system_prompt_parts.append(f"You are speaking with {user_profile.full_name}. Their email is {user_profile.email}.")
            
            system_prompt = "\n".join(system_prompt_parts)
            chat_history = [ChatMessage(role="system", content=system_prompt)]
            
            # --- STEP 3: GENERATE A CONTEXT-AWARE RAG QUERY ---
            # For property addresses, use the original message directly to avoid losing specificity
            # Only generate alternative queries for complex requests
            if any(keyword in request.message.lower() for keyword in ['tell me about', 'what is', 'show me', 'find']):
                query_gen_prompt = f"""
                Extract the key search terms from this real estate query. Focus on:
                - Exact property addresses (keep them exactly as stated)
                - Property features (size, rent, amenities)
                - Location names (streets, neighborhoods)

                User's Message: "{request.message}"

                Return only the key search terms, separated by spaces:
                """
                query_gen_response = await Settings.llm.achat([ChatMessage(role="user", content=query_gen_prompt)])
                search_query = query_gen_response.message.content or request.message
            else:
                # For direct address queries, use the original message
                search_query = request.message
            
            print(f"Original Query: {request.message}")
            print(f"Search Query: {search_query}") # For debugging

            # --- STEP 4: PERFORM THE RAG SEARCH WITH MULTI-STRATEGY APPROACH ---
            retrieved_nodes = []
            search_details: Dict[str, Any] = {}
            
            if user_index:
                try:
                    # Use optimized multi-strategy search for better results
                    print(f"Starting optimized multi-strategy search for: '{request.message}'")
                    multi_search_result = await rag_module.retrieve_context_optimized(request.message, request.user_id)
                    
                    if multi_search_result.nodes_found:
                        retrieved_nodes = multi_search_result.nodes_found
                        print(f"Multi-strategy search found {len(retrieved_nodes)} nodes")
                        
                        # Log search strategy details
                        if multi_search_result.best_result:
                            print(f"Best strategy: '{multi_search_result.best_result.strategy}' "
                                  f"with query: '{multi_search_result.best_result.query_used}'")
                        
                        search_details = {
                            "strategies_tried": len(multi_search_result.all_results),
                            "best_strategy": multi_search_result.best_result.strategy if multi_search_result.best_result else None,
                            "total_time_ms": multi_search_result.total_execution_time_ms,
                            "nodes_found": len(retrieved_nodes)
                        }
                    else:
                        print("Multi-strategy search found no results")
                        search_details = {
                            "strategies_tried": len(multi_search_result.all_results),
                            "best_strategy": None,
                            "total_time_ms": multi_search_result.total_execution_time_ms,
                            "nodes_found": 0
                        }
                        
                except Exception as e:
                    print(f"Error during multi-strategy search: {e}")
                    # Fallback to original search method
                    try:
                        fusion_retriever = rag_module.get_fusion_retriever(request.user_id)
                        if fusion_retriever:
                            print("Falling back to original search method")
                            retrieved_nodes = await fusion_retriever.aretrieve(request.message)
                            print(f"Fallback search found {len(retrieved_nodes)} nodes")
                            search_details = {"fallback_used": True, "nodes_found": len(retrieved_nodes)}
                        else:
                            print("No fusion retriever available")
                            search_details = {"error": "No retriever available"}
                    except Exception as fallback_error:
                        print(f"Fallback search also failed: {fallback_error}")
                        search_details = {"error": f"All search methods failed: {e}, {fallback_error}"}

            # --- STEP 5: GENERATE THE FINAL RESPONSE WITH STRICT VALIDATION ---
            # Use the new strict response generator that prevents hallucination
            print(f"🎯 Generating strict response with {len(retrieved_nodes)} retrieved nodes")
            
            try:
                strict_result = await strict_response_generator.generate_strict_response(
                    user_query=request.message,
                    retrieved_nodes=retrieved_nodes,
                    user_id=request.user_id
                )
                
                # Log the strict response generation results
                print(f"✅ Strict response generation completed:")
                print(f"   - Generation successful: {strict_result.generation_successful}")
                print(f"   - Context valid: {strict_result.context_validation.is_valid}")
                print(f"   - Quality valid: {strict_result.quality_validation.is_valid}")
                print(f"   - Fallback used: {strict_result.fallback_used}")
                print(f"   - Properties found: {strict_result.context_validation.property_count}")
                
                if strict_result.context_validation.validation_issues:
                    print(f"   - Context issues: {strict_result.context_validation.validation_issues}")
                
                if strict_result.quality_validation.quality_issues:
                    print(f"   - Quality issues: {strict_result.quality_validation.quality_issues}")
                
                response_text = strict_result.response_text
                
                # Add search details to the response metadata for debugging
                search_details["strict_generation"] = {
                    "context_confidence": strict_result.context_validation.confidence_score,
                    "quality_confidence": strict_result.quality_validation.confidence_score,
                    "generation_successful": strict_result.generation_successful,
                    "fallback_used": strict_result.fallback_used,
                    "properties_found": strict_result.context_validation.property_count
                }
                
            except Exception as e:
                print(f"❌ Error in strict response generation: {e}")
                # Fallback to a safe error message
                response_text = """I apologize, but I'm having trouble processing your request right now. 

Please try rephrasing your question or ask about a different property. I'm here to help you find the information you need from our available listings."""
            
            # --- STEP 6: SAVE AND RETURN ---
            if request.user_id and response_text:
                asyncio.create_task(history_module.add_message_to_history(request.user_id, request.message, response_text))

            return ChatResponse(answer=response_text or "", schedule_details=None)

        except HTTPException:
            raise
        except Exception as e:
            print(f"An unexpected chat API error occurred: {e}")
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

@app.get("/api/documents/list/{user_id}")
async def list_user_documents(user_id: str):
    """
    Lists all the .csv files found in a specific user's document directory.
    """
    try:
        user_doc_dir = os.path.join("user_documents", user_id)
        if not os.path.exists(user_doc_dir):
            return {"documents": []}
        
        files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/api/user", response_model=User)
async def handle_get_user(email: str):
    try:
        # First, try to fetch the user from the database
        user = await crm_module.get_user_by_email(email)
    except Exception as e:
        # If there's a database connection error, it's a 500
        print(f"Database error in handle_get_user: {e}")
        raise HTTPException(status_code=500, detail="A database error occurred.")

    # If the database call was successful but no user was returned...
    if not user:
        # ...then it's a 404 Not Found, which is what the frontend expects.
        raise HTTPException(status_code=404, detail="User not found.")

    # If a user was found, return their data
    return user


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


# --- USER CONTEXT DEBUG ENDPOINTS ---

@app.get("/api/debug/user-context/{user_id}")
async def debug_user_context(user_id: str):
    """
    Get comprehensive debug information for a user's context and document association.
    
    This endpoint validates:
    - User ID handling and path generation
    - Document discovery and loading
    - ChromaDB collection status
    - Index availability and retriever functionality
    """
    try:
        debug_info = await user_context_validator.get_comprehensive_debug_info(user_id)
        
        # Convert dataclasses to dictionaries for JSON serialization
        return {
            "user_id": debug_info.user_id,
            "validation_result": {
                "is_valid": debug_info.validation_result.is_valid,
                "issues": debug_info.validation_result.issues,
                "details": debug_info.validation_result.details,
                "timestamp": debug_info.validation_result.timestamp.isoformat()
            },
            "document_result": {
                "documents_found": debug_info.document_result.documents_found,
                "documents_loaded": debug_info.document_result.documents_loaded,
                "collection_name": debug_info.document_result.collection_name,
                "collection_exists": debug_info.document_result.collection_exists,
                "index_exists": debug_info.document_result.index_exists,
                "retriever_available": debug_info.document_result.retriever_available,
                "issues": debug_info.document_result.issues
            },
            "collection_stats": debug_info.collection_stats,
            "index_stats": debug_info.index_stats,
            "retriever_stats": debug_info.retriever_stats
        }
    except Exception as e:
        print(f"Error in debug_user_context: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


@app.post("/api/debug/fix-user-context/{user_id}")
async def fix_user_context(user_id: str):
    """
    Attempt to automatically fix common user context issues.
    
    This endpoint will:
    - Rebuild indexes if documents exist but index is missing
    - Recreate collections if they're corrupted
    - Validate the fixes and report results
    """
    try:
        fix_report = await user_context_validator.fix_user_context_issues(user_id)
        return fix_report
    except Exception as e:
        print(f"Error in fix_user_context: {e}")
        raise HTTPException(status_code=500, detail=f"Fix failed: {str(e)}")


@app.get("/api/debug/validate-user/{user_id}")
async def validate_user_context_only(user_id: str):
    """
    Validate only the user context handling (paths, collection names, etc.).
    Lighter weight than the full debug endpoint.
    """
    try:
        validation_result = await user_context_validator.validate_user_context(user_id)
        return {
            "user_id": validation_result.user_id,
            "is_valid": validation_result.is_valid,
            "issues": validation_result.issues,
            "details": validation_result.details,
            "timestamp": validation_result.timestamp.isoformat()
        }
    except Exception as e:
        print(f"Error in validate_user_context_only: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.get("/api/debug/test-retriever/{user_id}")
async def test_user_retriever(user_id: str, query: str = "test query"):
    """
    Test the user's retriever functionality with a sample query.
    
    This endpoint tests:
    - Fusion retriever creation
    - Search functionality
    - Result formatting
    """
    try:
        # Test retriever creation
        fusion_retriever = rag_module.get_fusion_retriever(user_id)
        if not fusion_retriever:
            return {
                "user_id": user_id,
                "retriever_available": False,
                "error": "No fusion retriever available",
                "suggestion": "Try rebuilding the user index"
            }
        
        # Test search functionality
        try:
            retrieved_nodes = await fusion_retriever.aretrieve(query)
            
            # Format results
            results = []
            for i, node in enumerate(retrieved_nodes):
                score_value = getattr(node, 'score', None)
                results.append({
                    "node_index": i,
                    "score": float(score_value) if score_value is not None else None,
                    "content_preview": node.get_content()[:200] + "..." if len(node.get_content()) > 200 else node.get_content(),
                    "metadata_keys": list(node.metadata.keys()) if hasattr(node, 'metadata') else []
                })
            
            return {
                "user_id": user_id,
                "retriever_available": True,
                "query": query,
                "results_count": len(retrieved_nodes),
                "results": results,
                "success": True
            }
            
        except Exception as search_error:
            return {
                "user_id": user_id,
                "retriever_available": True,
                "query": query,
                "search_error": str(search_error),
                "success": False
            }
            
    except Exception as e:
        print(f"Error in test_user_retriever: {e}")
        raise HTTPException(status_code=500, detail=f"Retriever test failed: {str(e)}")


@app.get("/api/debug/collection-info/{user_id}")
async def get_collection_info(user_id: str):
    """
    Get detailed information about the user's ChromaDB collection.
    """
    try:
        collection_stats = await user_context_validator.get_collection_stats(user_id)
        return {
            "user_id": user_id,
            "collection_stats": collection_stats
        }
    except Exception as e:
        print(f"Error in get_collection_info: {e}")
        raise HTTPException(status_code=500, detail=f"Collection info failed: {str(e)}")


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
    Extracts user details (name, company) from a message
    and updates the CRM in the background. Preference extraction has been removed
    to prevent preference leakage between sessions.
    """
    extraction_prompt = f"""
    Analyze the user's message to extract their full name and company name.

    The user's name will usually be preceded by a phrase like "my name is", "I'm", or "I am". Only extract a name if it's clearly stated. Do not guess a name from other words.

    Respond with ONLY a raw JSON object. Do not include any other text or markdown.
    The JSON object should contain one or more of the following keys if found: "full_name", "company_name".

    Example 1:
    If the user says "Hi, my name is John Doe and I work at Acme Inc.", your response should be:
    {{
        "full_name": "John Doe",
        "company_name": "Acme Inc."
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

        # Call the update function if any details were found
        if full_name or company_name:
            await crm_module.create_or_update_user(
                email=user_email,
                full_name=full_name,
                company_name=company_name,
                preferences=None # Explicitly set preferences to None
            )
            print(f"Updated CRM for {user_email} with details: {details}")

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Could not extract or parse CRM details: {e}")


async def classify_query(user_message: str) -> dict:
    """
    Uses the LLM to classify the user's message into a specific category.
    This is the first step in routing the user's request.
    """
    # NEW: Added more examples and a clearer "general_chat" category.
    prompt = f"""
    Classify the user's message into ONE of the following categories: "property_inquiry", "top_n_inquiry", "scheduling", "entity_lookup", or "general_chat".
    Respond with ONLY a single JSON object with a "type" key.

    - "property_inquiry": Asking for details about a specific property or type of property.
    - "top_n_inquiry": Asking for a ranked list using superlatives (e.g., "cheapest", "biggest", "top 5").
    - "scheduling": Wants to book, schedule, or view an appointment.
    - "entity_lookup": Asking for properties related to a specific named person or company.
    - "general_chat": A conversational greeting, closing, statement of opinion, or question not related to real estate.

    Examples:
    - "tell me about 22 W 32nd St" -> {{"type": "property_inquiry"}}
    - "show me the properties with the lowest rent" -> {{"type": "top_n_inquiry"}}
    - "i want to book an appointment" -> {{"type": "scheduling"}}
    - "what is associated with Jack Sparrow?" -> {{"type": "entity_lookup"}}
    - "how are you" -> {{"type": "general_chat"}}
    - "i like ice cream" -> {{"type": "general_chat"}}
    - "what are you talking about" -> {{"type": "general_chat"}}
    - "thank you" -> {{"type": "general_chat"}}

    User Message: "{user_message}"
    """
    try:
        response = await Settings.llm.achat([ChatMessage(role="user", content=prompt)])
        data = json.loads(response.message.content or '{}')
        valid_types = {"property_inquiry", "top_n_inquiry", "scheduling", "entity_lookup", "general_chat"}
        if data.get("type") in valid_types:
            return data
        else:
            # If classification is unclear, default to a property search.
            return {"type": "property_inquiry"}
    except Exception:
        return {"type": "property_inquiry"}


async def find_properties_by_metadata(user_message: str, user_id: str) -> Optional[str]:
    """
    Extracts an entity and category from the user message and performs a filtered
    search on the user's ChromaDB index metadata.
    """
    # Step 1: Extract the entity and a potential key
    extraction_prompt = f"""
    From the user's message, extract the person's name or company name ("value") and the category they belong to ("key").
    The "key" should be a potential metadata field like "property associate" or "tenant".
    Respond with ONLY a single JSON object.

    Examples:
    - "show me listings for Jack Sparrow" -> {{"key": "property associate", "value": "Jack Sparrow"}}
    - "what properties are associated with davy jones?" -> {{"key": "property associate", "value": "davy jones"}}
    - "which buildings does stark industries own" -> {{"key": "owner", "value": "stark industries"}}

    If you can't find a clear person/company name, return an empty JSON object: {{}}

    User Message: "{user_message}"
    """
    try:
        response = await Settings.llm.achat([ChatMessage(role="user", content=extraction_prompt)])
        params = json.loads(response.message.content or "{}")
        
        if "key" not in params or "value" not in params:
            return None # Fallback to general RAG search

        key = params["key"]
        value = params["value"]
    except Exception:
        return None

    # Step 2: Perform the filtered search using user's ChromaDB index
    index = await rag_module.get_user_index(user_id)
    if not index:
        return "The property index is not available."

    filters = MetadataFilters(filters=[ExactMatchFilter(key=key, value=value)])
    retriever = index.as_retriever(similarity_top_k=10, filters=filters)
    nodes = retriever.retrieve(value) # type: ignore

    if not nodes:
        return f"I couldn't find any properties where the '{key}' is listed as '{value}'. I can perform a general search for '{value}' instead if you'd like."

    # Step 3: Format the response
    output_lines = [f"I found {len(nodes)} properties where the {key} is '{value}':"]
    for node in nodes: # type: ignore
        # Assuming metadata contains 'property address'
        address = node.metadata.get("property address", "N/A")
        output_lines.append(f"- **{address}**")
        
    return "\n".join(output_lines)

async def find_top_n_properties(user_message: str, user_id: str) -> Optional[str]:
    """
    Uses a structured output approach to extract ranking parameters from the user's message,
    sorts properties from the user's ChromaDB index, and returns a formatted string.
    """
    try:
        # Use the LLM with the Pydantic tool to force structured output
        llm_with_tool = Settings.llm.with_structured_output(PropertyRankingTool) # type: ignore
        
        # Create a prompt that instructs the LLM to use the tool
        prompt = f"Given the user's request, extract the parameters for ranking properties using the PropertyRankingTool.\n\nUser Request: \"{user_message}\""
        
        params = await llm_with_tool.ainvoke(prompt)

        metric = params.metric
        # Convert Pydantic order to the boolean required by the sort function
        ascending = (params.order == "ascending")
        top_n = params.n

    except Exception as e:
        # This will catch errors if the LLM fails to populate the tool
        print(f"Error extracting ranking parameters: {e}")
        return "I can create a list of properties for you, but I need to know how to rank them. For example, you can ask for the 'cheapest', 'most expensive', 'largest', or 'smallest' properties."

    # --- The rest of the logic for retrieving and sorting from user's ChromaDB index ---
    index = await rag_module.get_user_index(user_id)
    if not index:
        return None

    docstore = index.docstore
    all_nodes = getattr(docstore, "docs", None)

    if not isinstance(all_nodes, dict):
        return "Could not find any properties to analyze."

    properties = []
    for node in all_nodes.values():
        content = node.get_content()
        prop_dict = {}
        for part in content.split(', '):
            try:
                key, value = part.split(':', 1)
                prop_dict[key.strip().lower()] = value.strip() # Standardize keys to lowercase
            except ValueError:
                continue
        properties.append(prop_dict)

    properties = [p for p in properties if metric in p]

    for p in properties:
        try:
            p[metric] = float(re.sub(r'[^\d.]', '', p[metric]))
        except (ValueError, TypeError):
            p[metric] = float('-inf') if not ascending else float('inf')

    sorted_properties = sorted(properties, key=lambda x: x.get(metric, 0), reverse=not ascending)
    top_properties = sorted_properties[:top_n]
    
    if not top_properties:
        return f"I couldn't find any properties with a metric for '{metric}'."

    output_lines = [f"Here are the top {top_n} properties based on {'lowest' if ascending else 'highest'} {metric}:"]
    for prop in top_properties:
        address = prop.get("property address", "N/A")
        value = prop.get(metric, "N/A")
        # Format currency for rent, and add units for size
        formatted_value = f"${value:,.0f}" if metric == "monthly rent" else f"{int(value):,} SF"
        output_lines.append(f"- **{address}**: {formatted_value}")

    return "\n".join(output_lines)


async def extract_schedule_details(user_message: str) -> dict:
    """
    Analyzes the user's message to extract scheduling details.
    This function assumes the message is already classified as a scheduling request.
    """
    details_prompt = f"""
    Analyze the user's message to extract the property address and the desired time for a viewing.
    The current date is {dt.date.today().isoformat()}.

    Respond with ONLY a raw JSON object.
    - If you find the address and time, use the keys "address" and "time".
    - "address" must be the full property address mentioned.
    - "time" must be the desired time converted into a full ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS).

    If you cannot find BOTH an address and a time, return an empty JSON object: {{}}

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


async def async_run_user_indexing(user_id: str, filenames: Optional[List[str]] = None):
    """
    Async version of user indexing that builds the RAG index from a user's documents using ChromaDB.
    """
    global indexing_status
    try:
        user_doc_dir = os.path.join("user_documents", user_id)
        if not os.path.exists(user_doc_dir):
            indexing_status = {"status": "idle", "message": "No documents found for this user."}
            await rag_module.clear_user_index(user_id)
            return

        all_files_in_dir = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
        
        # If specific filenames are requested, filter them. Otherwise, use all files.
        files_to_index = filenames if filenames else all_files_in_dir
        
        if not files_to_index:
            indexing_status = {"status": "idle", "message": "No documents selected or found to index."}
            await rag_module.clear_user_index(user_id)
            return
        
        # Create full paths for the selected files
        file_paths = [os.path.join(user_doc_dir, f) for f in files_to_index if f in all_files_in_dir]

        # Use the ChromaDB-backed user indexing
        index = await rag_module.build_user_index(user_id, file_paths)
        
        if index:
            indexing_status = {"status": "success", "message": f"{len(file_paths)} document(s) loaded successfully into ChromaDB."}
            print(f"Successfully built ChromaDB index for user {user_id} from {len(file_paths)} file(s).")
        else:
            indexing_status = {"status": "error", "message": "Failed to build ChromaDB index."}
            print(f"Failed to build ChromaDB index for user {user_id}")
        
    except Exception as e:
        error_message = f"Failed to build ChromaDB index for {user_id}: {str(e)}"
        indexing_status = {"status": "error", "message": error_message}
        print(error_message)


def run_user_indexing(user_id: str, filenames: Optional[List[str]] = None):
    """
    Synchronous wrapper that schedules the async indexing function.
    This is called by the background task system.
    """
    # Schedule the async function to run
    asyncio.create_task(async_run_user_indexing(user_id, filenames))


@app.post("/api/documents/load")
async def load_user_documents(background_tasks: BackgroundTasks, user_id: str = Body(...), filenames: Optional[List[str]] = Body(None)):
    """
    Checks for existing documents and starts a background task to index them.
    Accepts an optional list of filenames to index specific documents.
    """
    global indexing_status
    indexing_status = {"status": "in_progress", "message": "Indexing started..."}
    # Use the async version directly since we're in an async context
    background_tasks.add_task(async_run_user_indexing, user_id, filenames)
    return {"message": "Document indexing process has been started."}


@app.get("/api/health/chromadb")
async def health_check_chromadb():
    """Health check endpoint for ChromaDB integration."""
    try:
        from app.chroma_client import chroma_manager
        client = chroma_manager.get_client()
        collections = client.list_collections()
        return {
            "status": "healthy",
            "collections_count": len(collections),
            "message": "ChromaDB is accessible"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "ChromaDB connection failed"
        }

@app.get("/api/debug/user-index/{user_id}")
async def debug_user_index(user_id: str):
    """Enhanced debug endpoint to check user index status with detailed information."""
    try:
        import time
        start_time = time.time()
        
        # Check if user has documents
        user_doc_dir = os.path.join("user_documents", user_id)
        documents = []
        document_details = []
        total_documents_size = 0
        
        if os.path.exists(user_doc_dir):
            documents = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
            for doc in documents:
                doc_path = os.path.join(user_doc_dir, doc)
                doc_size = os.path.getsize(doc_path)
                doc_modified = os.path.getmtime(doc_path)
                total_documents_size += doc_size
                
                # Try to get document row count
                try:
                    import pandas as pd
                    df = pd.read_csv(doc_path)
                    row_count = len(df)
                    columns = list(df.columns)
                except Exception as e:
                    row_count = "Error reading file"
                    columns = []
                
                document_details.append({
                    "filename": doc,
                    "size_bytes": doc_size,
                    "modified_timestamp": doc_modified,
                    "row_count": row_count,
                    "columns": columns
                })
        
        # Check if user has index
        user_index = await rag_module.get_user_index(user_id)
        has_index = user_index is not None
        
        # Get index details if available
        index_details = {}
        if has_index and user_index:
            try:
                # Get document count from index
                docstore = user_index.docstore
                if hasattr(docstore, 'docs'):
                    index_details["document_count"] = len(docstore.docs)
                    index_details["document_ids"] = list(docstore.docs.keys())[:10]  # First 10 IDs
                else:
                    index_details["document_count"] = "Unknown"
                    index_details["document_ids"] = []
                
                # Check if it's ChromaDB backed
                vector_store = getattr(user_index, '_vector_store', None)
                index_details["storage_type"] = "ChromaDB" if vector_store else "In-Memory"
                
            except Exception as e:
                index_details["error"] = str(e)
        
        # Check if user has retriever
        retriever = rag_module.get_fusion_retriever(user_id)
        has_retriever = retriever is not None
        
        # Check BM25 retriever details
        bm25_details = {}
        if user_id in rag_module.user_bm25_retrievers:
            bm25_retriever = rag_module.user_bm25_retrievers[user_id]
            try:
                # Use safer attribute access for BM25 retriever
                node_count = "Unknown"
                nodes = getattr(bm25_retriever, '_nodes', None) or getattr(bm25_retriever, 'nodes', None)
                if nodes:
                    node_count = len(nodes)
                
                bm25_details["node_count"] = node_count
                bm25_details["similarity_top_k"] = getattr(bm25_retriever, 'similarity_top_k', "Unknown")
            except Exception as e:
                bm25_details["error"] = str(e)
        
        # ChromaDB collection status
        chromadb_status = {}
        try:
            from app.chroma_client import chroma_manager
            collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
            if collection:
                collection_count = await asyncio.to_thread(collection.count)
                chromadb_status = {
                    "collection_exists": True,
                    "document_count": collection_count,
                    "collection_name": collection.name
                }
            else:
                chromadb_status = {"collection_exists": False}
        except Exception as e:
            chromadb_status = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return {
            "user_id": user_id,
            "execution_time_ms": round(execution_time * 1000, 2),
            "documents": {
                "found": documents,
                "count": len(documents),
                "total_size_bytes": total_documents_size,
                "details": document_details
            },
            "index": {
                "exists": has_index,
                "details": index_details
            },
            "retriever": {
                "fusion_retriever_available": has_retriever,
                "bm25_details": bm25_details
            },
            "chromadb": chromadb_status,
            "cache_status": {
                "cached_indexes": list(rag_module.user_indexes.keys()),
                "cached_bm25_retrievers": list(rag_module.user_bm25_retrievers.keys())
            }
        }
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}

@app.post("/api/debug/test-search")
async def debug_test_search(user_id: str = Body(...), query: str = Body(...)):
    """Enhanced debug endpoint to test specific search queries with detailed results."""
    try:
        import time
        start_time = time.time()
        
        # Get user index
        user_index = await rag_module.get_user_index(user_id)
        if not user_index:
            return {"error": "No index found for user", "user_id": user_id}
        
        # Get retriever
        retriever = rag_module.get_fusion_retriever(user_id)
        if not retriever:
            return {"error": "No retriever found for user", "user_id": user_id}
        
        # Test different search strategies
        search_results = {}
        
        # 1. Fusion retriever search
        try:
            fusion_start = time.time()
            fusion_results = await retriever.aretrieve(query)
            fusion_time = time.time() - fusion_start
            
            formatted_fusion_results = []
            for i, node in enumerate(fusion_results):
                formatted_fusion_results.append({
                    "rank": i + 1,
                    "score": getattr(node, 'score', 'N/A'),
                    "content": node.get_content()[:300] + "..." if len(node.get_content()) > 300 else node.get_content(),
                    "metadata": getattr(node, 'metadata', {}),
                    "doc_id": getattr(node, 'doc_id', 'N/A')
                })
            
            search_results["fusion_retriever"] = {
                "results_count": len(fusion_results),
                "execution_time_ms": round(fusion_time * 1000, 2),
                "results": formatted_fusion_results
            }
        except Exception as e:
            search_results["fusion_retriever"] = {"error": str(e)}
        
        # 2. Vector-only search
        try:
            vector_start = time.time()
            vector_retriever = user_index.as_retriever(similarity_top_k=5)
            vector_results = vector_retriever.retrieve(query)
            vector_time = time.time() - vector_start
            
            formatted_vector_results = []
            for i, node in enumerate(vector_results):
                formatted_vector_results.append({
                    "rank": i + 1,
                    "score": getattr(node, 'score', 'N/A'),
                    "content": node.get_content()[:300] + "..." if len(node.get_content()) > 300 else node.get_content(),
                    "metadata": getattr(node, 'metadata', {}),
                    "doc_id": getattr(node, 'doc_id', 'N/A')
                })
            
            search_results["vector_only"] = {
                "results_count": len(vector_results),
                "execution_time_ms": round(vector_time * 1000, 2),
                "results": formatted_vector_results
            }
        except Exception as e:
            search_results["vector_only"] = {"error": str(e)}
        
        # 3. BM25-only search (if available)
        if user_id in rag_module.user_bm25_retrievers:
            try:
                bm25_start = time.time()
                bm25_retriever = rag_module.user_bm25_retrievers[user_id]
                bm25_results = bm25_retriever.retrieve(query)
                bm25_time = time.time() - bm25_start
                
                formatted_bm25_results = []
                for i, node in enumerate(bm25_results):
                    formatted_bm25_results.append({
                        "rank": i + 1,
                        "score": getattr(node, 'score', 'N/A'),
                        "content": node.get_content()[:300] + "..." if len(node.get_content()) > 300 else node.get_content(),
                        "metadata": getattr(node, 'metadata', {}),
                        "doc_id": getattr(node, 'doc_id', 'N/A')
                    })
                
                search_results["bm25_only"] = {
                    "results_count": len(bm25_results),
                    "execution_time_ms": round(bm25_time * 1000, 2),
                    "results": formatted_bm25_results
                }
            except Exception as e:
                search_results["bm25_only"] = {"error": str(e)}
        else:
            search_results["bm25_only"] = {"error": "BM25 retriever not available"}
        
        total_time = time.time() - start_time
        
        return {
            "user_id": user_id,
            "query": query,
            "total_execution_time_ms": round(total_time * 1000, 2),
            "search_strategies": search_results,
            "summary": {
                "best_strategy": max(search_results.keys(), key=lambda k: search_results[k].get("results_count", 0) if isinstance(search_results[k], dict) and "error" not in search_results[k] else 0),
                "total_unique_results": len(set(
                    result.get("doc_id", f"unknown_{i}") 
                    for strategy in search_results.values() 
                    if isinstance(strategy, dict) and "results" in strategy
                    for i, result in enumerate(strategy["results"])
                ))
            }
        }
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__, "user_id": user_id, "query": query}

@app.post("/api/debug/multi-strategy-search")
async def debug_multi_strategy_search(user_id: str = Body(...), query: str = Body(...)):
    """Debug endpoint to test multi-strategy search with detailed results from each strategy."""
    try:
        import time
        start_time = time.time()
        
        # Get user index
        user_index = await rag_module.get_user_index(user_id)
        if not user_index:
            return {"error": "No index found for user", "user_id": user_id}
        
        # Perform multi-strategy search
        try:
            # Check if multi-strategy search is available
            if hasattr(rag_module, 'retrieve_context_optimized'):
                multi_search_result = await rag_module.retrieve_context_optimized(query, user_id)
            else:
                # Fallback to basic search if multi-strategy not available
                print("Multi-strategy search not available, using basic search")
                from app.multi_strategy_search import MultiStrategySearchResult
                
                # Create a basic search result
                fusion_retriever = rag_module.get_fusion_retriever(user_id)
                if fusion_retriever:
                    retrieved_nodes = await fusion_retriever.aretrieve(query)
                    multi_search_result = MultiStrategySearchResult(
                        original_query=query,
                        best_result=None,
                        all_results=[],
                        total_execution_time_ms=0.0,
                        nodes_found=retrieved_nodes
                    )
                else:
                    multi_search_result = MultiStrategySearchResult(
                        original_query=query,
                        best_result=None,
                        all_results=[],
                        total_execution_time_ms=0.0,
                        nodes_found=[]
                    )
        except Exception as e:
            print(f"Error during multi-strategy search: {e}")
            # Fallback to original search method
            try:
                fusion_retriever = rag_module.get_fusion_retriever(user_id)
                if fusion_retriever:
                    print("Falling back to original search method")
                    retrieved_nodes = await fusion_retriever.aretrieve(query)
                    print(f"Fallback search found {len(retrieved_nodes)} nodes")
                    multi_search_result = MultiStrategySearchResult(
                        original_query=query,
                        best_result=None,
                        all_results=[],
                        total_execution_time_ms=0.0,
                        nodes_found=retrieved_nodes
                    )
                else:
                    multi_search_result = MultiStrategySearchResult(
                        original_query=query,
                        best_result=None,
                        all_results=[],
                        total_execution_time_ms=0.0,
                        nodes_found=[]
                    )
            except Exception as fallback_error:
                print(f"Fallback search also failed: {fallback_error}")
                multi_search_result = MultiStrategySearchResult(
                    original_query=query,
                    best_result=None,
                    all_results=[],
                    total_execution_time_ms=0.0,
                    nodes_found=[]
                )
        
        # Format results for debugging
        formatted_results = {
            "user_id": user_id,
            "original_query": query,
            "total_execution_time_ms": multi_search_result.total_execution_time_ms,
            "strategies_executed": len(multi_search_result.all_results),
            "total_nodes_found": len(multi_search_result.nodes_found),
            "best_strategy": multi_search_result.best_result.strategy if multi_search_result.best_result else None,
            "strategy_details": [],
            "final_ranked_results": [],
            "address_analysis": {},
            "query_analysis": {}
        }
        
        # Add details for each strategy
        for result in multi_search_result.all_results:
            strategy_detail = {
                "strategy": result.strategy,
                "query_used": result.query_used,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "nodes_found": len(result.nodes),
                "results": []
            }
            
            # Format nodes for this strategy
            for i, node in enumerate(result.nodes):
                strategy_detail["results"].append({
                    "rank": i + 1,
                    "score": getattr(node, 'score', 'N/A'),
                    "content": node.get_content()[:300] + "..." if len(node.get_content()) > 300 else node.get_content(),
                    "metadata": getattr(node, 'metadata', {}),
                    "doc_id": getattr(node, 'doc_id', 'N/A')
                })
            
            formatted_results["strategy_details"].append(strategy_detail)
        
        # Add final ranked results
        for i, node in enumerate(multi_search_result.nodes_found):
            formatted_results["final_ranked_results"].append({
                "final_rank": i + 1,
                "score": getattr(node, 'score', 'N/A'),
                "content": node.get_content()[:300] + "..." if len(node.get_content()) > 300 else node.get_content(),
                "metadata": getattr(node, 'metadata', {}),
                "doc_id": getattr(node, 'doc_id', 'N/A')
            })
        
        # Add address analysis
        from app.multi_strategy_search import AddressExtractor, QueryProcessor
        addresses = AddressExtractor.extract_addresses(query)
        key_terms = QueryProcessor.extract_key_terms(query)
        
        formatted_results["address_analysis"] = {
            "addresses_found": addresses,
            "address_count": len(addresses),
            "normalized_addresses": [AddressExtractor.normalize_address(addr) for addr in addresses]
        }
        
        formatted_results["query_analysis"] = {
            "key_terms": key_terms,
            "key_terms_count": len(key_terms),
            "fuzzy_query": QueryProcessor.create_fuzzy_query(query)
        }
        
        total_time = time.time() - start_time
        formatted_results["debug_execution_time_ms"] = round(total_time * 1000, 2)
        
        return formatted_results
        
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__, "user_id": user_id, "query": query}

@app.post("/api/debug/validate-documents")
async def debug_validate_documents(user_id: str = Body(...)):
    """Debug endpoint to validate user document processing."""
    try:
        import time
        import pandas as pd
        start_time = time.time()
        
        user_doc_dir = os.path.join("user_documents", user_id)
        if not os.path.exists(user_doc_dir):
            return {"error": f"User document directory not found: {user_doc_dir}"}
        
        validation_results = {
            "user_id": user_id,
            "document_directory": user_doc_dir,
            "documents": [],
            "processing_errors": [],
            "index_validation": {},
            "summary": {}
        }
        
        # Find and validate CSV files
        csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            file_path = os.path.join(user_doc_dir, filename)
            doc_validation = {
                "filename": filename,
                "file_path": file_path,
                "file_size_bytes": os.path.getsize(file_path),
                "readable": False,
                "row_count": 0,
                "column_count": 0,
                "columns": [],
                "sample_rows": [],
                "data_types": {},
                "missing_values": {},
                "errors": []
            }
            
            try:
                # Try to read the CSV file
                df = pd.read_csv(file_path)
                doc_validation["readable"] = True
                doc_validation["row_count"] = len(df)
                doc_validation["column_count"] = len(df.columns)
                doc_validation["columns"] = list(df.columns)
                
                # Get sample rows (first 3)
                sample_rows = df.head(3).to_dict('records')
                doc_validation["sample_rows"] = sample_rows
                
                # Get data types
                doc_validation["data_types"] = df.dtypes.astype(str).to_dict()
                
                # Check for missing values
                doc_validation["missing_values"] = df.isnull().sum().to_dict()
                
                # Validate required columns for property data
                expected_columns = ['property address', 'monthly rent', 'size (sf)']
                standardized_columns = [col.strip().lower() for col in df.columns]
                missing_expected = [col for col in expected_columns if col not in standardized_columns]
                if missing_expected:
                    doc_validation["errors"].append(f"Missing expected columns: {missing_expected}")
                
            except Exception as e:
                doc_validation["errors"].append(f"Failed to read CSV: {str(e)}")
                validation_results["processing_errors"].append({
                    "filename": filename,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            validation_results["documents"].append(doc_validation)
        
        # Validate index creation
        try:
            user_index = await rag_module.get_user_index(user_id)
            if user_index:
                validation_results["index_validation"]["index_exists"] = True
                
                # Check document count in index
                docstore = user_index.docstore
                if hasattr(docstore, 'docs'):
                    validation_results["index_validation"]["indexed_document_count"] = len(docstore.docs)
                    validation_results["index_validation"]["sample_doc_ids"] = list(docstore.docs.keys())[:5]
                
                # Test retriever creation
                retriever = rag_module.get_fusion_retriever(user_id)
                validation_results["index_validation"]["retriever_available"] = retriever is not None
                
                # Test a simple search
                if retriever:
                    try:
                        test_results = await retriever.aretrieve("test query")
                        validation_results["index_validation"]["search_test"] = {
                            "success": True,
                            "results_count": len(test_results)
                        }
                    except Exception as e:
                        validation_results["index_validation"]["search_test"] = {
                            "success": False,
                            "error": str(e)
                        }
            else:
                validation_results["index_validation"]["index_exists"] = False
                validation_results["index_validation"]["error"] = "Failed to get or create index"
        except Exception as e:
            validation_results["index_validation"]["error"] = str(e)
        
        # Generate summary
        total_rows = sum(doc["row_count"] for doc in validation_results["documents"] if doc["readable"])
        readable_docs = sum(1 for doc in validation_results["documents"] if doc["readable"])
        
        validation_results["summary"] = {
            "total_csv_files": len(csv_files),
            "readable_files": readable_docs,
            "total_data_rows": total_rows,
            "processing_errors_count": len(validation_results["processing_errors"]),
            "index_ready": validation_results["index_validation"].get("index_exists", False),
            "retriever_ready": validation_results["index_validation"].get("retriever_available", False)
        }
        
        validation_results["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return validation_results
        
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__, "user_id": user_id}

@app.post("/api/debug/rebuild-index")
async def debug_rebuild_index(user_id: str = Body(...), force: bool = Body(default=False)):
    """Debug endpoint to force index rebuild with progress tracking."""
    try:
        import time
        start_time = time.time()
        
        rebuild_log = {
            "user_id": user_id,
            "force_rebuild": force,
            "steps": [],
            "success": False,
            "error": None
        }
        
        # Step 1: Check user documents
        rebuild_log["steps"].append({"step": "check_documents", "status": "started", "timestamp": time.time()})
        
        user_doc_dir = os.path.join("user_documents", user_id)
        if not os.path.exists(user_doc_dir):
            rebuild_log["steps"][-1]["status"] = "failed"
            rebuild_log["steps"][-1]["error"] = f"User document directory not found: {user_doc_dir}"
            rebuild_log["error"] = "No documents found"
            return rebuild_log
        
        csv_files = [f for f in os.listdir(user_doc_dir) if f.endswith('.csv')]
        if not csv_files:
            rebuild_log["steps"][-1]["status"] = "failed"
            rebuild_log["steps"][-1]["error"] = "No CSV files found"
            rebuild_log["error"] = "No CSV files to index"
            return rebuild_log
        
        rebuild_log["steps"][-1]["status"] = "completed"
        rebuild_log["steps"][-1]["files_found"] = csv_files
        
        # Step 2: Clear existing index if force rebuild
        if force:
            rebuild_log["steps"].append({"step": "clear_existing_index", "status": "started", "timestamp": time.time()})
            try:
                success = await rag_module.clear_user_index(user_id)
                rebuild_log["steps"][-1]["status"] = "completed" if success else "failed"
                rebuild_log["steps"][-1]["cleared"] = success
            except Exception as e:
                rebuild_log["steps"][-1]["status"] = "failed"
                rebuild_log["steps"][-1]["error"] = str(e)
        
        # Step 3: Build new index
        rebuild_log["steps"].append({"step": "build_index", "status": "started", "timestamp": time.time()})
        
        try:
            file_paths = [os.path.join(user_doc_dir, f) for f in csv_files]
            user_index = await rag_module.build_user_index(user_id, file_paths)
            
            if user_index:
                rebuild_log["steps"][-1]["status"] = "completed"
                rebuild_log["steps"][-1]["index_created"] = True
                
                # Get index details
                docstore = user_index.docstore
                if hasattr(docstore, 'docs'):
                    rebuild_log["steps"][-1]["documents_indexed"] = len(docstore.docs)
            else:
                rebuild_log["steps"][-1]["status"] = "failed"
                rebuild_log["steps"][-1]["error"] = "Index creation returned None"
                rebuild_log["error"] = "Failed to create index"
                return rebuild_log
                
        except Exception as e:
            rebuild_log["steps"][-1]["status"] = "failed"
            rebuild_log["steps"][-1]["error"] = str(e)
            rebuild_log["error"] = f"Index building failed: {str(e)}"
            return rebuild_log
        
        # Step 4: Validate retriever creation
        rebuild_log["steps"].append({"step": "validate_retriever", "status": "started", "timestamp": time.time()})
        
        try:
            retriever = rag_module.get_fusion_retriever(user_id)
            if retriever:
                rebuild_log["steps"][-1]["status"] = "completed"
                rebuild_log["steps"][-1]["retriever_created"] = True
                
                # Test search functionality
                test_results = await retriever.aretrieve("test")
                rebuild_log["steps"][-1]["search_test_results"] = len(test_results)
            else:
                rebuild_log["steps"][-1]["status"] = "failed"
                rebuild_log["steps"][-1]["error"] = "Retriever creation failed"
                rebuild_log["error"] = "Retriever validation failed"
                return rebuild_log
                
        except Exception as e:
            rebuild_log["steps"][-1]["status"] = "failed"
            rebuild_log["steps"][-1]["error"] = str(e)
            rebuild_log["error"] = f"Retriever validation failed: {str(e)}"
            return rebuild_log
        
        # Success!
        rebuild_log["success"] = True
        rebuild_log["total_execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return rebuild_log
        
    except Exception as e:
        return {
            "user_id": user_id,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "force_rebuild": force
        }

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
        filename = file.filename if file.filename else "uploaded_file.csv"
        file_path = os.path.join(user_doc_dir, filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Start background indexing
        background_tasks.add_task(async_run_user_indexing, user_id)
        
        return {"message": f"File '{filename}' uploaded successfully. Indexing started in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.get("/api/documents/status")
async def get_indexing_status():
    """Returns the current status of the document indexing job."""
    return indexing_status

@app.get("/api/health/chromadb")
async def get_chromadb_health():
    """Returns the health status of ChromaDB connection."""
    try:
        from app.chroma_client import chroma_manager
        client = await asyncio.to_thread(chroma_manager.get_client)
        # Try to list collections to test connection
        collections = await asyncio.to_thread(client.list_collections)
        return {
            "status": "healthy",
            "message": "ChromaDB connection is working",
            "collections_count": len(collections)
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "message": f"ChromaDB connection failed: {str(e)}"
        }

@app.post("/api/logout")
async def logout():
    """
    Handles user logout. In this stateless app, it simply returns a success
    message. In a stateful app, this would invalidate a server-side session or token.
    """
    return {"message": "Logout successful"}


@app.post("/api/reset")
async def reset_application_data(user_id: Optional[str] = None):
    """
    Deletes data. If a user_id (email) is provided, only that user's
    conversations, CRM entry, and ChromaDB collection are deleted. Otherwise, all application
    data (RAG index, all users, all companies, all conversations, all ChromaDB collections) is wiped.
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
            
            # Clear user's ChromaDB collection
            await rag_module.clear_user_index(user_id)
            
            # Note: This does not delete the company, as it might be shared.
            return {"message": f"All conversation, CRM data, and ChromaDB collection for user '{user_id}' has been reset."}

        else:
            # Global reset
            # Clear all ChromaDB collections by clearing all user indexes
            rag_module.clear_index()
            
            # Clear database collections
            await db["users"].delete_many({})
            await db["companies"].delete_many({})
            await db["conversation_history"].delete_many({})
            
            return {"message": "All application data and ChromaDB collections have been reset."}

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

# Add this after the existing chat endpoint

@app.post("/api/chat/recommendation-response")
async def recommendation_response(session_id: str = Body(...), user_response: str = Body(...)):
    """
    Handle user responses within an active recommendation workflow.
    
    This endpoint is used when users are responding to clarifying questions
    during the recommendation process.
    """
    print(f"🔄 Processing recommendation response for session {session_id}")
    
    try:
        # Process the user response in the workflow
        workflow_step = await recommendation_workflow_manager.process_user_response(session_id, user_response)
        
        if workflow_step.success:
            print(f"✅ Workflow step completed: {workflow_step.step_name}")
            
            # If workflow is complete, generate final recommendations
            if workflow_step.step_name == "recommendations_generated":
                recommendation_result = await recommendation_workflow_manager.complete_workflow(session_id)
                
                return {
                    "success": True,
                    "message": workflow_step.response_message,
                    "workflow_complete": True,
                    "recommendations": [rec.model_dump() for rec in recommendation_result.recommendations]
                }
            else:
                return {
                    "success": True,
                    "message": workflow_step.response_message,
                    "workflow_complete": False,
                    "next_step": workflow_step.next_step
                }
        else:
            print(f"❌ Workflow step failed: {workflow_step.step_name}")
            return {
                "success": False,
                "message": workflow_step.response_message,
                "workflow_complete": True,
                "error": f"Workflow step failed: {workflow_step.step_name}"
            }
            
    except Exception as e:
        print(f"❌ Error processing recommendation response: {e}")
        return {
            "success": False,
            "message": "I encountered an issue processing your response. Let's start fresh with your property search.",
            "workflow_complete": True,
            "error": str(e)
        }

@app.get("/api/debug/recommendation-workflow/{session_id}")
async def debug_recommendation_workflow(session_id: str):
    """
    Debug endpoint to check recommendation workflow status.
    """
    try:
        from app.database import get_database
        
        db = get_database()
        
        # Get workflow session
        workflow_session = await db["workflow_sessions"].find_one({"_id": session_id})
        
        # Get conversation session if available
        conversation_session = None
        if workflow_session and workflow_session.get("data", {}).get("conversation_session_id"):
            conv_session_id = workflow_session["data"]["conversation_session_id"]
            conversation_session = await db["conversation_sessions"].find_one({"_id": conv_session_id})
        
        return {
            "session_id": session_id,
            "workflow_session": workflow_session,
            "conversation_session": conversation_session,
            "session_found": workflow_session is not None
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "session_found": False
        }

@app.post("/api/debug/test-recommendation-intent")
async def test_recommendation_intent(message: str = Body(...)):
    """
    Debug endpoint to test recommendation intent detection.
    """
    try:
        intent = await intent_detection_service.detect_recommendation_intent(message)
        return {
            "message": message,
            "intent": intent.model_dump(),
            "is_recommendation": intent.is_recommendation_request,
            "confidence": intent.confidence
        }
    except Exception as e:
        return {
            "message": message,
            "error": str(e),
            "is_recommendation": False,
            "confidence": 0.0
        }

# Add this after the existing recommendation response endpoint

@app.post("/api/chat/appointment-response")
async def appointment_response(session_id: str = Body(...), user_response: str = Body(...)):
    """
    Handle user responses within an active appointment booking workflow.
    
    This endpoint is used when users are responding to questions during
    the appointment booking process.
    """
    print(f"🗓️ Processing appointment response for session {session_id}")
    
    try:
        # Process the user response in the workflow
        workflow_response = await appointment_workflow_manager.process_user_response(session_id, user_response)
        
        if workflow_response.success:
            print(f"✅ Appointment workflow step completed: {workflow_response.step_name}")
            
            # If workflow is at confirmation stage, include UI components
            if workflow_response.step_name == "awaiting_confirmation" and workflow_response.ui_components:
                return {
                    "success": True,
                    "message": workflow_response.message,
                    "workflow_complete": False,
                    "step_name": workflow_response.step_name,
                    "ui_components": workflow_response.ui_components.__dict__,
                    "session_id": workflow_response.session_id
                }
            elif workflow_response.step_name == "appointment_confirmed":
                return {
                    "success": True,
                    "message": workflow_response.message,
                    "workflow_complete": True,
                    "step_name": workflow_response.step_name,
                    "appointment_data": workflow_response.appointment_data.__dict__ if workflow_response.appointment_data else None
                }
            else:
                return {
                    "success": True,
                    "message": workflow_response.message,
                    "workflow_complete": False,
                    "step_name": workflow_response.step_name,
                    "next_step": workflow_response.next_step,
                    "session_id": workflow_response.session_id
                }
        else:
            print(f"❌ Appointment workflow step failed: {workflow_response.step_name}")
            return {
                "success": False,
                "message": workflow_response.message,
                "workflow_complete": True,
                "error": workflow_response.error_details.__dict__ if workflow_response.error_details else None
            }
            
    except Exception as e:
        print(f"❌ Error processing appointment response: {e}")
        return {
            "success": False,
            "message": "I encountered an issue processing your response. Let's start fresh with your appointment booking.",
            "workflow_complete": True,
            "error": str(e)
        }

@app.post("/api/appointments/confirm")
async def confirm_appointment(session_id: str = Body(...)):
    """
    Confirm an appointment booking.
    """
    print(f"📅 Confirming appointment for session {session_id}")
    try:
        workflow_response = await appointment_workflow_manager.confirm_appointment_workflow(session_id)
        
        if workflow_response.success:
            print(f"✅ Appointment confirmed successfully")
            return {
                "success": True,
                "message": workflow_response.message,
                "appointment_data": workflow_response.appointment_data.__dict__ if workflow_response.appointment_data else None
            }
        else:
            print(f"❌ Appointment confirmation failed")
            return {
                "success": False,
                "message": workflow_response.message,
                "error": workflow_response.error_details.__dict__ if workflow_response.error_details else None
            }
            
    except Exception as e:
        print(f"❌ Error confirming appointment: {e}")
        return {
            "success": False,
            "message": "Sorry, there was an issue confirming your appointment. Please try again.",
            "error": str(e)
        }

@app.post("/api/appointments/cancel")
async def cancel_appointment(session_id: str = Body(...)):
    """
    Cancel an appointment booking.
    """
    print(f"❌ Cancelling appointment for session {session_id}")
    
    try:
        workflow_response = await appointment_workflow_manager.cancel_appointment(session_id)
        
        return {
            "success": True,
            "message": workflow_response.message,
            "cancelled": True
        }
        
    except Exception as e:
        print(f"❌ Error cancelling appointment: {e}")
        return {
            "success": True,  # Still successful from user perspective
            "message": "Your appointment booking has been cancelled. Feel free to ask me anything else!",
            "cancelled": True
        }

@app.get("/api/debug/appointment-workflow/{session_id}")
async def debug_appointment_workflow(session_id: str):
    """
    Debug endpoint to check appointment workflow status.
    """
    try:
        from app.database import get_database
        
        db = get_database()
        
        # Get appointment session
        appointment_session = await db["appointment_sessions"].find_one({"_id": session_id})
        
        return {
            "session_id": session_id,
            "appointment_session": appointment_session,
            "session_found": appointment_session is not None
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "session_found": False
        }

@app.post("/api/debug/test-appointment-intent")
async def test_appointment_intent(message: str = Body(...)):
    """
    Debug endpoint to test appointment intent detection.
    """
    try:
        intent = await appointment_intent_detection_service.detect_appointment_intent(message)
        return {
            "message": message,
            "intent": intent.__dict__,
            "is_appointment": intent.is_appointment_request,
            "confidence": intent.confidence
        }
    except Exception as e:
        return {
            "message": message,
            "error": str(e),
            "is_appointment": False,
            "confidence": 0.0
        }
