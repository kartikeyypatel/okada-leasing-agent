# /app/models.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Tuple
from bson import ObjectId
import datetime as dt
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, *args, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class Company(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str
    domain: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True


class User(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    full_name: str
    email: EmailStr
    company_id: Optional[PyObjectId] = None
    preferences: Dict = Field(default_factory=dict)
    scheduled_events: List[str] = Field(default_factory=list) # To store event URLs


    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

class ChatMessage(BaseModel):
    role: str
    content: str
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None

class ConversationHistory(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_email: EmailStr
    messages: List[ChatMessage]
    tags: List[str] = Field(default_factory=list)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)

class ScheduleDetails(BaseModel):
    address: str
    time: str

class ProcessingMetadata(BaseModel):
    response_time_ms: float
    tokens_used: Optional[int] = None
    documents_retrieved: int = 0
    processing_steps: List[str] = Field(default_factory=list)
    rag_enabled: bool = False
    user_info_extracted: bool = False

class ChatResponse(BaseModel):
    answer: str
    schedule_details: Optional[ScheduleDetails] = None
    metadata: ProcessingMetadata

class DocumentUploadRequest(BaseModel):
    user_id: str
    filenames: List[str] = Field(default_factory=list)

class ScheduleRequest(BaseModel):
    email: str
    address: str
    time: str

class AppointmentState(BaseModel):
    user_id: str
    step: str  # "property_number", "purpose", "datetime", "email", "confirmation"
    property_number: Optional[str] = None
    purpose: Optional[str] = None
    datetime: Optional[str] = None
    email: Optional[str] = None
    property_address: Optional[str] = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class PropertySuggestionState(BaseModel):
    user_id: str
    step: str  # "budget", "size", "location", "type", "amenities", "analysis"
    budget: Optional[str] = None
    size_preference: Optional[str] = None
    location_preference: Optional[str] = None
    property_type: Optional[str] = None
    amenities: Optional[str] = None
    additional_requirements: Optional[str] = None
    chat_history_preferences: Dict = Field(default_factory=dict)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

# End of models - optimization models removed