# /app/models.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict
from bson import ObjectId
import datetime as dt

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
    message: str
    history: List[dict] # Comes from client as list of dicts
    user_id: Optional[str] = None # User's email

class ChatResponse(BaseModel):
    answer: str
    user: Optional[User] = None
    schedule_details: Optional[dict] = None

class DocumentUploadRequest(BaseModel):
    filename: str
    content: str # Base64 encoded content

class ScheduleRequest(BaseModel):
    email: EmailStr
    address: str
    time: str # ISO 8601 format