# /app/history.py
from app.database import get_database
from app.models import ConversationHistory, ChatMessage as ChatMessageModel
from motor.motor_asyncio import AsyncIOMotorCollection
from typing import List
import datetime as dt

async def get_history_collection() -> AsyncIOMotorCollection:
    """Returns the conversation_history collection from the database."""
    db = get_database()
    return db["conversation_history"]

async def get_user_history(user_email: str) -> List[dict]:
    """
    Retrieves the most recent conversation history for a user.
    """
    history_collection = await get_history_collection()
    # Find the most recently updated conversation for the user
    conversation_data = await history_collection.find_one(
        {"user_email": user_email},
        sort=[("updated_at", -1)]
    )
    if conversation_data and "messages" in conversation_data:
        # Convert Pydantic models to dicts for API response
        return [msg.model_dump() for msg in ConversationHistory(**conversation_data).messages]
    return []

async def add_message_to_history(user_email: str, user_message: str, assistant_message: str):
    """
    Adds a new user message and assistant response to the user's conversation history.
    It updates an existing conversation or creates a new one.
    """
    history_collection = await get_history_collection()
    
    new_messages = [
        ChatMessageModel(role="user", content=user_message),
        ChatMessageModel(role="assistant", content=assistant_message),
    ]
    
    # Find the most recent conversation for the user
    conversation = await history_collection.find_one(
        {"user_email": user_email},
        sort=[("updated_at", -1)]
    )

    if conversation:
        # Append messages to the existing conversation
        await history_collection.update_one(
            {"_id": conversation["_id"]},
            {
                "$push": {"messages": {"$each": [msg.model_dump() for msg in new_messages]}},
                "$set": {"updated_at": dt.datetime.utcnow()}
            }
        )
        print(f"Appended messages to existing history for user {user_email}.")
    else:
        # Create a new conversation history document
        new_history = ConversationHistory(
            user_email=user_email,
            messages=new_messages
        )
        history_dict = new_history.model_dump(by_alias=True, exclude_none=True)
        await history_collection.insert_one(history_dict)
        print(f"Created new conversation history for user {user_email}.") 