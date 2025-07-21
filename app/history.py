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

async def add_message_to_history(user_email: str, user_message: str, assistant_message: str, user_tags=None, user_category=None, assistant_tags=None, assistant_category=None):
    """
    Adds a new user message and assistant response to the user's conversation history.
    It updates an existing conversation or creates a new one.
    Now supports optional tags and category for each message.
    """
    history_collection = await get_history_collection()
    
    new_messages = [
        ChatMessageModel(role="user", content=user_message, tags=user_tags or [], category=user_category),
        ChatMessageModel(role="assistant", content=assistant_message, tags=assistant_tags or [], category=assistant_category),
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

async def update_message_tags_and_category(user_email: str, message_index: int, tags=None, category=None):
    """
    Update tags and/or category for a specific message in the user's most recent conversation.
    """
    history_collection = await get_history_collection()
    conversation = await history_collection.find_one(
        {"user_email": user_email},
        sort=[("updated_at", -1)]
    )
    if not conversation or "messages" not in conversation or message_index >= len(conversation["messages"]):
        return False
    update_fields = {}
    if tags is not None:
        update_fields[f"messages.{message_index}.tags"] = tags
    if category is not None:
        update_fields[f"messages.{message_index}.category"] = category
    if update_fields:
        await history_collection.update_one(
            {"_id": conversation["_id"]},
            {"$set": update_fields}
        )
        return True
    return False 