# /app/crm.py
import re
from typing import Optional
from app.database import get_database
from app.models import User, Company
from motor.motor_asyncio import AsyncIOMotorCollection

def get_domain_from_email(email: str) -> Optional[str]:
    """Extracts the domain from an email address."""
    match = re.search(r"@([\w.-]+)", email)
    return match.group(1) if match else None

async def get_or_create_company(domain: str, company_name: Optional[str] = None) -> Company:
    """
    Retrieves a company by its domain. If it doesn't exist, a new company is created.
    """
    db = get_database()
    company_collection: AsyncIOMotorCollection = db["companies"]
    
    # Use a case-insensitive query for the domain
    company_data = await company_collection.find_one({"domain": re.compile(f"^{re.escape(domain)}$", re.IGNORECASE)})
    
    if company_data:
        return Company(**company_data)
    else:
        # If no company name is provided, use the domain
        name_to_use = company_name if company_name else domain
        new_company = Company(name=name_to_use, domain=domain)
        
        # Convert to dict, excluding the 'id' field if it's None before insertion
        company_dict = new_company.model_dump(by_alias=True, exclude_none=True)

        result = await company_collection.insert_one(company_dict)
        new_company.id = result.inserted_id
        return new_company

async def get_user_by_email(email: str) -> Optional[User]:
    """
    Retrieves a user by their email address.
    """
    db = get_database()
    user_collection: AsyncIOMotorCollection = db["users"]
    user_data = await user_collection.find_one({"email": email})
    return User(**user_data) if user_data else None


async def create_or_update_user(
    email: str,
    full_name: Optional[str] = None,
    company_name: Optional[str] = None,
    preferences: Optional[dict] = None
) -> User:
    """
    Creates a new user or updates an existing one.
    Handles partial updates and merges preferences.
    """
    db = get_database()
    user_collection: AsyncIOMotorCollection = db["users"]
    
    # Check if user already exists
    existing_user_data = await user_collection.find_one({"email": email})

    if existing_user_data:
        # User exists, prepare for update
        update_data = {}
        if full_name:
            update_data["full_name"] = full_name
        
        # If company name is provided, get/create company and update ID
        if company_name:
            domain = get_domain_from_email(email)
            if domain:
                company = await get_or_create_company(domain, company_name)
                update_data["company_id"] = company.id
        
        # Merge preferences: new values overwrite old ones
        if preferences:
            # Using dot notation for updating nested dictionary fields in MongoDB
            for key, value in preferences.items():
                update_data[f"preferences.{key}"] = value

        if update_data:
            await user_collection.update_one({"_id": existing_user_data["_id"]}, {"$set": update_data})
        
        # Fetch the updated user data
        updated_user_data = await user_collection.find_one({"_id": existing_user_data["_id"]})
        if not updated_user_data:
            # This case is unlikely but possible if the user is deleted between operations.
            raise ValueError(f"User with ID {existing_user_data['_id']} could not be found after update.")
        return User(**updated_user_data)

    else:
        # User does not exist, create new one
        if not full_name:
            raise ValueError("Full name is required for new user creation.")

        domain = get_domain_from_email(email)
        company_id = None
        if domain:
            company = await get_or_create_company(domain, company_name)
            company_id = company.id

        new_user = User(
            full_name=full_name,
            email=email,
            company_id=company_id,
            preferences=preferences or {}
        )
        user_dict = new_user.model_dump(by_alias=True, exclude_none=True)
        result = await user_collection.insert_one(user_dict)
        new_user.id = result.inserted_id
        return new_user


async def delete_user_by_email(email: str) -> bool:
    """
    Deletes a single user from the database based on their email.
    Returns True if a user was deleted, False otherwise.
    """
    db = get_database()
    user_collection: AsyncIOMotorCollection = db["users"]
    delete_result = await user_collection.delete_one({"email": email})
    return delete_result.deleted_count > 0