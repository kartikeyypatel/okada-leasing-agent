# /app/database_migrations.py
"""
Database Migrations for Smart Property Recommendations

This module handles database schema migrations and data updates for the
Smart Property Recommendations system, ensuring proper data structure
and backward compatibility.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from pymongo import IndexModel, ASCENDING, DESCENDING
from app.database import get_database

logger = logging.getLogger(__name__)


class DatabaseMigrationManager:
    """
    Manages database migrations for the recommendation system.
    """
    
    def __init__(self):
        self.migration_history = []
    
    async def run_all_migrations(self) -> Dict[str, Any]:
        """
        Run all pending migrations for the recommendation system.
        
        Returns:
            Migration results and status
        """
        logger.info("Starting database migrations for recommendation system")
        
        migration_results = {
            "migrations_run": [],
            "migrations_skipped": [],
            "errors": [],
            "total_migrations": 0,
            "successful_migrations": 0
        }
        
        # List of all migrations to run
        migrations = [
            ("001_add_recommendation_fields_to_users", self._migration_001_add_recommendation_fields),
            ("002_create_workflow_sessions_collection", self._migration_002_create_workflow_sessions),
            ("003_create_conversation_sessions_collection", self._migration_003_create_conversation_sessions),
            ("004_create_recommendation_activities_collection", self._migration_004_create_activities),
            ("005_add_indexes_for_performance", self._migration_005_add_indexes),
            ("006_migrate_existing_user_data", self._migration_006_migrate_user_data),
            ("007_create_recommendation_analytics_collection", self._migration_007_create_analytics)
        ]
        
        migration_results["total_migrations"] = len(migrations)
        
        for migration_name, migration_func in migrations:
            try:
                # Check if migration was already run
                if await self._is_migration_completed(migration_name):
                    migration_results["migrations_skipped"].append(migration_name)
                    logger.info(f"Migration {migration_name} already completed, skipping")
                    continue
                
                # Run the migration
                logger.info(f"Running migration: {migration_name}")
                await migration_func()
                
                # Mark migration as completed
                await self._mark_migration_completed(migration_name)
                
                migration_results["migrations_run"].append(migration_name)
                migration_results["successful_migrations"] += 1
                logger.info(f"Migration {migration_name} completed successfully")
                
            except Exception as e:
                error_msg = f"Migration {migration_name} failed: {str(e)}"
                logger.error(error_msg)
                migration_results["errors"].append(error_msg)
        
        logger.info(f"Database migrations completed. {migration_results['successful_migrations']}/{migration_results['total_migrations']} successful")
        return migration_results
    
    async def _migration_001_add_recommendation_fields(self) -> None:
        """Add recommendation fields to existing users collection."""
        db = get_database()
        users_collection = db["users"]
        
        # Add recommendation fields to users who don't have them
        update_result = await users_collection.update_many(
            {"recommendation_preferences": {"$exists": False}},
            {"$set": {
                "recommendation_preferences": {},
                "last_recommendation_date": None,
                "recommendation_history": [],
                "appointment_history": [],
                "appointment_preferences": {}
            }}
        )
        
        logger.info(f"Updated {update_result.modified_count} users with recommendation fields")
    
    async def _migration_002_create_workflow_sessions(self) -> None:
        """Create workflow_sessions collection with proper schema."""
        db = get_database()
        
        # Create collection if it doesn't exist
        collections = await db.list_collection_names()
        if "workflow_sessions" not in collections:
            await db.create_collection("workflow_sessions")
            logger.info("Created workflow_sessions collection")
        
        # Ensure proper schema validation
        workflow_sessions = db["workflow_sessions"]
        
        # Create validation schema
        validation_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["_id", "user_id", "current_step"],
                "properties": {
                    "_id": {"bsonType": "string"},
                    "user_id": {"bsonType": "string"},
                    "current_step": {"bsonType": "string"},
                    "data": {"bsonType": "object"},
                    "created_at": {"bsonType": "date"},
                    "updated_at": {"bsonType": "date"}
                }
            }
        }
        
        try:
            await db.command("collMod", "workflow_sessions", validator=validation_schema)
            logger.info("Added validation schema to workflow_sessions collection")
        except Exception as e:
            logger.warning(f"Could not add validation schema to workflow_sessions: {e}")
    
    async def _migration_003_create_conversation_sessions(self) -> None:
        """Create conversation_sessions collection with proper schema."""
        db = get_database()
        
        # Create collection if it doesn't exist
        collections = await db.list_collection_names()
        if "conversation_sessions" not in collections:
            await db.create_collection("conversation_sessions")
            logger.info("Created conversation_sessions collection")
        
        # Create validation schema
        validation_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["_id", "user_id", "state"],
                "properties": {
                    "_id": {"bsonType": "string"},
                    "user_id": {"bsonType": "string"},
                    "state": {"bsonType": "string"},
                    "collected_preferences": {"bsonType": "object"},
                    "questions_asked": {"bsonType": "array"},
                    "responses_received": {"bsonType": "array"},
                    "created_at": {"bsonType": "date"},
                    "updated_at": {"bsonType": "date"}
                }
            }
        }
        
        try:
            conversation_sessions = db["conversation_sessions"]
            await db.command("collMod", "conversation_sessions", validator=validation_schema)
            logger.info("Added validation schema to conversation_sessions collection")
        except Exception as e:
            logger.warning(f"Could not add validation schema to conversation_sessions: {e}")
    
    async def _migration_004_create_activities(self) -> None:
        """Create recommendation_activities collection for tracking."""
        db = get_database()
        
        # Create collection if it doesn't exist
        collections = await db.list_collection_names()
        if "recommendation_activities" not in collections:
            await db.create_collection("recommendation_activities")
            logger.info("Created recommendation_activities collection")
        
        # Create validation schema
        validation_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["user_email", "activity_type", "timestamp"],
                "properties": {
                    "user_email": {"bsonType": "string"},
                    "activity_type": {"bsonType": "string"},
                    "session_id": {"bsonType": ["string", "null"]},
                    "metadata": {"bsonType": "object"},
                    "timestamp": {"bsonType": "date"}
                }
            }
        }
        
        try:
            activities_collection = db["recommendation_activities"]
            await db.command("collMod", "recommendation_activities", validator=validation_schema)
            logger.info("Added validation schema to recommendation_activities collection")
        except Exception as e:
            logger.warning(f"Could not add validation schema to recommendation_activities: {e}")
    
    async def _migration_005_add_indexes(self) -> None:
        """Add performance indexes for recommendation collections."""
        db = get_database()
        
        # Indexes for users collection
        users_collection = db["users"]
        user_indexes = [
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("last_recommendation_date", DESCENDING)]),
            IndexModel([("recommendation_preferences.budget.min", ASCENDING)]),
            IndexModel([("recommendation_preferences.budget.max", ASCENDING)]),
            IndexModel([("recommendation_preferences.location", ASCENDING)])
        ]
        
        try:
            await users_collection.create_indexes(user_indexes)
            logger.info("Created indexes for users collection")
        except Exception as e:
            logger.warning(f"Some user indexes may already exist: {e}")
        
        # Indexes for workflow_sessions collection
        workflow_sessions = db["workflow_sessions"]
        workflow_indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("current_step", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("updated_at", DESCENDING)]),
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)])
        ]
        
        try:
            await workflow_sessions.create_indexes(workflow_indexes)
            logger.info("Created indexes for workflow_sessions collection")
        except Exception as e:
            logger.warning(f"Some workflow session indexes may already exist: {e}")
        
        # Indexes for conversation_sessions collection
        conversation_sessions = db["conversation_sessions"]
        conversation_indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("state", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("updated_at", DESCENDING)])
        ]
        
        try:
            await conversation_sessions.create_indexes(conversation_indexes)
            logger.info("Created indexes for conversation_sessions collection")
        except Exception as e:
            logger.warning(f"Some conversation session indexes may already exist: {e}")
        
        # Indexes for recommendation_activities collection
        activities_collection = db["recommendation_activities"]
        activity_indexes = [
            IndexModel([("user_email", ASCENDING)]),
            IndexModel([("activity_type", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("user_email", ASCENDING), ("timestamp", DESCENDING)])
        ]
        
        try:
            await activities_collection.create_indexes(activity_indexes)
            logger.info("Created indexes for recommendation_activities collection")
        except Exception as e:
            logger.warning(f"Some activity indexes may already exist: {e}")
    
    async def _migration_006_migrate_user_data(self) -> None:
        """Migrate existing user data to new recommendation schema."""
        db = get_database()
        users_collection = db["users"]
        
        # Find users with old preference format and migrate them
        migration_count = 0
        
        async for user in users_collection.find({}):
            needs_migration = False
            update_data = {}
            
            # Check if user has old-style preferences that need migration
            old_preferences = user.get("preferences", {})
            if old_preferences and not user.get("recommendation_preferences"):
                # Migrate general preferences to recommendation preferences
                rec_prefs = {}
                
                # Look for budget information in old preferences
                if "budget" in old_preferences:
                    rec_prefs["budget"] = old_preferences["budget"]
                
                # Look for location preferences
                if "location" in old_preferences:
                    rec_prefs["location"] = old_preferences["location"]
                
                # Look for feature preferences
                if "features" in old_preferences:
                    rec_prefs["required_features"] = old_preferences["features"]
                
                if rec_prefs:
                    update_data["recommendation_preferences"] = rec_prefs
                    needs_migration = True
            
            # Ensure all required fields exist
            if "recommendation_history" not in user:
                update_data["recommendation_history"] = []
                needs_migration = True
            
            if "appointment_history" not in user:
                update_data["appointment_history"] = []
                needs_migration = True
            
            if "appointment_preferences" not in user:
                update_data["appointment_preferences"] = {}
                needs_migration = True
            
            # Apply migration if needed
            if needs_migration:
                await users_collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": update_data}
                )
                migration_count += 1
        
        logger.info(f"Migrated {migration_count} users to new recommendation schema")
    
    async def _migration_007_create_analytics(self) -> None:
        """Create recommendation_analytics collection for system metrics."""
        db = get_database()
        
        # Create collection if it doesn't exist
        collections = await db.list_collection_names()
        if "recommendation_analytics" not in collections:
            await db.create_collection("recommendation_analytics")
            logger.info("Created recommendation_analytics collection")
        
        # Create validation schema
        validation_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["metric_type", "value", "timestamp"],
                "properties": {
                    "metric_type": {"bsonType": "string"},
                    "value": {"bsonType": ["number", "object"]},
                    "timestamp": {"bsonType": "date"},
                    "metadata": {"bsonType": "object"},
                    "user_id": {"bsonType": ["string", "null"]},
                    "session_id": {"bsonType": ["string", "null"]}
                }
            }
        }
        
        try:
            analytics_collection = db["recommendation_analytics"]
            await db.command("collMod", "recommendation_analytics", validator=validation_schema)
            logger.info("Added validation schema to recommendation_analytics collection")
        except Exception as e:
            logger.warning(f"Could not add validation schema to recommendation_analytics: {e}")
        
        # Create indexes for analytics
        analytics_indexes = [
            IndexModel([("metric_type", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("metric_type", ASCENDING), ("timestamp", DESCENDING)])
        ]
        
        try:
            await analytics_collection.create_indexes(analytics_indexes)
            logger.info("Created indexes for recommendation_analytics collection")
        except Exception as e:
            logger.warning(f"Some analytics indexes may already exist: {e}")
    
    async def _is_migration_completed(self, migration_name: str) -> bool:
        """Check if a migration has already been completed."""
        db = get_database()
        
        # Create migrations collection if it doesn't exist
        collections = await db.list_collection_names()
        if "database_migrations" not in collections:
            await db.create_collection("database_migrations")
        
        migrations_collection = db["database_migrations"]
        migration_record = await migrations_collection.find_one({"migration_name": migration_name})
        
        return migration_record is not None
    
    async def _mark_migration_completed(self, migration_name: str) -> None:
        """Mark a migration as completed."""
        db = get_database()
        migrations_collection = db["database_migrations"]
        
        await migrations_collection.insert_one({
            "migration_name": migration_name,
            "completed_at": datetime.now(),
            "version": "1.0.0"
        })
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get the status of all migrations."""
        db = get_database()
        
        # Check if migrations collection exists
        collections = await db.list_collection_names()
        if "database_migrations" not in collections:
            return {
                "migrations_run": 0,
                "total_migrations": 7,
                "completed_migrations": [],
                "status": "not_started"
            }
        
        migrations_collection = db["database_migrations"]
        completed_migrations = []
        
        async for migration in migrations_collection.find({}):
            completed_migrations.append({
                "name": migration["migration_name"],
                "completed_at": migration["completed_at"],
                "version": migration.get("version", "unknown")
            })
        
        return {
            "migrations_run": len(completed_migrations),
            "total_migrations": 7,
            "completed_migrations": completed_migrations,
            "status": "completed" if len(completed_migrations) >= 7 else "partial"
        }
    
    async def rollback_migration(self, migration_name: str) -> Dict[str, Any]:
        """
        Rollback a specific migration (limited support).
        
        Args:
            migration_name: Name of the migration to rollback
            
        Returns:
            Rollback results
        """
        logger.warning(f"Attempting to rollback migration: {migration_name}")
        
        # Only support rollback for certain migrations
        if migration_name == "001_add_recommendation_fields_to_users":
            return await self._rollback_001_recommendation_fields()
        elif migration_name.startswith("002_create_") or migration_name.startswith("003_create_"):
            return {"error": "Cannot rollback collection creation migrations safely"}
        else:
            return {"error": f"Rollback not supported for migration: {migration_name}"}
    
    async def _rollback_001_recommendation_fields(self) -> Dict[str, Any]:
        """Rollback the recommendation fields migration."""
        db = get_database()
        users_collection = db["users"]
        
        # Remove recommendation fields from users
        update_result = await users_collection.update_many(
            {},
            {"$unset": {
                "recommendation_preferences": "",
                "last_recommendation_date": "",
                "recommendation_history": "",
                "appointment_history": "",
                "appointment_preferences": ""
            }}
        )
        
        # Remove migration record
        migrations_collection = db["database_migrations"]
        await migrations_collection.delete_one({"migration_name": "001_add_recommendation_fields_to_users"})
        
        return {
            "success": True,
            "users_updated": update_result.modified_count,
            "migration_record_removed": True
        }


# Global migration manager instance
migration_manager = DatabaseMigrationManager() 