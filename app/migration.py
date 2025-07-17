"""
Data Migration Utilities for ChromaDB Integration

This module provides utilities to migrate existing user documents from in-memory
storage to ChromaDB persistent storage, with validation and rollback capabilities.
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

import app.rag as rag_module
from app.chroma_client import chroma_manager
from app.database import get_database
from app.config import settings

logger = logging.getLogger(__name__)

class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass

class DataMigrationManager:
    """Manages data migration from in-memory to ChromaDB storage."""
    
    def __init__(self):
        self.migration_log: List[Dict] = []
        self.backup_data: Dict = {}
        
    async def discover_existing_documents(self) -> Dict[str, List[str]]:
        """
        Discover existing user documents in the user_documents directory.
        
        Returns:
            Dict mapping user_id to list of their document files
        """
        user_documents = {}
        user_docs_path = Path("user_documents")
        
        if not user_docs_path.exists():
            logger.info("No user_documents directory found")
            return user_documents
            
        for user_dir in user_docs_path.iterdir():
            if user_dir.is_dir():
                user_id = user_dir.name
                csv_files = [f.name for f in user_dir.glob("*.csv")]
                if csv_files:
                    user_documents[user_id] = csv_files
                    logger.info(f"Found {len(csv_files)} documents for user {user_id}")
        
        return user_documents
    
    async def validate_chromadb_connection(self) -> bool:
        """Validate ChromaDB connection before migration."""
        try:
            client = await asyncio.to_thread(chroma_manager.get_client)
            await asyncio.to_thread(client.list_collections)
            logger.info("ChromaDB connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"ChromaDB connection validation failed: {e}")
            return False
    
    async def backup_existing_collections(self, user_ids: List[str]) -> bool:
        """
        Backup existing ChromaDB collections for rollback purposes.
        
        Args:
            user_ids: List of user IDs to backup
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            backup_timestamp = datetime.now().isoformat()
            self.backup_data = {
                "timestamp": backup_timestamp,
                "collections": {}
            }
            
            client = await asyncio.to_thread(chroma_manager.get_client)
            
            for user_id in user_ids:
                try:
                    collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
                    # Get all documents from the collection
                    result = await asyncio.to_thread(collection.get)
                    
                    self.backup_data["collections"][user_id] = {
                        "documents": result.get("documents", []),
                        "metadatas": result.get("metadatas", []),
                        "ids": result.get("ids", [])
                    }
                    logger.info(f"Backed up collection for user {user_id}")
                except Exception as e:
                    logger.warning(f"Could not backup collection for user {user_id}: {e}")
            
            # Save backup to file
            backup_file = f"migration_backup_{backup_timestamp.replace(':', '-')}.json"
            with open(backup_file, 'w') as f:
                json.dump(self.backup_data, f, indent=2)
            
            logger.info(f"Backup saved to {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    async def migrate_user_documents(self, user_id: str, file_paths: List[str]) -> bool:
        """
        Migrate documents for a specific user to ChromaDB.
        
        Args:
            user_id: User identifier
            file_paths: List of full paths to user's CSV files
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info(f"Starting migration for user {user_id} with {len(file_paths)} files")
            
            # Clear existing index for clean migration
            await rag_module.clear_user_index(user_id)
            
            # Build new ChromaDB index
            index = await rag_module.build_user_index(user_id, file_paths)
            
            if index:
                migration_entry = {
                    "user_id": user_id,
                    "files": file_paths,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
                self.migration_log.append(migration_entry)
                logger.info(f"Successfully migrated user {user_id}")
                return True
            else:
                migration_entry = {
                    "user_id": user_id,
                    "files": file_paths,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": "Index creation failed"
                }
                self.migration_log.append(migration_entry)
                logger.error(f"Failed to migrate user {user_id}: Index creation failed")
                return False
                
        except Exception as e:
            migration_entry = {
                "user_id": user_id,
                "files": file_paths,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }
            self.migration_log.append(migration_entry)
            logger.error(f"Failed to migrate user {user_id}: {e}")
            return False
    
    async def validate_migration(self, user_id: str) -> bool:
        """
        Validate that migration was successful for a user.
        
        Args:
            user_id: User identifier to validate
            
        Returns:
            True if validation successful, False otherwise
        """
        try:
            # Check if user index exists and is functional
            index = await rag_module.get_user_index(user_id)
            if not index:
                logger.error(f"Validation failed: No index found for user {user_id}")
                return False
            
            # Try to perform a simple search
            retriever = rag_module.get_fusion_retriever(user_id)
            if not retriever:
                logger.error(f"Validation failed: No retriever available for user {user_id}")
                return False
            
            # Test search functionality
            test_results = await retriever.aretrieve("test query")
            logger.info(f"Validation successful for user {user_id}: {len(test_results)} results available")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for user {user_id}: {e}")
            return False
    
    async def rollback_migration(self, user_id: str) -> bool:
        """
        Rollback migration for a specific user.
        
        Args:
            user_id: User identifier to rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            if user_id not in self.backup_data.get("collections", {}):
                logger.error(f"No backup data found for user {user_id}")
                return False
            
            # Clear current collection
            await rag_module.clear_user_index(user_id)
            
            # Restore from backup
            backup_collection_data = self.backup_data["collections"][user_id]
            
            if backup_collection_data["documents"]:
                collection = await asyncio.to_thread(chroma_manager.get_or_create_collection, user_id)
                
                # Add documents back to collection
                await asyncio.to_thread(
                    collection.add,
                    documents=backup_collection_data["documents"],
                    metadatas=backup_collection_data["metadatas"],
                    ids=backup_collection_data["ids"]
                )
                
                logger.info(f"Rollback successful for user {user_id}")
                return True
            else:
                logger.info(f"No data to rollback for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Rollback failed for user {user_id}: {e}")
            return False
    
    async def migrate_all_users(self, validate: bool = True) -> Dict[str, str]:
        """
        Migrate all existing user documents to ChromaDB.
        
        Args:
            validate: Whether to validate migration success
            
        Returns:
            Dict mapping user_id to migration status
        """
        logger.info("Starting full migration to ChromaDB")
        
        # Validate ChromaDB connection
        if not await self.validate_chromadb_connection():
            raise MigrationError("ChromaDB connection validation failed")
        
        # Discover existing documents
        user_documents = await self.discover_existing_documents()
        if not user_documents:
            logger.info("No existing documents found to migrate")
            return {}
        
        # Backup existing collections
        user_ids = list(user_documents.keys())
        if not await self.backup_existing_collections(user_ids):
            raise MigrationError("Failed to create backup")
        
        # Migrate each user
        migration_results = {}
        
        for user_id, filenames in user_documents.items():
            # Create full file paths
            file_paths = [
                os.path.join("user_documents", user_id, filename)
                for filename in filenames
            ]
            
            # Migrate user documents
            success = await self.migrate_user_documents(user_id, file_paths)
            
            if success and validate:
                # Validate migration
                validation_success = await self.validate_migration(user_id)
                if not validation_success:
                    logger.warning(f"Migration validation failed for user {user_id}")
                    success = False
            
            migration_results[user_id] = "success" if success else "failed"
            
            # Add delay between migrations to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        # Save migration log
        log_file = f"migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
        
        logger.info(f"Migration completed. Log saved to {log_file}")
        logger.info(f"Results: {migration_results}")
        
        return migration_results
    
    def get_migration_summary(self) -> Dict:
        """Get a summary of the migration process."""
        total = len(self.migration_log)
        successful = sum(1 for entry in self.migration_log if entry["status"] == "success")
        failed = total - successful
        
        return {
            "total_migrations": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/total*100):.1f}%" if total > 0 else "0%",
            "migration_log": self.migration_log
        }


# CLI-style functions for easy usage
async def migrate_to_chromadb(validate: bool = True) -> Dict[str, str]:
    """
    Convenience function to migrate all user documents to ChromaDB.
    
    Args:
        validate: Whether to validate migration success
        
    Returns:
        Dict mapping user_id to migration status
    """
    manager = DataMigrationManager()
    return await manager.migrate_all_users(validate=validate)


async def validate_chromadb_migration() -> bool:
    """
    Validate that ChromaDB migration was successful.
    
    Returns:
        True if validation successful, False otherwise
    """
    manager = DataMigrationManager()
    return await manager.validate_chromadb_connection()


if __name__ == "__main__":
    # Example usage
    async def main():
        try:
            results = await migrate_to_chromadb(validate=True)
            print("Migration Results:")
            for user_id, status in results.items():
                print(f"  {user_id}: {status}")
        except Exception as e:
            print(f"Migration failed: {e}")
    
    asyncio.run(main()) 