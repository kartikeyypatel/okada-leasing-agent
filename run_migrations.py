#!/usr/bin/env python3
"""
Database Migration Runner

Simple script to run database migrations for the Smart Property Recommendations system.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database_migrations import migration_manager
from app.database import connect_to_mongo, close_mongo_connection


async def run_migrations():
    """Run all database migrations."""
    print("🚀 Starting database migrations for Smart Property Recommendations...")
    
    try:
        results = await migration_manager.run_all_migrations()
        
        print("\n📊 Migration Results:")
        print(f"   Total migrations: {results['total_migrations']}")
        print(f"   Successful migrations: {results['successful_migrations']}")
        print(f"   Migrations run: {len(results['migrations_run'])}")
        print(f"   Migrations skipped: {len(results['migrations_skipped'])}")
        
        if results['migrations_run']:
            print(f"\n✅ Successfully executed migrations:")
            for migration in results['migrations_run']:
                print(f"   - {migration}")
        
        if results['migrations_skipped']:
            print(f"\n⏭️  Skipped migrations (already completed):")
            for migration in results['migrations_skipped']:
                print(f"   - {migration}")
        
        if results['errors']:
            print(f"\n❌ Errors encountered:")
            for error in results['errors']:
                print(f"   - {error}")
            return False
        
        print(f"\n🎉 Database migrations completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n💥 Migration failed with error: {e}")
        return False


async def check_migration_status():
    """Check the current migration status."""
    print("📋 Checking migration status...")
    
    try:
        status = await migration_manager.get_migration_status()
        
        print(f"\n📈 Migration Status:")
        print(f"   Migrations completed: {status['migrations_run']}/{status['total_migrations']}")
        print(f"   Overall status: {status['status']}")
        
        if status['completed_migrations']:
            print(f"\n✅ Completed migrations:")
            for migration in status['completed_migrations']:
                print(f"   - {migration['name']} (completed: {migration['completed_at']})")
        
        return status['status'] == 'completed'
        
    except Exception as e:
        print(f"\n💥 Status check failed: {e}")
        return False


async def main():
    """Main function."""
    print("🏠 Smart Property Recommendations - Database Migration Tool\n")
    
    # Establish database connection
    print("🔌 Connecting to MongoDB...")
    try:
        await connect_to_mongo()
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("   Please ensure MongoDB is running and connection settings are correct.")
        sys.exit(1)
    
    try:
        # Check current status
        await check_migration_status()
        
        print("\n" + "="*60)
        
        # Run migrations
        success = await run_migrations()
        
        if success:
            print("\n🎯 All migrations completed successfully!")
            print("   The Smart Property Recommendations system is ready to use.")
        else:
            print("\n⚠️  Some migrations failed. Please check the errors above.")
            return False
            
    finally:
        # Close database connection
        print("\n🔌 Closing database connection...")
        try:
            await close_mongo_connection()
            print("✅ Database connection closed")
        except Exception as e:
            print(f"⚠️  Error closing database connection: {e}")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 