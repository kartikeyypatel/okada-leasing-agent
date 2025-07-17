#!/usr/bin/env python3
"""
Script to rebuild user index from CSV files
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def rebuild_index():
    """Rebuild the user index"""
    try:
        from app.rag import build_user_index, clear_user_index
        
        user_id = "ok@gmail.com"
        user_docs_path = f"user_documents/{user_id}"
        
        print(f"🔧 Rebuilding index for user: {user_id}")
        
        # Check if user documents exist
        if not os.path.exists(user_docs_path):
            print(f"❌ User documents directory not found: {user_docs_path}")
            return False
        
        # Find CSV files
        csv_files = [f for f in os.listdir(user_docs_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"❌ No CSV files found in {user_docs_path}")
            return False
        
        file_paths = [os.path.join(user_docs_path, f) for f in csv_files]
        print(f"📄 Found {len(csv_files)} CSV files: {csv_files}")
        
        # Clear existing index
        print("🧹 Clearing existing index...")
        await clear_user_index(user_id)
        
        # Build new index
        print("🏗️  Building new index...")
        new_index = await build_user_index(user_id, file_paths)
        
        if new_index:
            print("✅ Index rebuilt successfully!")
            
            # Test the index
            print("🧪 Testing the new index...")
            if hasattr(new_index.docstore, 'docs'):
                doc_count = len(new_index.docstore.docs)
                print(f"   📊 Index contains {doc_count} documents")
                
                # Show sample document
                if doc_count > 0:
                    sample_doc = list(new_index.docstore.docs.values())[0]
                    print(f"   📝 Sample document: {sample_doc.text[:200]}...")
            
            return True
        else:
            print("❌ Failed to build index")
            return False
            
    except Exception as e:
        print(f"❌ Error rebuilding index: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(rebuild_index())
    sys.exit(0 if success else 1)