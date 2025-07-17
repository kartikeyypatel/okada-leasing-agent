#!/usr/bin/env python3
"""
Debug script to check BM25 retriever creation
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def debug_bm25():
    """Debug BM25 retriever creation"""
    try:
        from app.rag import get_user_index, user_bm25_retrievers, user_indexes
        from llama_index.retrievers.bm25 import BM25Retriever
        
        user_id = "ok@gmail.com"
        
        print(f"🔍 Debugging BM25 retriever for user: {user_id}")
        
        # Get the index
        user_index = await get_user_index(user_id)
        if not user_index:
            print("❌ No user index found")
            return False
        
        print("✅ User index found")
        
        # Check if BM25 retriever exists
        if user_id in user_bm25_retrievers:
            print("✅ BM25 retriever already exists")
            bm25_retriever = user_bm25_retrievers[user_id]
            print(f"   📊 BM25 retriever type: {type(bm25_retriever)}")
            return True
        else:
            print("❌ BM25 retriever not found, attempting to create...")
        
        # Try to create BM25 retriever manually
        try:
            # Get nodes from the index
            if hasattr(user_index.docstore, 'docs'):
                nodes = list(user_index.docstore.docs.values())
                print(f"   📄 Found {len(nodes)} nodes in docstore")
                
                if len(nodes) > 0:
                    # Show sample node
                    sample_node = nodes[0]
                    print(f"   📝 Sample node type: {type(sample_node)}")
                    print(f"   📝 Sample node text: {sample_node.text[:100]}...")
                    
                    # Try to create BM25 retriever
                    bm25_retriever = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=5
                    )
                    
                    # Cache it
                    user_bm25_retrievers[user_id] = bm25_retriever
                    
                    print("✅ BM25 retriever created successfully!")
                    
                    # Test it
                    test_results = bm25_retriever.retrieve("36 W 36th St")
                    print(f"   🧪 Test search returned {len(test_results)} results")
                    
                    return True
                else:
                    print("❌ No nodes found in docstore")
                    return False
            else:
                print("❌ Index docstore has no 'docs' attribute")
                return False
                
        except Exception as e:
            print(f"❌ Error creating BM25 retriever: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_bm25())
    sys.exit(0 if success else 1)