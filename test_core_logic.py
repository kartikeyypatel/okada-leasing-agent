#!/usr/bin/env python3
"""
Test the core RAG logic directly
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_core_logic():
    """Test the core RAG logic with real queries"""
    try:
        from app.rag import get_user_index, get_fusion_retriever, user_bm25_retrievers
        from app.strict_response_generator import StrictResponseGenerator
        from app.multi_strategy_search import multi_strategy_search
        
        user_id = "ok@gmail.com"
        
        print("🔍 Testing core RAG logic...")
        
        # Check current state
        print(f"📊 Current BM25 retrievers: {list(user_bm25_retrievers.keys())}")
        
        # Get fusion retriever
        fusion_retriever = get_fusion_retriever(user_id)
        if not fusion_retriever:
            print("❌ No fusion retriever available")
            return False
        
        print("✅ Fusion retriever available")
        
        # Test queries that should work with the actual data
        test_queries = [
            "tell me about 36 W 36th St",
            "what properties are available on 15 W 38th St", 
            "show me properties with rent under $90 per square foot",
            "give me the cheapest properties"
        ]
        
        generator = StrictResponseGenerator()
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            
            try:
                # Search
                search_result = await multi_strategy_search(fusion_retriever, query)
                nodes_found = search_result.nodes_found
                
                print(f"   📄 Found {len(nodes_found)} documents")
                
                if len(nodes_found) > 0:
                    # Generate response
                    strict_result = await generator.generate_strict_response(
                        user_query=query,
                        retrieved_nodes=nodes_found,
                        user_id=user_id
                    )
                    
                    print(f"   ✅ Generation successful: {strict_result.generation_successful}")
                    print(f"   📝 Response preview: {strict_result.response_text[:150]}...")
                else:
                    print("   ⚠️  No documents found")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in core logic test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_core_logic())
    sys.exit(0 if success else 1)