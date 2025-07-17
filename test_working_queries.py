#!/usr/bin/env python3
"""
Test the RAG system with queries that should work with the actual data
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_working_queries():
    """Test with queries that match the actual data"""
    try:
        from app.rag import get_user_index, get_fusion_retriever
        from app.strict_response_generator import StrictResponseGenerator
        from app.multi_strategy_search import multi_strategy_search
        
        print("🔍 Testing RAG system with working queries...")
        
        user_id = "ok@gmail.com"
        
        # Get the index
        user_index = await get_user_index(user_id)
        if not user_index:
            print("❌ No user index found")
            return False
        
        print("✅ User index found")
        
        # Get fusion retriever
        fusion_retriever = get_fusion_retriever(user_id)
        if not fusion_retriever:
            print("❌ Failed to create fusion retriever")
            return False
        
        print("✅ Fusion retriever created")
        
        # Test queries that should work with the actual data
        test_queries = [
            "tell me about 36 W 36th St",  # This address exists in the data
            "what properties are available on 15 W 38th St",  # Multiple properties at this address
            "show me properties with rent under $100 per square foot",  # Should find several
            "what is the monthly rent for properties on 15 W 38th St",  # Should find multiple
            "give me the cheapest properties by rent per square foot"  # Should work with the data
        ]
        
        generator = StrictResponseGenerator()
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            try:
                # Search for documents
                search_result = await multi_strategy_search(fusion_retriever, query)
                nodes_found = search_result.nodes_found
                
                print(f"   📄 Found {len(nodes_found)} documents")
                
                if len(nodes_found) > 0:
                    # Show sample of what was found
                    sample_text = nodes_found[0].text[:150] + "..." if len(nodes_found[0].text) > 150 else nodes_found[0].text
                    print(f"   📝 Sample result: {sample_text}")
                
                # Generate response
                strict_result = await generator.generate_strict_response(
                    user_query=query,
                    retrieved_nodes=nodes_found,
                    user_id=user_id
                )
                
                print(f"   ✅ Response generated: {strict_result.generation_successful}")
                print(f"   📋 Context valid: {strict_result.context_validation.is_valid}")
                print(f"   🎯 Quality valid: {strict_result.quality_validation.is_valid}")
                
                # Show the response
                response_preview = strict_result.response_text[:300] + "..." if len(strict_result.response_text) > 300 else strict_result.response_text
                print(f"   💬 Response: {response_preview}")
                
                if strict_result.quality_validation.quality_issues:
                    print(f"   ⚠️  Quality issues: {strict_result.quality_validation.quality_issues}")
                
            except Exception as e:
                print(f"   ❌ Error testing query '{query}': {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 Working queries test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_working_queries())
    sys.exit(0 if success else 1)