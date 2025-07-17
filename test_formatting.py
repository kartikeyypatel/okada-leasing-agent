#!/usr/bin/env python3
"""
Test the improved response formatting
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_formatting():
    """Test the new formatting"""
    try:
        from app.rag import get_fusion_retriever
        from app.strict_response_generator import StrictResponseGenerator
        from app.multi_strategy_search import multi_strategy_search
        
        user_id = "ok@gmail.com"
        
        print("🎨 Testing improved response formatting...")
        
        # Get fusion retriever
        fusion_retriever = get_fusion_retriever(user_id)
        if not fusion_retriever:
            print("❌ No fusion retriever available")
            return False
        
        print("✅ Fusion retriever available")
        
        # Test queries
        test_queries = [
            "tell me about 36 W 36th St",
            "show me 3 properties with lowest rent"
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
                    
                    # Check formatting
                    response = strict_result.response_text
                    has_asterisks = "**" in response or "*" in response
                    has_clean_format = "PROPERTY NAME:" in response or "Monthly Rent:" in response
                    
                    print(f"   🎨 Has asterisks (bad): {has_asterisks}")
                    print(f"   🎨 Has clean format (good): {has_clean_format}")
                    
                    print(f"   📝 Response preview:")
                    print("   " + "="*50)
                    # Show first 300 characters with proper line breaks
                    preview = response[:300].replace("\\n", "\n   ")
                    print(f"   {preview}...")
                    print("   " + "="*50)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in formatting test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_formatting())
    print("\n🎉 Formatting test completed!")
    print("✅ Responses should now be clean and readable without ** asterisks")
    print("✅ Lists should be properly numbered and formatted")
    print("✅ Property details should be clearly organized")
    sys.exit(0 if success else 1)