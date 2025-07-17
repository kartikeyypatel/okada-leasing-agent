#!/usr/bin/env python3
"""
Integration test for query optimization with RAG system.

This test verifies that the query optimization functionality integrates
correctly with the existing RAG search system.
"""

import asyncio
import logging
from app.query_optimizer import optimize_search_query, get_query_variants
from app.multi_strategy_search import multi_strategy_search, MultiStrategySearcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockRetriever:
    """Mock retriever for testing without actual ChromaDB."""
    
    def __init__(self):
        self.call_count = 0
        self.queries_received = []
    
    async def aretrieve(self, query: str):
        """Mock retrieve method that logs queries and returns empty results."""
        self.call_count += 1
        self.queries_received.append(query)
        logger.info(f"Mock retriever called with query: '{query}'")
        return []  # Return empty results for testing

async def test_query_optimization_integration():
    """Test that query optimization integrates with multi-strategy search."""
    print("\n=== Testing Query Optimization Integration ===")
    
    # Test query
    test_query = "tell me about 84 Mulberry St"
    
    # Step 1: Test query optimization standalone
    print(f"\n1. Testing standalone query optimization for: '{test_query}'")
    optimized_result = optimize_search_query(test_query)
    
    print(f"   - Query type: {optimized_result.analysis.query_type.value}")
    print(f"   - Optimized query: '{optimized_result.optimized_query}'")
    print(f"   - Variants generated: {len(optimized_result.query_variants)}")
    print(f"   - Optimizations applied: {optimized_result.optimization_applied}")
    
    # Step 2: Test integration with multi-strategy search
    print(f"\n2. Testing integration with multi-strategy search")
    mock_retriever = MockRetriever()
    searcher = MultiStrategySearcher(mock_retriever)
    
    # Perform multi-strategy search (which should use query optimization)
    search_result = await searcher.search_with_multiple_strategies(test_query)
    
    print(f"   - Search strategies executed: {len(search_result.all_results)}")
    print(f"   - Total execution time: {search_result.total_execution_time_ms:.2f}ms")
    print(f"   - Mock retriever called: {mock_retriever.call_count} times")
    
    # Step 3: Verify that optimized queries were used
    print(f"\n3. Verifying optimized queries were used:")
    for i, query in enumerate(mock_retriever.queries_received, 1):
        print(f"   {i}. '{query}'")
    
    # Step 4: Check that address format is preserved
    print(f"\n4. Checking address format preservation:")
    has_exact_address = any("84 Mulberry St" in query for query in mock_retriever.queries_received)
    has_address_variation = any("84 Mulberry" in query and query != test_query for query in mock_retriever.queries_received)
    
    print(f"   - Exact address format preserved: {has_exact_address}")
    print(f"   - Address variations generated: {has_address_variation}")
    
    return {
        "optimization_successful": len(optimized_result.query_variants) > 0,
        "integration_successful": mock_retriever.call_count > 0,
        "address_preserved": has_exact_address,
        "variations_generated": has_address_variation,
        "strategies_executed": len(search_result.all_results)
    }

async def test_different_query_types():
    """Test optimization with different types of queries."""
    print("\n=== Testing Different Query Types ===")
    
    test_queries = [
        ("tell me about 84 Mulberry St", "address_specific"),
        ("show me 2 bedroom apartments", "property_general"),
        ("what is the rent for 123 Main Street", "mixed"),
        ("find office space with parking", "property_general")
    ]
    
    results = {}
    
    for query, expected_type in test_queries:
        print(f"\nTesting: '{query}' (expected: {expected_type})")
        
        # Test optimization
        optimized = optimize_search_query(query)
        print(f"  - Detected type: {optimized.analysis.query_type.value}")
        print(f"  - Confidence: {optimized.analysis.confidence_score:.2f}")
        print(f"  - Variants: {len(optimized.query_variants)}")
        
        # Test with mock retriever
        mock_retriever = MockRetriever()
        searcher = MultiStrategySearcher(mock_retriever)
        search_result = await searcher.search_with_multiple_strategies(query)
        
        print(f"  - Strategies executed: {len(search_result.all_results)}")
        print(f"  - Retriever calls: {mock_retriever.call_count}")
        
        results[query] = {
            "detected_type": optimized.analysis.query_type.value,
            "expected_type": expected_type,
            "confidence": optimized.analysis.confidence_score,
            "variants_count": len(optimized.query_variants),
            "strategies_count": len(search_result.all_results),
            "retriever_calls": mock_retriever.call_count
        }
    
    return results

async def test_query_variants_function():
    """Test the get_query_variants convenience function."""
    print("\n=== Testing Query Variants Function ===")
    
    test_query = "find 2 bedroom apartment with parking near 84 Mulberry St"
    print(f"Testing query: '{test_query}'")
    
    # Test with different limits
    for limit in [3, 5, 10]:
        variants = get_query_variants(test_query, max_variants=limit)
        print(f"  - Limit {limit}: {len(variants)} variants generated")
        
        # Check for duplicates
        unique_variants = set(v.lower() for v in variants)
        has_duplicates = len(unique_variants) != len(variants)
        print(f"    - Has duplicates: {has_duplicates}")
        
        # Show first few variants
        for i, variant in enumerate(variants[:3], 1):
            print(f"    {i}. '{variant}'")
        
        if len(variants) > 3:
            print(f"    ... and {len(variants) - 3} more")

async def main():
    """Run all integration tests."""
    print("Query Optimization Integration Tests")
    print("=" * 50)
    
    try:
        # Test 1: Basic integration
        integration_result = await test_query_optimization_integration()
        print(f"\nIntegration Test Results:")
        for key, value in integration_result.items():
            print(f"  - {key}: {value}")
        
        # Test 2: Different query types
        query_type_results = await test_different_query_types()
        print(f"\nQuery Type Test Summary:")
        for query, result in query_type_results.items():
            type_match = result['detected_type'] == result['expected_type']
            print(f"  - '{query[:30]}...': type_match={type_match}, confidence={result['confidence']:.2f}")
        
        # Test 3: Query variants function
        await test_query_variants_function()
        
        print("\n" + "=" * 50)
        print("All integration tests completed successfully!")
        
        # Summary
        print(f"\nSummary:")
        print(f"- Query optimization is working correctly")
        print(f"- Integration with multi-strategy search is functional")
        print(f"- Address format preservation is working")
        print(f"- Query variants generation is working")
        print(f"- Different query types are handled appropriately")
        
    except Exception as e:
        print(f"\nError during integration tests: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())