# Simple test for multi-strategy search functionality
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.multi_strategy_search import (
    AddressExtractor,
    QueryProcessor,
    SearchResultRanker
)

def test_address_extractor():
    """Test the AddressExtractor functionality."""
    print("Testing AddressExtractor...")
    
    # Test 1: Extract full addresses
    text = "tell me about 84 Mulberry St and 123 Main Street"
    addresses = AddressExtractor.extract_addresses(text)
    print(f"  Input: '{text}'")
    print(f"  Extracted addresses: {addresses}")
    assert "84 Mulberry St" in addresses or "84 Mulberry" in addresses
    
    # Test 2: Simple pattern
    text2 = "what about 84 Mulberry"
    addresses2 = AddressExtractor.extract_addresses(text2)
    print(f"  Input: '{text2}'")
    print(f"  Extracted addresses: {addresses2}")
    assert len(addresses2) > 0
    
    # Test 3: Normalization
    address = "  84   Mulberry   St  "
    normalized = AddressExtractor.normalize_address(address)
    print(f"  Original: '{address}'")
    print(f"  Normalized: '{normalized}'")
    assert normalized == "84 Mulberry St"
    
    print("  ✅ AddressExtractor tests passed!")

def test_query_processor():
    """Test the QueryProcessor functionality."""
    print("Testing QueryProcessor...")
    
    # Test 1: Extract key terms
    query = "tell me about the property with rent and size"
    terms = QueryProcessor.extract_key_terms(query)
    print(f"  Input: '{query}'")
    print(f"  Key terms: {terms}")
    assert "property" in terms
    assert "rent" in terms
    
    # Test 2: Create fuzzy query
    query2 = "tell me about 84 Mulberry St with good rent"
    fuzzy = QueryProcessor.create_fuzzy_query(query2)
    print(f"  Input: '{query2}'")
    print(f"  Fuzzy query: '{fuzzy}'")
    assert len(fuzzy) > 0
    
    print("  ✅ QueryProcessor tests passed!")

def main():
    """Run all tests."""
    print("Running Multi-Strategy Search Tests...")
    print("=" * 50)
    
    try:
        test_address_extractor()
        print()
        test_query_processor()
        print()
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)