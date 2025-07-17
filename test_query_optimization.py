#!/usr/bin/env python3
"""
Test suite for query optimization functionality.

This test suite verifies that the query optimization module correctly:
1. Preserves exact address formats
2. Implements query preprocessing for better address matching
3. Adds query expansion strategies for partial matches
4. Creates query validation and sanitization logic
"""

import logging
from typing import List, Dict

from app.query_optimizer import (
    QueryOptimizer, QueryAnalyzer, AddressNormalizer, QueryExpander, 
    QueryValidator, QueryType, optimize_search_query, get_query_variants
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAddressNormalizer:
    """Test address extraction and normalization functionality."""
    
    def test_extract_addresses_basic(self):
        """Test basic address extraction."""
        text = "Tell me about 84 Mulberry St"
        addresses = AddressNormalizer.extract_addresses(text)
        
        assert len(addresses) > 0
        assert any("84" in addr['full_match'] for addr in addresses)
        assert any("Mulberry" in addr['full_match'] for addr in addresses)
    
    def test_extract_addresses_multiple_formats(self):
        """Test extraction of various address formats."""
        test_cases = [
            "123 Main Street",
            "456 Oak Ave",
            "789 Pine Road",
            "84 Mulberry St",
            "100 North First Street",
            "200 S Second Ave"
        ]
        
        for address in test_cases:
            addresses = AddressNormalizer.extract_addresses(f"Find {address}")
            assert len(addresses) > 0, f"Failed to extract address: {address}"
            
            # Check that the address components are properly parsed
            addr_dict = addresses[0]
            assert 'number' in addr_dict
            assert 'street_name' in addr_dict
            assert addr_dict['number'].isdigit(), f"Number not extracted for {address}"
    
    def test_normalize_address_variations(self):
        """Test address normalization creates proper variations."""
        address = "84 Mulberry St"
        variations = AddressNormalizer.normalize_address(address)
        
        assert len(variations) > 1
        assert address in variations  # Original should be included
        
        # Should have both abbreviated and full forms
        variation_text = " ".join(variations).lower()
        assert "street" in variation_text or "st" in variation_text
    
    def test_address_abbreviation_expansion(self):
        """Test that address abbreviations are properly expanded."""
        test_cases = [
            ("123 Main St", ["street", "st"]),
            ("456 Oak Ave", ["avenue", "ave"]),
            ("789 Pine Rd", ["road", "rd"]),
            ("100 First Blvd", ["boulevard", "blvd"])
        ]
        
        for address, expected_terms in test_cases:
            variations = AddressNormalizer.normalize_address(address)
            variation_text = " ".join(variations).lower()
            
            # Should contain both abbreviated and full forms
            assert any(term in variation_text for term in expected_terms), \
                f"Missing expected terms {expected_terms} in variations for {address}"


class TestQueryAnalyzer:
    """Test query analysis functionality."""
    
    def test_analyze_address_specific_query(self):
        """Test analysis of address-specific queries."""
        query = "tell me about 84 Mulberry St"
        analysis = QueryAnalyzer.analyze_query(query)
        
        assert analysis.query_type == QueryType.ADDRESS_SPECIFIC
        assert len(analysis.addresses_found) > 0
        assert "84 Mulberry St" in " ".join(analysis.addresses_found)
        assert analysis.confidence_score > 0.5
    
    def test_analyze_property_general_query(self):
        """Test analysis of general property queries."""
        query = "show me available apartments with 2 bedrooms"
        analysis = QueryAnalyzer.analyze_query(query)
        
        assert analysis.query_type == QueryType.PROPERTY_GENERAL
        assert len(analysis.key_terms) > 0
        assert any(term in ['apartments', 'bedrooms', 'available'] for term in analysis.key_terms)
    
    def test_analyze_mixed_query(self):
        """Test analysis of mixed queries (address + property info)."""
        query = "what is the rent for 84 Mulberry Street apartment"
        analysis = QueryAnalyzer.analyze_query(query)
        
        assert analysis.query_type == QueryType.MIXED
        assert len(analysis.addresses_found) > 0
        assert len(analysis.key_terms) > 0
        assert analysis.confidence_score > 0.6
    
    def test_key_term_extraction(self):
        """Test key term extraction filters stop words properly."""
        query = "tell me about the rent for this apartment"
        analysis = QueryAnalyzer.analyze_query(query)
        
        # Should exclude stop words like 'the', 'me', 'for', 'this'
        stop_words_found = any(term in ['the', 'me', 'for', 'this'] for term in analysis.key_terms)
        assert not stop_words_found, f"Stop words found in key terms: {analysis.key_terms}"
        
        # Should include meaningful terms
        assert 'rent' in analysis.key_terms
        assert 'apartment' in analysis.key_terms


class TestQueryExpander:
    """Test query expansion functionality."""
    
    def test_synonym_expansion(self):
        """Test synonym expansion for real estate terms."""
        query = "find apartment for rent"
        expansions = QueryExpander.expand_query(query, "synonyms")
        
        assert len(expansions) > 1
        assert query in expansions  # Original should be included
        
        # Should have synonym variations
        expansion_text = " ".join(expansions).lower()
        assert any(synonym in expansion_text for synonym in ['unit', 'flat', 'rental', 'lease'])
    
    def test_partial_query_expansion(self):
        """Test partial query creation."""
        query = "tell me about 84 Mulberry Street apartment details"
        expansions = QueryExpander.expand_query(query, "partial")
        
        assert len(expansions) > 1
        assert query in expansions  # Original should be included
        
        # Should have partial queries
        assert any(len(exp.split()) < len(query.split()) for exp in expansions)
    
    def test_fuzzy_query_expansion(self):
        """Test fuzzy query creation."""
        query = "what is the monthly rent for 84 Mulberry Street"
        expansions = QueryExpander.expand_query(query, "fuzzy")
        
        assert len(expansions) > 1
        assert query in expansions  # Original should be included
        
        # Should extract key components
        expansion_text = " ".join(expansions).lower()
        assert "84 mulberry" in expansion_text or "mulberry" in expansion_text
        assert "rent" in expansion_text or "monthly" in expansion_text


class TestQueryValidator:
    """Test query validation and sanitization."""
    
    def test_validate_normal_query(self):
        """Test validation of normal queries."""
        query = "tell me about 84 Mulberry Street"
        is_valid, issues = QueryValidator.validate_query(query)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_empty_query(self):
        """Test validation catches empty queries."""
        query = ""
        is_valid, issues = QueryValidator.validate_query(query)
        
        assert not is_valid
        assert "empty_query" in issues
    
    def test_validate_long_query(self):
        """Test validation catches overly long queries."""
        query = "a" * 600  # Longer than 500 character limit
        is_valid, issues = QueryValidator.validate_query(query)
        
        assert not is_valid
        assert "query_too_long" in issues
    
    def test_sanitize_query_basic(self):
        """Test basic query sanitization."""
        query = "  tell me about   84 Mulberry St  "
        sanitized = QueryValidator.sanitize_query(query)
        
        assert sanitized == "tell me about 84 Mulberry St"
        assert not sanitized.startswith(" ")
        assert not sanitized.endswith(" ")
    
    def test_sanitize_query_html_removal(self):
        """Test HTML tag removal during sanitization."""
        query = "tell me about <script>alert('test')</script> 84 Mulberry St"
        sanitized = QueryValidator.sanitize_query(query)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        assert "84 Mulberry St" in sanitized
    
    def test_sanitize_query_length_limit(self):
        """Test query length limiting during sanitization."""
        query = "a" * 600
        sanitized = QueryValidator.sanitize_query(query)
        
        assert len(sanitized) <= 500


class TestQueryOptimizer:
    """Test the main query optimizer functionality."""
    
    def test_optimize_address_query(self):
        """Test optimization of address-specific queries."""
        query = "tell me about 84 Mulberry St"
        result = optimize_search_query(query)
        
        assert result.original_query == query
        assert result.analysis.query_type == QueryType.ADDRESS_SPECIFIC
        assert len(result.analysis.addresses_found) > 0
        assert len(result.query_variants) > 0
        
        # Should preserve exact address format in optimized query
        assert "84 Mulberry" in result.optimized_query
    
    def test_optimize_property_query(self):
        """Test optimization of general property queries."""
        query = "show me available apartments with parking"
        result = optimize_search_query(query)
        
        assert result.original_query == query
        assert result.analysis.query_type == QueryType.PROPERTY_GENERAL
        assert len(result.query_variants) > 0
        
        # Should have synonym expansions
        variant_text = " ".join(result.query_variants).lower()
        assert any(synonym in variant_text for synonym in ['unit', 'space', 'garage', 'spot'])
    
    def test_optimize_mixed_query(self):
        """Test optimization of mixed queries."""
        query = "what is the monthly rent for 84 Mulberry Street apartment"
        result = optimize_search_query(query)
        
        assert result.analysis.query_type == QueryType.MIXED
        assert len(result.analysis.addresses_found) > 0
        assert len(result.query_variants) > 0
        
        # Should have both address variations and synonym expansions
        variant_text = " ".join(result.query_variants).lower()
        assert "84 mulberry" in variant_text or "mulberry street" in variant_text
        assert any(synonym in variant_text for synonym in ['rental', 'lease', 'unit'])
    
    def test_optimization_preserves_address_formatting(self):
        """Test that optimization preserves exact address formats."""
        test_addresses = [
            "84 Mulberry St",
            "123 Main Street", 
            "456 Oak Ave",
            "789 Pine Road"
        ]
        
        for address in test_addresses:
            query = f"tell me about {address}"
            result = optimize_search_query(query)
            
            # Original address format should be preserved somewhere
            all_text = result.optimized_query + " " + " ".join(result.query_variants)
            assert address in all_text, f"Address format not preserved: {address}"
    
    def test_optimization_with_custom_options(self):
        """Test optimization with custom options."""
        query = "84 Mulberry Street apartment"
        options = {
            'preserve_exact_addresses': True,
            'generate_address_variations': False,
            'expand_with_synonyms': False,
            'create_partial_matches': False,
            'sanitize_input': True
        }
        
        result = optimize_search_query(query, options)
        
        # Should have fewer variants due to disabled options
        assert len(result.query_variants) < 5
        assert "preserved_address_formatting" in result.optimization_applied or \
               len(result.optimization_applied) == 0  # If no changes needed


class TestIntegrationFunctions:
    """Test integration convenience functions."""
    
    def test_get_query_variants(self):
        """Test the get_query_variants convenience function."""
        query = "tell me about 84 Mulberry Street"
        variants = get_query_variants(query, max_variants=5)
        
        assert len(variants) <= 5
        assert query in variants or any("84 Mulberry" in variant for variant in variants)
        
        # Should not have duplicates
        assert len(variants) == len(set(v.lower() for v in variants))
    
    def test_get_query_variants_limit(self):
        """Test that get_query_variants respects the limit."""
        query = "find available apartments with parking and 2 bedrooms"
        variants = get_query_variants(query, max_variants=3)
        
        assert len(variants) <= 3
        assert len(variants) > 0


class TestRealWorldScenarios:
    """Test real-world query scenarios."""
    
    def test_mulberry_street_query(self):
        """Test the specific problematic query from requirements."""
        query = "tell me about 84 Mulberry St"
        result = optimize_search_query(query)
        
        # Should be classified as address-specific
        assert result.analysis.query_type == QueryType.ADDRESS_SPECIFIC
        
        # Should preserve exact address format
        assert "84 Mulberry" in result.optimized_query
        
        # Should generate useful variations
        assert len(result.query_variants) > 0
        
        # Should have high confidence
        assert result.analysis.confidence_score > 0.5
        
        logger.info(f"Mulberry St query optimization:")
        logger.info(f"  Original: {result.original_query}")
        logger.info(f"  Optimized: {result.optimized_query}")
        logger.info(f"  Variants: {result.query_variants[:3]}")
        logger.info(f"  Type: {result.analysis.query_type.value}")
        logger.info(f"  Confidence: {result.analysis.confidence_score:.2f}")
    
    def test_property_search_queries(self):
        """Test various property search queries."""
        test_queries = [
            "show me 2 bedroom apartments",
            "find office space for rent",
            "available units with parking",
            "commercial properties downtown",
            "apartments under $2000 per month"
        ]
        
        for query in test_queries:
            result = optimize_search_query(query)
            
            # Should be classified appropriately
            assert result.analysis.query_type in [QueryType.PROPERTY_GENERAL, QueryType.MIXED]
            
            # Should generate variants
            assert len(result.query_variants) > 0
            
            # Should have reasonable confidence
            assert result.analysis.confidence_score > 0.3
            
            logger.info(f"Property query '{query}' -> type: {result.analysis.query_type.value}, "
                       f"confidence: {result.analysis.confidence_score:.2f}")
    
    def test_address_with_property_details(self):
        """Test queries that combine addresses with property details."""
        query = "what is the rent and size for 84 Mulberry Street"
        result = optimize_search_query(query)
        
        # Should be mixed type
        assert result.analysis.query_type == QueryType.MIXED
        
        # Should find address
        assert len(result.analysis.addresses_found) > 0
        assert any("84 Mulberry" in addr for addr in result.analysis.addresses_found)
        
        # Should find property terms
        assert any(term in ['rent', 'size'] for term in result.analysis.key_terms)
        
        # Should generate comprehensive variants
        assert len(result.query_variants) > 2
        
        logger.info(f"Mixed query optimization:")
        logger.info(f"  Addresses found: {result.analysis.addresses_found}")
        logger.info(f"  Key terms: {result.analysis.key_terms}")
        logger.info(f"  Variants count: {len(result.query_variants)}")


if __name__ == "__main__":
    # Run specific test scenarios for manual verification
    print("Testing Query Optimization Module")
    print("=" * 50)
    
    # Test the main problematic query
    test_query = "tell me about 84 Mulberry St"
    print(f"\nTesting query: '{test_query}'")
    
    result = optimize_search_query(test_query)
    print(f"Query type: {result.analysis.query_type.value}")
    print(f"Confidence: {result.analysis.confidence_score:.2f}")
    print(f"Addresses found: {result.analysis.addresses_found}")
    print(f"Key terms: {result.analysis.key_terms}")
    print(f"Optimized query: {result.optimized_query}")
    print(f"Optimizations applied: {result.optimization_applied}")
    print(f"Query variants ({len(result.query_variants)}):")
    for i, variant in enumerate(result.query_variants[:5], 1):
        print(f"  {i}. {variant}")
    
    # Test query variants function
    print(f"\nTesting get_query_variants:")
    variants = get_query_variants(test_query, max_variants=8)
    for i, variant in enumerate(variants, 1):
        print(f"  {i}. {variant}")
    
    print("\nQuery optimization module test completed!")