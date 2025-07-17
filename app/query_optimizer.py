# /app/query_optimizer.py
"""
Query optimization module for RAG chatbot search functionality.

This module provides query preprocessing, validation, sanitization, and expansion
capabilities to improve search accuracy, especially for address-based queries.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be processed."""
    ADDRESS_SPECIFIC = "address_specific"
    PROPERTY_GENERAL = "property_general"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Analysis result of a query."""
    original_query: str
    query_type: QueryType
    addresses_found: List[str]
    key_terms: List[str]
    confidence_score: float
    preprocessing_suggestions: List[str]


@dataclass
class OptimizedQuery:
    """Result of query optimization."""
    original_query: str
    optimized_query: str
    query_variants: List[str]
    analysis: QueryAnalysis
    optimization_applied: List[str]


class AddressNormalizer:
    """Handles address normalization and standardization."""
    
    # Common address abbreviations and their full forms
    STREET_ABBREVIATIONS = {
        'st': 'street',
        'ave': 'avenue', 
        'rd': 'road',
        'blvd': 'boulevard',
        'dr': 'drive',
        'ln': 'lane',
        'ct': 'court',
        'pl': 'place',
        'cir': 'circle',
        'way': 'way',
        'pkwy': 'parkway',
        'hwy': 'highway'
    }
    
    # Directional abbreviations
    DIRECTIONAL_ABBREVIATIONS = {
        'n': 'north',
        's': 'south', 
        'e': 'east',
        'w': 'west',
        'ne': 'northeast',
        'nw': 'northwest',
        'se': 'southeast',
        'sw': 'southwest'
    }
    
    # Address patterns for extraction
    ADDRESS_PATTERNS = [
        # Full address with number, street name, and type
        r'\b(\d+)\s+([A-Za-z\s]+?)\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place|Cir|Circle|Way|Pkwy|Parkway|Hwy|Highway)\b',
        # Address with directional
        r'\b(\d+)\s+([NSEW]|North|South|East|West|NE|NW|SE|SW)\s+([A-Za-z\s]+?)\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place)\b',
        # Simple number + street name (like "84 Mulberry")
        r'\b(\d+)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\b(?=\s|$|[,.!?])',
    ]
    
    @classmethod
    def extract_addresses(cls, text: str) -> List[Dict[str, str]]:
        """
        Extract addresses from text with detailed components.
        
        Returns:
            List of dictionaries with address components
        """
        addresses = []
        
        for pattern in cls.ADDRESS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 2:
                    address_dict = {
                        'full_match': match.group(0).strip(),
                        'number': groups[0] if groups[0] else '',
                        'street_name': '',
                        'street_type': '',
                        'directional': ''
                    }
                    
                    # Parse based on pattern structure
                    if len(groups) == 2:  # Simple pattern
                        address_dict['street_name'] = groups[1].strip()
                    elif len(groups) == 3:  # Number + street + type
                        address_dict['street_name'] = groups[1].strip()
                        address_dict['street_type'] = groups[2].strip()
                    elif len(groups) == 4:  # With directional
                        address_dict['directional'] = groups[1].strip()
                        address_dict['street_name'] = groups[2].strip()
                        address_dict['street_type'] = groups[3].strip()
                    
                    addresses.append(address_dict)
        
        # Remove duplicates while preserving order
        unique_addresses = []
        seen = set()
        for addr in addresses:
            addr_key = addr['full_match'].lower().strip()
            if addr_key not in seen:
                seen.add(addr_key)
                unique_addresses.append(addr)
        
        return unique_addresses
    
    @classmethod
    def normalize_address(cls, address: str) -> List[str]:
        """
        Generate normalized variations of an address.
        
        Returns:
            List of normalized address variations
        """
        variations = []
        original = address.strip()
        variations.append(original)
        
        # Extract address components
        addresses = cls.extract_addresses(address)
        
        for addr_dict in addresses:
            # Create variations with different abbreviations
            base_parts = []
            
            if addr_dict['number']:
                base_parts.append(addr_dict['number'])
            
            if addr_dict['directional']:
                # Add both abbreviated and full directional
                directional = addr_dict['directional'].lower()
                if directional in cls.DIRECTIONAL_ABBREVIATIONS:
                    base_parts.append(cls.DIRECTIONAL_ABBREVIATIONS[directional])
                    # Also add abbreviated version
                    variations.append(' '.join(base_parts[:-1] + [directional] + 
                                             [addr_dict['street_name']] + 
                                             ([addr_dict['street_type']] if addr_dict['street_type'] else [])))
                else:
                    base_parts.append(directional)
            
            if addr_dict['street_name']:
                base_parts.append(addr_dict['street_name'])
            
            if addr_dict['street_type']:
                street_type = addr_dict['street_type'].lower()
                # Add full form
                if street_type in cls.STREET_ABBREVIATIONS:
                    variations.append(' '.join(base_parts + [cls.STREET_ABBREVIATIONS[street_type]]))
                    # Add abbreviated form
                    variations.append(' '.join(base_parts + [street_type]))
                else:
                    # Check if it's already a full form
                    for abbrev, full in cls.STREET_ABBREVIATIONS.items():
                        if street_type == full:
                            variations.append(' '.join(base_parts + [abbrev]))
                            break
                    variations.append(' '.join(base_parts + [street_type]))
            else:
                # Address without street type
                variations.append(' '.join(base_parts))
        
        # Remove duplicates and empty strings
        unique_variations = []
        seen = set()
        for var in variations:
            var_clean = var.strip()
            if var_clean and var_clean.lower() not in seen:
                seen.add(var_clean.lower())
                unique_variations.append(var_clean)
        
        return unique_variations


class QueryAnalyzer:
    """Analyzes queries to determine type and extract key information."""
    
    # Property-related keywords
    PROPERTY_KEYWORDS = {
        'rent', 'rental', 'lease', 'leasing', 'property', 'building', 'apartment',
        'unit', 'space', 'office', 'commercial', 'residential', 'sqft', 'square feet',
        'bedroom', 'bathroom', 'parking', 'amenities', 'price', 'cost', 'monthly',
        'available', 'vacancy', 'floor', 'size'
    }
    
    # Question words and phrases
    QUESTION_INDICATORS = {
        'what', 'where', 'when', 'how', 'why', 'who', 'which', 'tell me', 'show me',
        'find', 'search', 'look for', 'information about', 'details about'
    }
    
    @classmethod
    def analyze_query(cls, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine its type and characteristics.
        
        Args:
            query: The user's search query
            
        Returns:
            QueryAnalysis with detailed information about the query
        """
        query_lower = query.lower().strip()
        
        # Extract addresses
        addresses = AddressNormalizer.extract_addresses(query)
        address_strings = [addr['full_match'] for addr in addresses]
        
        # Extract key terms (excluding stop words)
        key_terms = cls._extract_key_terms(query)
        
        # Determine query type
        query_type = cls._determine_query_type(query_lower, address_strings, key_terms)
        
        # Calculate confidence score
        confidence_score = cls._calculate_confidence_score(query_lower, address_strings, key_terms, query_type)
        
        # Generate preprocessing suggestions
        suggestions = cls._generate_preprocessing_suggestions(query, address_strings, key_terms, query_type)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            addresses_found=address_strings,
            key_terms=key_terms,
            confidence_score=confidence_score,
            preprocessing_suggestions=suggestions
        )
    
    @classmethod
    def _extract_key_terms(cls, query: str) -> List[str]:
        """Extract key terms from query, excluding stop words."""
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    @classmethod
    def _determine_query_type(cls, query_lower: str, addresses: List[str], key_terms: List[str]) -> QueryType:
        """Determine the type of query based on content analysis."""
        has_addresses = len(addresses) > 0
        has_property_keywords = any(keyword in query_lower for keyword in cls.PROPERTY_KEYWORDS)
        
        if has_addresses and has_property_keywords:
            return QueryType.MIXED
        elif has_addresses:
            return QueryType.ADDRESS_SPECIFIC
        elif has_property_keywords:
            return QueryType.PROPERTY_GENERAL
        else:
            return QueryType.UNKNOWN
    
    @classmethod
    def _calculate_confidence_score(cls, query_lower: str, addresses: List[str], 
                                  key_terms: List[str], query_type: QueryType) -> float:
        """Calculate confidence score for query analysis."""
        score = 0.0
        
        # Base score for having content
        if key_terms:
            score += 0.3
        
        # Address-specific scoring
        if addresses:
            score += 0.4
            # Bonus for well-formed addresses
            for addr in addresses:
                if re.search(r'\d+\s+\w+\s+(st|street|ave|avenue|rd|road)', addr.lower()):
                    score += 0.1
        
        # Property keyword scoring
        property_keyword_count = sum(1 for keyword in cls.PROPERTY_KEYWORDS if keyword in query_lower)
        score += min(property_keyword_count * 0.1, 0.3)
        
        # Question indicator scoring
        has_question_indicators = any(indicator in query_lower for indicator in cls.QUESTION_INDICATORS)
        if has_question_indicators:
            score += 0.1
        
        # Query type consistency bonus
        if query_type != QueryType.UNKNOWN:
            score += 0.1
        
        return min(score, 1.0)
    
    @classmethod
    def _generate_preprocessing_suggestions(cls, query: str, addresses: List[str], 
                                         key_terms: List[str], query_type: QueryType) -> List[str]:
        """Generate suggestions for query preprocessing."""
        suggestions = []
        
        if addresses:
            suggestions.append("normalize_addresses")
            suggestions.append("create_address_variations")
        
        if query_type == QueryType.ADDRESS_SPECIFIC:
            suggestions.append("prioritize_exact_address_matching")
        
        if len(key_terms) > 5:
            suggestions.append("limit_key_terms")
        
        if query_type == QueryType.UNKNOWN:
            suggestions.append("expand_query_with_synonyms")
        
        # Check for potential typos or formatting issues
        if re.search(r'\b\d+[a-zA-Z]', query):  # Number directly followed by letters
            suggestions.append("fix_address_spacing")
        
        return suggestions


class QueryExpander:
    """Handles query expansion for better matching."""
    
    # Synonyms for common real estate terms
    SYNONYMS = {
        'apartment': ['apt', 'unit', 'flat', 'residence'],
        'rent': ['rental', 'lease', 'monthly'],
        'size': ['sqft', 'square feet', 'area', 'footage'],
        'bedroom': ['bed', 'br', 'room'],
        'bathroom': ['bath', 'ba', 'restroom'],
        'parking': ['garage', 'spot', 'space'],
        'available': ['vacant', 'open', 'free'],
        'building': ['property', 'structure', 'complex']
    }
    
    @classmethod
    def expand_query(cls, query: str, expansion_type: str = "synonyms") -> List[str]:
        """
        Expand query with additional terms for better matching.
        
        Args:
            query: Original query
            expansion_type: Type of expansion ("synonyms", "partial", "fuzzy")
            
        Returns:
            List of expanded query variations
        """
        expansions = [query]  # Always include original
        
        if expansion_type == "synonyms":
            expansions.extend(cls._expand_with_synonyms(query))
        elif expansion_type == "partial":
            expansions.extend(cls._create_partial_queries(query))
        elif expansion_type == "fuzzy":
            expansions.extend(cls._create_fuzzy_queries(query))
        
        # Remove duplicates while preserving order
        unique_expansions = []
        seen = set()
        for exp in expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique_expansions.append(exp)
        
        return unique_expansions
    
    @classmethod
    def _expand_with_synonyms(cls, query: str) -> List[str]:
        """Expand query by adding synonyms for key terms."""
        expansions = []
        words = query.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in cls.SYNONYMS:
                for synonym in cls.SYNONYMS[word_lower]:
                    # Create new query with synonym
                    new_words = words.copy()
                    new_words[i] = synonym
                    expansions.append(' '.join(new_words))
        
        return expansions
    
    @classmethod
    def _create_partial_queries(cls, query: str) -> List[str]:
        """Create partial queries for broader matching."""
        words = query.split()
        partials = []
        
        if len(words) > 2:
            # First half
            partials.append(' '.join(words[:len(words)//2]))
            # Last half
            partials.append(' '.join(words[len(words)//2:]))
            # First 3 words
            if len(words) > 3:
                partials.append(' '.join(words[:3]))
            # Last 3 words
            if len(words) > 3:
                partials.append(' '.join(words[-3:]))
        
        return partials
    
    @classmethod
    def _create_fuzzy_queries(cls, query: str) -> List[str]:
        """Create fuzzy queries by extracting key terms."""
        # Extract addresses and key terms
        addresses = AddressNormalizer.extract_addresses(query)
        key_terms = QueryAnalyzer._extract_key_terms(query)
        
        fuzzy_queries = []
        
        # Address-only queries
        for addr in addresses:
            fuzzy_queries.append(addr['full_match'])
        
        # Key terms only
        if key_terms:
            fuzzy_queries.append(' '.join(key_terms[:5]))  # Top 5 key terms
        
        # Combination of addresses and top key terms
        if addresses and key_terms:
            for addr in addresses:
                combined = f"{addr['full_match']} {' '.join(key_terms[:3])}"
                fuzzy_queries.append(combined)
        
        return fuzzy_queries


class QueryValidator:
    """Validates and sanitizes queries."""
    
    # Potentially problematic patterns
    PROBLEMATIC_PATTERNS = [
        r'[<>{}[\]\\]',  # HTML/code injection
        r'(script|javascript|eval)',  # Script injection
        r'(\bor\b|\band\b).*[=<>]',  # SQL-like injection patterns
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, List[str]]:
        """
        Validate query for safety and format issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check length
        if len(query.strip()) == 0:
            issues.append("empty_query")
        elif len(query) > 500:
            issues.append("query_too_long")
        
        # Check for problematic patterns
        for pattern in cls.PROBLEMATIC_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(f"suspicious_pattern: {pattern}")
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', query)) / len(query) if query else 0
        if special_char_ratio > 0.3:
            issues.append("excessive_special_characters")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """
        Sanitize query by removing or fixing problematic content.
        
        Args:
            query: Original query
            
        Returns:
            Sanitized query
        """
        sanitized = query.strip()
        
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove potentially dangerous characters but preserve addresses
        # Keep alphanumeric, spaces, common punctuation for addresses
        sanitized = re.sub(r'[^\w\s\-.,#&/()]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 500:
            sanitized = sanitized[:500].strip()
        
        return sanitized


class QueryOptimizer:
    """Main class that orchestrates query optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_query(self, query: str, optimization_options: Optional[Dict] = None) -> OptimizedQuery:
        """
        Optimize a query for better search performance.
        
        Args:
            query: Original user query
            optimization_options: Optional configuration for optimization
            
        Returns:
            OptimizedQuery with optimized query and variants
        """
        if optimization_options is None:
            optimization_options = {
                'preserve_exact_addresses': True,
                'generate_address_variations': True,
                'expand_with_synonyms': True,
                'create_partial_matches': True,
                'sanitize_input': True
            }
        
        self.logger.info(f"Optimizing query: '{query}'")
        
        # Step 1: Validate and sanitize
        optimizations_applied = []
        working_query = query
        
        if optimization_options.get('sanitize_input', True):
            is_valid, issues = QueryValidator.validate_query(query)
            if not is_valid:
                self.logger.warning(f"Query validation issues: {issues}")
                optimizations_applied.append("validation_issues_detected")
            
            sanitized = QueryValidator.sanitize_query(query)
            if sanitized != query:
                working_query = sanitized
                optimizations_applied.append("sanitized_input")
                self.logger.info(f"Sanitized query: '{working_query}'")
        
        # Step 2: Analyze query
        analysis = QueryAnalyzer.analyze_query(working_query)
        self.logger.info(f"Query analysis: type={analysis.query_type.value}, "
                        f"addresses={len(analysis.addresses_found)}, "
                        f"confidence={analysis.confidence_score:.2f}")
        
        # Step 3: Apply optimizations based on analysis
        optimized_query = working_query
        query_variants = []
        
        # Address-specific optimizations
        if analysis.addresses_found and optimization_options.get('preserve_exact_addresses', True):
            # Ensure exact address format is preserved
            optimized_query = self._preserve_address_formatting(working_query, analysis.addresses_found)
            if optimized_query != working_query:
                optimizations_applied.append("preserved_address_formatting")
        
        # Generate address variations
        if analysis.addresses_found and optimization_options.get('generate_address_variations', True):
            address_variants = self._generate_address_variants(working_query, analysis.addresses_found)
            query_variants.extend(address_variants)
            optimizations_applied.append("generated_address_variations")
        
        # Expand with synonyms
        if optimization_options.get('expand_with_synonyms', True):
            synonym_variants = QueryExpander.expand_query(optimized_query, "synonyms")
            query_variants.extend(synonym_variants)
            optimizations_applied.append("expanded_with_synonyms")
        
        # Create partial matches
        if optimization_options.get('create_partial_matches', True):
            partial_variants = QueryExpander.expand_query(optimized_query, "partial")
            query_variants.extend(partial_variants)
            optimizations_applied.append("created_partial_matches")
        
        # Remove duplicates from variants
        unique_variants = []
        seen = set()
        seen.add(optimized_query.lower())  # Don't duplicate the main optimized query
        
        for variant in query_variants:
            if variant.lower() not in seen:
                seen.add(variant.lower())
                unique_variants.append(variant)
        
        self.logger.info(f"Generated {len(unique_variants)} query variants with optimizations: {optimizations_applied}")
        
        return OptimizedQuery(
            original_query=query,
            optimized_query=optimized_query,
            query_variants=unique_variants,
            analysis=analysis,
            optimization_applied=optimizations_applied
        )
    
    def _preserve_address_formatting(self, query: str, addresses: List[str]) -> str:
        """Ensure address formatting is preserved in the optimized query."""
        # For now, return the query as-is since we want to preserve exact formatting
        # Future enhancements could include smart formatting fixes
        return query
    
    def _generate_address_variants(self, query: str, addresses: List[str]) -> List[str]:
        """Generate address-based query variants."""
        variants = []
        
        for address in addresses:
            # Generate normalized variations of each address
            address_variations = AddressNormalizer.normalize_address(address)
            
            # Create query variants by replacing the original address with variations
            for variation in address_variations:
                if variation != address:  # Don't duplicate original
                    variant_query = query.replace(address, variation)
                    variants.append(variant_query)
        
        return variants


# Convenience functions for integration
def optimize_search_query(query: str, options: Optional[Dict] = None) -> OptimizedQuery:
    """
    Convenience function to optimize a search query.
    
    Args:
        query: The user's search query
        options: Optional optimization configuration
        
    Returns:
        OptimizedQuery with optimized query and variants
    """
    optimizer = QueryOptimizer()
    return optimizer.optimize_query(query, options)


def get_query_variants(query: str, max_variants: int = 10) -> List[str]:
    """
    Get optimized query variants for multi-strategy search.
    
    Args:
        query: Original query
        max_variants: Maximum number of variants to return
        
    Returns:
        List of query variants including the optimized main query
    """
    optimized = optimize_search_query(query)
    
    # Combine optimized query with variants
    all_variants = [optimized.optimized_query] + optimized.query_variants
    
    # Remove duplicates and limit count
    unique_variants = []
    seen = set()
    
    for variant in all_variants:
        if variant.lower() not in seen and len(unique_variants) < max_variants:
            seen.add(variant.lower())
            unique_variants.append(variant)
    
    return unique_variants