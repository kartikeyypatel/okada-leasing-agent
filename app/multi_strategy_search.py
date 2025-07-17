# /app/multi_strategy_search.py
import re
import logging
import asyncio
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle

# Import query optimization functionality
from app.query_optimizer import optimize_search_query, QueryType, OptimizedQuery

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a single search strategy."""
    strategy: str  # "exact", "original", "address_only", "fuzzy"
    nodes: List[NodeWithScore]
    success: bool
    execution_time_ms: float
    query_used: str


@dataclass
class MultiStrategySearchResult:
    """Combined result from all search strategies."""
    original_query: str
    best_result: Optional[SearchResult]
    all_results: List[SearchResult]
    total_execution_time_ms: float
    nodes_found: List[NodeWithScore]


class AddressExtractor:
    """Utility class for extracting and normalizing addresses from queries."""
    
    # Common address patterns
    ADDRESS_PATTERNS = [
        # Full address with number and street
        r'\b\d+\s+[A-Za-z\s]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place)\b',
        # Street name without number
        r'\b[A-Za-z\s]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Pl|Place)\b',
        # Simple number + word pattern (like "84 Mulberry")
        r'\b\d+\s+[A-Za-z]+\b'
    ]
    
    @classmethod
    def extract_addresses(cls, text: str) -> List[str]:
        """Extract potential addresses from text."""
        addresses = []
        for pattern in cls.ADDRESS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_addresses = []
        for addr in addresses:
            addr_normalized = addr.strip().lower()
            if addr_normalized not in seen:
                seen.add(addr_normalized)
                unique_addresses.append(addr.strip())
        
        return unique_addresses
    
    @classmethod
    def normalize_address(cls, address: str) -> str:
        """Normalize address for better matching."""
        # Basic normalization - preserve exact format but clean up
        normalized = address.strip()
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized


class QueryProcessor:
    """Utility class for processing and generating search queries."""
    
    @classmethod
    def extract_key_terms(cls, query: str) -> List[str]:
        """Extract key search terms from a query."""
        # Remove common stop words but keep important real estate terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    @classmethod
    def create_fuzzy_query(cls, original_query: str) -> str:
        """Create a fuzzy search query by extracting key terms."""
        addresses = AddressExtractor.extract_addresses(original_query)
        key_terms = cls.extract_key_terms(original_query)
        
        # Combine addresses and key terms
        fuzzy_terms = []
        fuzzy_terms.extend(addresses)
        fuzzy_terms.extend(key_terms[:5])  # Limit to top 5 key terms
        
        return ' '.join(fuzzy_terms)


class SearchResultRanker:
    """Utility class for ranking and validating search results."""
    
    @classmethod
    def rank_nodes(cls, nodes: List[NodeWithScore], original_query: str) -> List[NodeWithScore]:
        """Rank nodes based on relevance to the original query."""
        if not nodes:
            return nodes
        
        # Extract addresses from original query for address-specific ranking
        query_addresses = AddressExtractor.extract_addresses(original_query)
        query_lower = original_query.lower()
        
        def calculate_relevance_score(node: NodeWithScore) -> float:
            """Calculate relevance score for a node."""
            content = node.get_content().lower()
            base_score = node.score if hasattr(node, 'score') and node.score else 0.0
            
            # Address matching bonus
            address_bonus = 0.0
            for addr in query_addresses:
                if addr.lower() in content:
                    address_bonus += 0.3  # Significant bonus for address match
            
            # Exact phrase matching bonus
            phrase_bonus = 0.0
            if query_lower in content:
                phrase_bonus = 0.2
            
            # Key terms matching
            key_terms = QueryProcessor.extract_key_terms(original_query)
            term_matches = sum(1 for term in key_terms if term in content)
            term_bonus = (term_matches / len(key_terms)) * 0.1 if key_terms else 0.0
            
            total_score = base_score + address_bonus + phrase_bonus + term_bonus
            return total_score
        
        # Sort by relevance score (descending)
        ranked_nodes = sorted(nodes, key=calculate_relevance_score, reverse=True)
        
        # Update scores with new relevance scores
        for node in ranked_nodes:
            node.score = calculate_relevance_score(node)
        
        return ranked_nodes
    
    @classmethod
    def validate_results(cls, nodes: List[NodeWithScore], original_query: str) -> List[NodeWithScore]:
        """Validate that results are relevant to the query."""
        if not nodes:
            return nodes
        
        query_addresses = AddressExtractor.extract_addresses(original_query)
        query_terms = QueryProcessor.extract_key_terms(original_query)
        
        validated_nodes = []
        
        for node in nodes:
            content = node.get_content().lower()
            is_relevant = False
            
            # Check for address matches
            for addr in query_addresses:
                if addr.lower() in content:
                    is_relevant = True
                    break
            
            # Check for key term matches (at least 1 term should match)
            if not is_relevant and query_terms:
                term_matches = sum(1 for term in query_terms if term in content)
                if term_matches > 0:
                    is_relevant = True
            
            # If no specific criteria, include nodes with reasonable scores
            if not is_relevant and hasattr(node, 'score') and node.score and node.score > 0.1:
                is_relevant = True
            
            if is_relevant:
                validated_nodes.append(node)
        
        return validated_nodes


class MultiStrategySearcher:
    """Main class for implementing multi-strategy search logic."""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def _execute_search_strategy(self, query: str, strategy_name: str) -> SearchResult:
        """Execute a single search strategy and measure performance."""
        start_time = datetime.now()
        
        try:
            # Perform the search
            nodes = await self.retriever.aretrieve(query)
            success = True
            self.logger.debug(f"Strategy '{strategy_name}' found {len(nodes)} nodes for query: '{query}'")
            
        except Exception as e:
            self.logger.error(f"Strategy '{strategy_name}' failed for query '{query}': {e}")
            nodes = []
            success = False
        
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return SearchResult(
            strategy=strategy_name,
            nodes=nodes,
            success=success,
            execution_time_ms=execution_time_ms,
            query_used=query
        )
    
    async def search_with_multiple_strategies(self, original_query: str) -> MultiStrategySearchResult:
        """
        Perform search using multiple strategies with query optimization.
        
        Enhanced strategies with query optimization:
        1. Optimized - Use the query optimizer's main optimized query
        2. Address variations - Use normalized address variations
        3. Synonym expansion - Use synonym-expanded queries
        4. Partial matches - Use partial query variations
        5. Fallback - Original legacy strategies if optimization fails
        """
        start_time = datetime.now()
        all_results = []
        
        # Step 1: Optimize the query
        self.logger.info(f"Optimizing query: '{original_query}'")
        try:
            optimized_query_result = optimize_search_query(original_query)
            self.logger.info(f"Query optimization completed: type={optimized_query_result.analysis.query_type.value}, "
                           f"optimizations={optimized_query_result.optimization_applied}")
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}, falling back to original strategies")
            optimized_query_result = None
        
        if optimized_query_result:
            # Strategy 1: Optimized main query
            optimized_query = optimized_query_result.optimized_query
            if optimized_query != original_query:
                self.logger.info(f"Executing optimized query strategy for: '{optimized_query}'")
                optimized_result = await self._execute_search_strategy(optimized_query, "optimized")
                all_results.append(optimized_result)
            
            # Strategy 2: Query variants (address variations, synonyms, etc.)
            for i, variant in enumerate(optimized_query_result.query_variants[:5]):  # Limit to top 5 variants
                variant_strategy = f"variant_{i+1}"
                self.logger.info(f"Executing {variant_strategy} strategy for: '{variant}'")
                variant_result = await self._execute_search_strategy(variant, variant_strategy)
                all_results.append(variant_result)
            
            # Strategy 3: Address-specific optimization for address queries
            if optimized_query_result.analysis.query_type == QueryType.ADDRESS_SPECIFIC:
                for addr in optimized_query_result.analysis.addresses_found:
                    self.logger.info(f"Executing address-specific strategy for: '{addr}'")
                    addr_result = await self._execute_search_strategy(addr, f"address_specific_{addr}")
                    all_results.append(addr_result)
        
        # Strategy 4: Always try the exact original query
        self.logger.info(f"Executing exact original query strategy for: '{original_query}'")
        exact_result = await self._execute_search_strategy(original_query, "exact_original")
        all_results.append(exact_result)
        
        # Strategy 5: Fallback to legacy strategies if we don't have enough results
        total_nodes_so_far = sum(len(result.nodes) for result in all_results if result.success)
        if total_nodes_so_far < 3:  # If we have fewer than 3 results, try legacy strategies
            self.logger.info("Insufficient results from optimization, trying legacy strategies")
            
            # Legacy address-only search
            addresses = AddressExtractor.extract_addresses(original_query)
            if addresses:
                address_nodes = []
                address_success = True
                address_time = 0.0
                
                for addr in addresses:
                    self.logger.info(f"Executing legacy address-only strategy for: '{addr}'")
                    addr_result = await self._execute_search_strategy(addr, f"legacy_address_{addr}")
                    address_nodes.extend(addr_result.nodes)
                    address_time += addr_result.execution_time_ms
                    if not addr_result.success:
                        address_success = False
                
                # Remove duplicates
                unique_address_nodes = []
                seen_content = set()
                for node in address_nodes:
                    content_hash = hash(node.get_content())
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_address_nodes.append(node)
                
                if unique_address_nodes:
                    address_result = SearchResult(
                        strategy="legacy_address_only",
                        nodes=unique_address_nodes,
                        success=address_success,
                        execution_time_ms=address_time,
                        query_used=" | ".join(addresses)
                    )
                    all_results.append(address_result)
            
            # Legacy fuzzy search
            fuzzy_query = QueryProcessor.create_fuzzy_query(original_query)
            if fuzzy_query and fuzzy_query != original_query:
                self.logger.info(f"Executing legacy fuzzy strategy for: '{fuzzy_query}'")
                fuzzy_result = await self._execute_search_strategy(fuzzy_query, "legacy_fuzzy")
                all_results.append(fuzzy_result)
        
        # Determine the best result using enhanced selection
        best_result = self._select_best_result_enhanced(all_results, original_query, optimized_query_result)
        
        # Combine and rank all unique nodes
        all_nodes = []
        seen_content = set()
        
        for result in all_results:
            for node in result.nodes:
                content_hash = hash(node.get_content())
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_nodes.append(node)
        
        # Enhanced ranking and validation using query analysis
        if optimized_query_result:
            ranked_nodes = self._rank_nodes_enhanced(all_nodes, original_query, optimized_query_result.analysis)
            validated_nodes = self._validate_results_enhanced(ranked_nodes, original_query, optimized_query_result.analysis)
        else:
            # Fallback to original ranking
            ranked_nodes = SearchResultRanker.rank_nodes(all_nodes, original_query)
            validated_nodes = SearchResultRanker.validate_results(ranked_nodes, original_query)
        
        end_time = datetime.now()
        total_execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        self.logger.info(f"Enhanced multi-strategy search completed: {len(validated_nodes)} final nodes from {len(all_results)} strategies")
        
        return MultiStrategySearchResult(
            original_query=original_query,
            best_result=best_result,
            all_results=all_results,
            total_execution_time_ms=total_execution_time_ms,
            nodes_found=validated_nodes
        )
    
    def _select_best_result_enhanced(self, results: List[SearchResult], original_query: str, 
                                   optimized_query_result: Optional[OptimizedQuery]) -> Optional[SearchResult]:
        """Enhanced result selection using query analysis."""
        if not results:
            return None
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.nodes]
        
        if not successful_results:
            return None
        
        # Enhanced scoring with query analysis
        strategy_weights = {
            "optimized": 1.0,
            "exact_original": 0.95,
            "variant_1": 0.9,
            "variant_2": 0.85,
            "address_specific": 0.9,
            "exact": 0.8,
            "address_only": 0.75,
            "fuzzy": 0.6,
            "partial": 0.5,
            "legacy_address_only": 0.4,
            "legacy_fuzzy": 0.3
        }
        
        # Use query analysis if available
        if optimized_query_result:
            query_type = optimized_query_result.analysis.query_type
            addresses_found = optimized_query_result.analysis.addresses_found
        else:
            query_type = QueryType.UNKNOWN
            addresses_found = AddressExtractor.extract_addresses(original_query)
        
        def score_result_enhanced(result: SearchResult) -> float:
            base_score = len(result.nodes) * strategy_weights.get(result.strategy, 0.3)
            
            # Query type specific bonuses
            if query_type == QueryType.ADDRESS_SPECIFIC:
                if "address" in result.strategy or "optimized" in result.strategy:
                    base_score *= 1.3
            elif query_type == QueryType.PROPERTY_GENERAL:
                if "variant" in result.strategy or "optimized" in result.strategy:
                    base_score *= 1.2
            
            # Address matching bonus
            if addresses_found:
                for addr in addresses_found:
                    if any(addr.lower() in node.get_content().lower() for node in result.nodes):
                        base_score *= 1.2
                        break
            
            # Penalty for too many results (might be too broad)
            if len(result.nodes) > 15:
                base_score *= 0.7
            elif len(result.nodes) > 10:
                base_score *= 0.9
            
            return base_score
        
        # Select the result with the highest score
        best_result = max(successful_results, key=score_result_enhanced)
        
        self.logger.info(f"Enhanced selection: '{best_result.strategy}' with {len(best_result.nodes)} nodes")
        return best_result
    
    def _rank_nodes_enhanced(self, nodes: List[NodeWithScore], original_query: str, 
                           query_analysis) -> List[NodeWithScore]:
        """Enhanced node ranking using query analysis."""
        if not nodes:
            return nodes
        
        query_lower = original_query.lower()
        addresses_found = query_analysis.addresses_found
        key_terms = query_analysis.key_terms
        query_type = query_analysis.query_type
        
        def calculate_enhanced_relevance_score(node: NodeWithScore) -> float:
            content = node.get_content().lower()
            base_score = node.score if hasattr(node, 'score') and node.score else 0.0
            
            # Address matching bonus (higher for address-specific queries)
            address_bonus = 0.0
            for addr in addresses_found:
                if addr.lower() in content:
                    if query_type == QueryType.ADDRESS_SPECIFIC:
                        address_bonus += 0.5  # Higher bonus for address queries
                    else:
                        address_bonus += 0.3
            
            # Exact phrase matching bonus
            phrase_bonus = 0.0
            if query_lower in content:
                phrase_bonus = 0.3
            
            # Key terms matching with weighted importance
            term_bonus = 0.0
            if key_terms:
                term_matches = sum(1 for term in key_terms if term in content)
                term_bonus = (term_matches / len(key_terms)) * 0.2
            
            # Query type specific bonuses
            type_bonus = 0.0
            if query_type == QueryType.ADDRESS_SPECIFIC:
                # Look for property-specific information
                if any(word in content for word in ['rent', 'lease', 'sqft', 'bedroom', 'bathroom']):
                    type_bonus += 0.1
            elif query_type == QueryType.PROPERTY_GENERAL:
                # General property information bonus
                if any(word in content for word in ['property', 'building', 'unit', 'space']):
                    type_bonus += 0.1
            
            total_score = base_score + address_bonus + phrase_bonus + term_bonus + type_bonus
            return total_score
        
        # Sort by enhanced relevance score (descending)
        ranked_nodes = sorted(nodes, key=calculate_enhanced_relevance_score, reverse=True)
        
        # Update scores with new relevance scores
        for node in ranked_nodes:
            node.score = calculate_enhanced_relevance_score(node)
        
        return ranked_nodes
    
    def _validate_results_enhanced(self, nodes: List[NodeWithScore], original_query: str, 
                                 query_analysis) -> List[NodeWithScore]:
        """Enhanced result validation using query analysis."""
        if not nodes:
            return nodes
        
        addresses_found = query_analysis.addresses_found
        key_terms = query_analysis.key_terms
        query_type = query_analysis.query_type
        
        validated_nodes = []
        
        for node in nodes:
            content = node.get_content().lower()
            is_relevant = False
            relevance_reasons = []
            
            # Address matching validation
            for addr in addresses_found:
                if addr.lower() in content:
                    is_relevant = True
                    relevance_reasons.append(f"address_match:{addr}")
                    break
            
            # Key term validation (require at least 1 match for general queries)
            if not is_relevant and key_terms:
                term_matches = sum(1 for term in key_terms if term in content)
                if query_type == QueryType.ADDRESS_SPECIFIC:
                    # For address queries, be more lenient with key terms
                    if term_matches > 0:
                        is_relevant = True
                        relevance_reasons.append(f"key_terms:{term_matches}")
                else:
                    # For general queries, require more key term matches
                    if term_matches >= max(1, len(key_terms) // 3):
                        is_relevant = True
                        relevance_reasons.append(f"key_terms:{term_matches}")
            
            # Score-based validation for high-confidence results
            if not is_relevant and hasattr(node, 'score') and node.score:
                if query_type == QueryType.ADDRESS_SPECIFIC and node.score > 0.2:
                    is_relevant = True
                    relevance_reasons.append(f"high_score:{node.score:.3f}")
                elif node.score > 0.15:
                    is_relevant = True
                    relevance_reasons.append(f"score:{node.score:.3f}")
            
            if is_relevant:
                validated_nodes.append(node)
                self.logger.debug(f"Node validated: {relevance_reasons}")
        
        self.logger.info(f"Enhanced validation: {len(validated_nodes)}/{len(nodes)} nodes passed validation")
        return validated_nodes
    
    def _select_best_result(self, results: List[SearchResult], original_query: str) -> Optional[SearchResult]:
        """Select the best result from all strategies."""
        if not results:
            return None
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.nodes]
        
        if not successful_results:
            return None
        
        # Scoring criteria:
        # 1. Number of results (more is generally better, up to a point)
        # 2. Strategy preference (exact > address_only > fuzzy > partial)
        # 3. Address matching for address queries
        
        strategy_weights = {
            "exact": 1.0,
            "address_only": 0.9,
            "fuzzy": 0.7,
            "partial": 0.6
        }
        
        query_addresses = AddressExtractor.extract_addresses(original_query)
        has_address_query = len(query_addresses) > 0
        
        def score_result(result: SearchResult) -> float:
            base_score = len(result.nodes) * strategy_weights.get(result.strategy, 0.5)
            
            # Bonus for address queries that find address matches
            if has_address_query and result.strategy in ["exact", "address_only"]:
                base_score *= 1.2
            
            # Penalty for too many results (might be too broad)
            if len(result.nodes) > 10:
                base_score *= 0.8
            
            return base_score
        
        # Select the result with the highest score
        best_result = max(successful_results, key=score_result)
        
        self.logger.info(f"Selected best strategy: '{best_result.strategy}' with {len(best_result.nodes)} nodes")
        return best_result


# Convenience function for integration with existing code
async def multi_strategy_search(retriever: BaseRetriever, query: str) -> MultiStrategySearchResult:
    """
    Convenience function to perform multi-strategy search.
    
    Args:
        retriever: The retriever to use for searching
        query: The original user query
        
    Returns:
        MultiStrategySearchResult with the best results from all strategies
    """
    searcher = MultiStrategySearcher(retriever)
    return await searcher.search_with_multiple_strategies(query)