# /test/test_multi_strategy_search.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from llama_index.core.schema import NodeWithScore, TextNode
from app.multi_strategy_search import (
    MultiStrategySearcher,
    AddressExtractor,
    QueryProcessor,
    SearchResultRanker,
    multi_strategy_search
)


class TestAddressExtractor:
    """Test the AddressExtractor utility class."""
    
    def test_extract_addresses_full_address(self):
        """Test extracting full addresses with street types."""
        text = "tell me about 84 Mulberry St and 123 Main Street"
        addresses = AddressExtractor.extract_addresses(text)
        assert "84 Mulberry St" in addresses
        assert "123 Main Street" in addresses
        assert len(addresses) == 2
    
    def test_extract_addresses_simple_pattern(self):
        """Test extracting simple number + word patterns."""
        text = "what about 84 Mulberry"
        addresses = AddressExtractor.extract_addresses(text)
        assert "84 Mulberry" in addresses
    
    def test_extract_addresses_no_duplicates(self):
        """Test that duplicate addresses are removed."""
        text = "84 Mulberry St and 84 Mulberry St again"
        addresses = AddressExtractor.extract_addresses(text)
        assert len(addresses) == 1
        assert "84 Mulberry St" in addresses
    
    def test_normalize_address(self):
        """Test address normalization."""
        address = "  84   Mulberry   St  "
        normalized = AddressExtractor.normalize_address(address)
        assert normalized == "84 Mulberry St"


class TestQueryProcessor:
    """Test the QueryProcessor utility class."""
    
    def test_extract_key_terms(self):
        """Test extracting key terms from queries."""
        query = "tell me about the property with rent and size"
        terms = QueryProcessor.extract_key_terms(query)
        assert "property" in terms
        assert "rent" in terms
        assert "size" in terms
        # Stop words should be filtered out
        assert "the" not in terms
        assert "with" not in terms
    
    def test_create_fuzzy_query(self):
        """Test creating fuzzy queries."""
        query = "tell me about 84 Mulberry St with good rent"
        fuzzy = QueryProcessor.create_fuzzy_query(query)
        assert "84 Mulberry St" in fuzzy
        assert "rent" in fuzzy


class TestSearchResultRanker:
    """Test the SearchResultRanker utility class."""
    
    def test_rank_nodes_with_address_match(self):
        """Test that nodes with address matches get higher scores."""
        # Create mock nodes
        node1 = Mock(spec=NodeWithScore)
        node1.get_content.return_value = "Property at 84 Mulberry St with rent $1000"
        node1.score = 0.5
        
        node2 = Mock(spec=NodeWithScore)
        node2.get_content.return_value = "Different property with rent $2000"
        node2.score = 0.7
        
        nodes = [node1, node2]
        query = "84 Mulberry St"
        
        ranked = SearchResultRanker.rank_nodes(nodes, query)
        
        # Node1 should be ranked higher due to address match
        assert ranked[0] == node1
        assert ranked[0].score > ranked[1].score
    
    def test_validate_results_filters_irrelevant(self):
        """Test that validation filters out irrelevant results."""
        # Create mock nodes
        relevant_node = Mock(spec=NodeWithScore)
        relevant_node.get_content.return_value = "Property at 84 Mulberry St"
        relevant_node.score = 0.5
        
        irrelevant_node = Mock(spec=NodeWithScore)
        irrelevant_node.get_content.return_value = "Completely unrelated content"
        irrelevant_node.score = 0.1
        
        nodes = [relevant_node, irrelevant_node]
        query = "84 Mulberry St"
        
        validated = SearchResultRanker.validate_results(nodes, query)
        
        # Only the relevant node should remain
        assert len(validated) == 1
        assert validated[0] == relevant_node


class TestMultiStrategySearcher:
    """Test the main MultiStrategySearcher class."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever for testing."""
        retriever = AsyncMock()
        return retriever
    
    @pytest.fixture
    def searcher(self, mock_retriever):
        """Create a MultiStrategySearcher instance for testing."""
        return MultiStrategySearcher(mock_retriever)
    
    @pytest.mark.asyncio
    async def test_execute_search_strategy_success(self, searcher, mock_retriever):
        """Test successful execution of a search strategy."""
        # Mock successful search
        mock_node = Mock(spec=NodeWithScore)
        mock_node.get_content.return_value = "Test content"
        mock_retriever.aretrieve.return_value = [mock_node]
        
        result = await searcher._execute_search_strategy("test query", "exact")
        
        assert result.strategy == "exact"
        assert result.success is True
        assert len(result.nodes) == 1
        assert result.query_used == "test query"
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_execute_search_strategy_failure(self, searcher, mock_retriever):
        """Test handling of search strategy failures."""
        # Mock failed search
        mock_retriever.aretrieve.side_effect = Exception("Search failed")
        
        result = await searcher._execute_search_strategy("test query", "exact")
        
        assert result.strategy == "exact"
        assert result.success is False
        assert len(result.nodes) == 0
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_search_with_multiple_strategies(self, searcher, mock_retriever):
        """Test the full multi-strategy search process."""
        # Mock different results for different strategies
        def mock_aretrieve(query):
            if "84 Mulberry" in query:
                node = Mock(spec=NodeWithScore)
                node.get_content.return_value = f"Property at 84 Mulberry St, query: {query}"
                node.score = 0.8
                return [node]
            else:
                return []
        
        mock_retriever.aretrieve.side_effect = mock_aretrieve
        
        result = await searcher.search_with_multiple_strategies("tell me about 84 Mulberry St")
        
        assert result.original_query == "tell me about 84 Mulberry St"
        assert len(result.all_results) > 0
        assert result.best_result is not None
        assert len(result.nodes_found) > 0
        assert result.total_execution_time_ms > 0


@pytest.mark.asyncio
async def test_multi_strategy_search_convenience_function():
    """Test the convenience function for multi-strategy search."""
    # Create a mock retriever
    mock_retriever = AsyncMock()
    mock_node = Mock(spec=NodeWithScore)
    mock_node.get_content.return_value = "Test property content"
    mock_node.score = 0.7
    mock_retriever.aretrieve.return_value = [mock_node]
    
    # Test the convenience function
    result = await multi_strategy_search(mock_retriever, "test query")
    
    assert result.original_query == "test query"
    assert len(result.all_results) > 0
    assert result.total_execution_time_ms > 0


if __name__ == "__main__":
    # Run a simple test to verify the module works
    async def simple_test():
        print("Testing AddressExtractor...")
        addresses = AddressExtractor.extract_addresses("tell me about 84 Mulberry St")
        print(f"Extracted addresses: {addresses}")
        
        print("Testing QueryProcessor...")
        terms = QueryProcessor.extract_key_terms("tell me about the property")
        print(f"Key terms: {terms}")
        
        fuzzy = QueryProcessor.create_fuzzy_query("tell me about 84 Mulberry St")
        print(f"Fuzzy query: {fuzzy}")
        
        print("All tests completed successfully!")
    
    asyncio.run(simple_test())