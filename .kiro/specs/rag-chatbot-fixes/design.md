# Design Document

## Overview

This design addresses critical issues in the RAG chatbot system that prevent accurate property information retrieval. The main problems identified are: fusion retriever not being created, user-specific indexing failures, and response hallucination. The solution involves fixing the indexing pipeline, improving search logic, and implementing strict response generation.

## Architecture

### Current Issues Identified
1. **Fusion Retriever Creation Failure**: "No fusion retriever available" error indicates the user's index or BM25 retriever isn't being created properly
2. **User Context Mismatch**: The system may not be properly associating queries with the correct user's data (ok@gmail.com)
3. **Search Query Processing**: Generated search queries may not match the exact address format in the data
4. **Response Hallucination**: The system may be generating responses without proper context validation

### Proposed Architecture Fixes

```mermaid
graph TD
    A[Chat Request with user_id] --> B{User Index Exists?}
    B -->|No| C[Check User Documents]
    C --> D{Documents Found?}
    D -->|Yes| E[Auto-Build Index]
    D -->|No| F[Return Error]
    E --> G[Create Fusion Retriever]
    B -->|Yes| G
    G --> H[Perform Search]
    H --> I{Results Found?}
    I -->|Yes| J[Generate Response with Context]
    I -->|No| K[Try Alternative Search]
    K --> L{Alternative Results?}
    L -->|Yes| J
    L -->|No| M[Return "Not Found" Message]
    J --> N[Return Response]
    M --> N
    F --> N
```

## Components and Interfaces

### 1. Enhanced Index Management
**Purpose**: Ensure reliable user-specific index creation and retrieval
**Key Changes**:
- Automatic index building when documents exist but no index is found
- Better error handling and logging for index operations
- Validation of index completeness before creating retrievers

```python
async def ensure_user_index(user_id: str) -> Optional[VectorStoreIndex]:
    """Ensure user has a valid index, building it if necessary"""
    # Check if index exists and is valid
    # Auto-build if documents exist but no index
    # Return None only if no documents exist
```

### 2. Improved Search Logic
**Purpose**: Better handling of exact address searches and fallback strategies
**Key Changes**:
- Preserve exact address format in search queries
- Implement multiple search strategies (exact match, partial match, fuzzy match)
- Better logging of search operations for debugging

```python
async def multi_strategy_search(retriever, query: str, original_query: str) -> List[NodeWithScore]:
    """Try multiple search strategies to find relevant documents"""
    # Strategy 1: Exact query
    # Strategy 2: Original user message
    # Strategy 3: Address-only extraction
    # Strategy 4: Fuzzy matching
```

### 3. Strict Response Generation
**Purpose**: Prevent hallucination and ensure responses only use retrieved context
**Key Changes**:
- Strict prompts that explicitly forbid making up information
- Context validation before response generation
- Clear "not found" messages when no relevant documents exist

```python
def create_strict_prompt(context: str, user_query: str) -> str:
    """Create a prompt that strictly uses only provided context"""
    return f"""Based ONLY on the following property information, answer the user's question.
    If the specific property is not in the provided information, clearly state that you don't have information about it.
    
    Property Information:
    {context}
    
    User's Question: {user_query}
    
    CRITICAL INSTRUCTIONS:
    - Only use information from the provided property data
    - If the exact property address is not found, say so clearly
    - Do not make up or invent property information
    - Be helpful and suggest alternatives if available"""
```

### 4. Comprehensive Debugging System
**Purpose**: Provide tools to diagnose and fix RAG system issues
**Components**:
- Health check endpoints for ChromaDB and indexing status
- Debug endpoints to test search functionality directly
- Detailed logging of all RAG operations
- User index status inspection tools

## Data Models

### Enhanced Logging Model
```python
@dataclass
class RAGOperationLog:
    timestamp: datetime
    user_id: str
    operation: str  # "index_check", "search", "response_generation"
    query: str
    results_count: int
    success: bool
    error_message: Optional[str]
    execution_time_ms: float
```

### Search Strategy Result
```python
@dataclass
class SearchResult:
    strategy: str  # "exact", "original", "address_only", "fuzzy"
    nodes: List[NodeWithScore]
    success: bool
    execution_time_ms: float
```

## Error Handling

### Index Creation Failures
- **Scenario**: ChromaDB connection issues or document processing errors
- **Handling**: 
  - Log detailed error information
  - Attempt fallback to in-memory indexing
  - Provide clear user feedback about the issue
  - Implement retry mechanisms with exponential backoff

### Search Failures
- **Scenario**: Retriever creation fails or search operations timeout
- **Handling**:
  - Log the specific failure point
  - Try alternative search strategies
  - Return informative error messages
  - Maintain system stability

### Response Generation Issues
- **Scenario**: LLM fails to generate response or generates invalid content
- **Handling**:
  - Validate response content before returning
  - Implement fallback response templates
  - Log generation failures for analysis
  - Ensure user always receives a response

## Testing Strategy

### Unit Tests
- Index creation and validation functions
- Search strategy implementations
- Response generation with various contexts
- Error handling scenarios

### Integration Tests
- End-to-end chat flow with real user data
- Multi-user isolation testing
- ChromaDB integration testing
- Performance under load

### Debug Testing
- All debug endpoints functionality
- Logging accuracy and completeness
- Error message clarity and usefulness
- System recovery from failures

## Implementation Priorities

### Phase 1: Critical Fixes (Immediate)
1. Fix fusion retriever creation issue
2. Implement automatic index building
3. Add comprehensive logging
4. Create debug endpoints

### Phase 2: Search Improvements (Short-term)
1. Implement multi-strategy search
2. Improve address matching logic
3. Add search result validation
4. Enhance error handling

### Phase 3: Response Quality (Medium-term)
1. Implement strict response generation
2. Add context validation
3. Improve "not found" handling
4. Add response quality metrics

## Performance Considerations

### Index Building
- Async operations to prevent blocking
- Progress tracking for large documents
- Caching of successfully built indexes
- Efficient document processing

### Search Operations
- Timeout handling for long searches
- Result caching for common queries
- Efficient retriever management
- Memory usage optimization

### Response Generation
- Prompt optimization for faster LLM responses
- Context size management
- Response caching where appropriate
- Error recovery mechanisms
</content>
</invoke>