# Design Document

## Overview

The RAG Chatbot Testing Framework is designed to comprehensively validate the chatbot's ability to retrieve and respond accurately to queries about commercial real estate properties. The framework will generate diverse test cases based on the HackathonInternalKnowledgeBase.csv dataset and validate responses against expected outcomes.

## Architecture

### Core Components

1. **Test Case Generator**: Analyzes the CSV dataset to create comprehensive test scenarios
2. **Query Executor**: Sends test queries to the RAG chatbot system
3. **Response Validator**: Compares chatbot responses against expected results
4. **Report Generator**: Creates detailed test reports with pass/fail status and analysis

### Data Flow

```
CSV Dataset → Test Case Generator → Query Executor → RAG Chatbot
                                                          ↓
Report Generator ← Response Validator ← Chatbot Response
```

## Components and Interfaces

### Test Case Generator

**Purpose**: Generate diverse test queries based on real estate data patterns

**Key Functions**:
- Parse CSV data to extract unique values and patterns
- Generate basic property information queries
- Create complex filtering and search scenarios
- Generate statistical and analytical queries
- Create edge case and error scenario tests

**Input**: HackathonInternalKnowledgeBase.csv
**Output**: Structured test cases with expected responses

### Query Categories

#### 1. Basic Property Information Queries
- Property address lookups
- Size and square footage queries
- Rental rate inquiries
- Floor and suite information
- Monthly/annual rent calculations

#### 2. Search and Filtering Queries
- Price range searches
- Size range filtering
- Location-based searches
- Multi-criteria filtering
- Threshold-based queries

#### 3. Statistical and Analytical Queries
- Average rental rate calculations
- Most/least expensive property identification
- Property count summaries
- Comparative analysis
- GCI projection queries

#### 4. Associate and Broker Queries
- Associate property assignments
- Broker contact information
- Associate workload analysis
- Team composition queries

#### 5. Edge Cases and Error Scenarios
- Non-existent property queries
- Ambiguous query handling
- Out-of-scope questions
- Malformed query processing

### Query Executor

**Purpose**: Interface with the RAG chatbot system to execute test queries

**Key Functions**:
- Send HTTP requests to chatbot API
- Handle authentication and session management
- Manage request timeouts and retries
- Log all interactions for debugging

**Interface**:
```python
class QueryExecutor:
    def execute_query(self, query: str) -> ChatbotResponse
    def batch_execute(self, queries: List[str]) -> List[ChatbotResponse]
```

### Response Validator

**Purpose**: Validate chatbot responses against expected outcomes

**Validation Types**:
1. **Exact Match Validation**: For specific data points (addresses, prices, names)
2. **Numerical Validation**: For calculations and ranges with tolerance
3. **Semantic Validation**: For natural language responses using NLP techniques
4. **Completeness Validation**: Ensuring all required information is present
5. **Consistency Validation**: Checking for consistent responses to similar queries

**Key Functions**:
- Extract structured data from natural language responses
- Compare numerical values with acceptable tolerance
- Validate presence of required information elements
- Check response relevance and accuracy

### Test Data Structure

```python
@dataclass
class TestCase:
    id: str
    category: str
    query: str
    expected_response: Dict[str, Any]
    validation_type: str
    tolerance: Optional[float] = None
    
@dataclass
class TestResult:
    test_case: TestCase
    actual_response: str
    passed: bool
    score: float
    errors: List[str]
    execution_time: float
```

## Data Models

### Property Data Model
```python
@dataclass
class Property:
    unique_id: int
    address: str
    floor: str
    suite: str
    size_sf: int
    rent_per_sf_year: float
    associate_1: str
    broker_email: str
    associate_2: str
    associate_3: str
    associate_4: str
    annual_rent: float
    monthly_rent: float
    gci_3_years: float
```

### Test Categories and Sample Queries

#### Basic Information Tests
- "What is the address of property ID 1?"
- "How big is the space at 36 W 36th St floor E3?"
- "What's the rent per square foot for suite 300 at 15 W 38th St?"

#### Search and Filter Tests
- "Show me properties with rent between $80-$90 per square foot"
- "Find properties larger than 15,000 square feet"
- "List all properties on Broadway"

#### Statistical Tests
- "What's the average rent per square foot across all properties?"
- "Which property has the highest annual rent?"
- "How many properties are managed by Hector Barbossa?"

#### Associate Tests
- "Who are the associates for 345 Seventh Avenue?"
- "What properties does Jack Sparrow handle?"
- "What's the broker email for properties on Times Square?"

#### Edge Case Tests
- "Tell me about 999 Fake Street"
- "What's the rent for a property with negative square footage?"
- "Show me properties managed by John Doe"

## Error Handling

### Response Validation Errors
- **Data Extraction Failures**: When structured data cannot be extracted from responses
- **Numerical Comparison Errors**: When expected numerical values don't match within tolerance
- **Missing Information Errors**: When required data elements are absent from responses
- **Format Validation Errors**: When responses don't follow expected formats

### System Integration Errors
- **API Connection Failures**: Handle chatbot service unavailability
- **Timeout Errors**: Manage slow response scenarios
- **Authentication Errors**: Handle access token issues
- **Rate Limiting**: Manage API rate limits with backoff strategies

## Testing Strategy

### Test Execution Phases

1. **Smoke Tests**: Basic functionality validation with simple queries
2. **Comprehensive Tests**: Full test suite execution across all categories
3. **Performance Tests**: Response time and throughput validation
4. **Stress Tests**: High-volume query handling
5. **Regression Tests**: Validation after system updates

### Success Criteria

- **Accuracy Rate**: >95% for basic property information queries
- **Search Precision**: >90% for filtering and search queries
- **Statistical Accuracy**: >98% for numerical calculations
- **Response Time**: <3 seconds for 95% of queries
- **Error Handling**: Graceful handling of 100% of edge cases

### Reporting and Analytics

**Test Report Structure**:
- Executive summary with overall pass/fail rates
- Category-wise performance breakdown
- Individual test case results with detailed analysis
- Performance metrics and response time analysis
- Error analysis and recommendations for improvement
- Trend analysis for regression testing

**Metrics Tracked**:
- Test pass/fail rates by category
- Average response times
- Error frequency and types
- Data accuracy scores
- Coverage metrics for dataset elements