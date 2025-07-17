# /app/models.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Tuple
from bson import ObjectId
import datetime as dt
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, *args, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class Company(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str
    domain: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True


class User(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    full_name: str
    email: EmailStr
    company_id: Optional[PyObjectId] = None
    preferences: Dict = Field(default_factory=dict)
    scheduled_events: List[str] = Field(default_factory=list) # To store event URLs
    # NEW: Smart Recommendations fields
    recommendation_preferences: Dict[str, Any] = Field(default_factory=dict)
    last_recommendation_date: Optional[datetime] = None
    recommendation_history: List[str] = Field(default_factory=list)

    # Add appointment-related fields
    appointment_history: List[str] = Field(default_factory=list)  # List of appointment session IDs
    appointment_preferences: Dict[str, Any] = Field(default_factory=dict)  # User's appointment preferences

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

class ChatMessage(BaseModel):
    role: str
    content: str

class ConversationHistory(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_email: EmailStr
    messages: List[ChatMessage]
    tags: List[str] = Field(default_factory=list)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)

class ScheduleDetails(BaseModel):
    address: str
    time: str

class ChatResponse(BaseModel):
    answer: str
    schedule_details: Optional[ScheduleDetails] = None

class DocumentUploadRequest(BaseModel):
    user_id: str
    filenames: List[str] = Field(default_factory=list)

class ScheduleRequest(BaseModel):
    email: str
    address: str
    time: str

# NEW: Smart Property Recommendations Models

class ConversationState(str, Enum):
    """States of the recommendation conversation workflow"""
    INITIATED = "initiated"
    GATHERING_PREFERENCES = "gathering_preferences"
    CLARIFYING_DETAILS = "clarifying_details"
    GENERATING_RECOMMENDATIONS = "generating_recommendations"
    COMPLETED = "completed"
    FAILED = "failed"

class RecommendationIntent(BaseModel):
    """Detected intent from user message for recommendations"""
    is_recommendation_request: bool
    confidence: float = Field(ge=0.0, le=1.0)
    initial_preferences: Dict[str, Any] = Field(default_factory=dict)
    trigger_phrases: List[str] = Field(default_factory=list)

class UserContext(BaseModel):
    """User context for personalized recommendations"""
    user_id: str
    historical_preferences: Dict[str, Any] = Field(default_factory=dict)
    budget_range: Optional[Tuple[int, int]] = None
    preferred_locations: List[str] = Field(default_factory=list)
    required_features: List[str] = Field(default_factory=list)
    excluded_features: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)

class ConversationSession(BaseModel):
    """Conversation session state for recommendation workflow"""
    session_id: str
    user_id: str
    state: ConversationState
    collected_preferences: Dict[str, Any] = Field(default_factory=dict)
    questions_asked: List[str] = Field(default_factory=list)
    responses_received: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class PropertyRecommendation(BaseModel):
    """Individual property recommendation with explanation"""
    property_id: str
    property_data: Dict[str, Any] = Field(default_factory=dict)
    match_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    matching_criteria: List[str] = Field(default_factory=list)

class RecommendationResult(BaseModel):
    """Final result from recommendation workflow"""
    session_id: str
    recommendations: List[PropertyRecommendation] = Field(default_factory=list)
    user_context: UserContext
    conversation_summary: str
    total_properties_considered: int = 0
    recommendations_generated: int = 0

class WorkflowSession(BaseModel):
    """Workflow session management"""
    session_id: str
    user_id: str
    current_step: str
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class WorkflowStep(BaseModel):
    """Individual workflow step result"""
    step_name: str
    success: bool
    response_message: str
    next_step: Optional[str] = None
    collected_data: Dict[str, Any] = Field(default_factory=dict)

# Appointment Booking Models
from enum import Enum
from typing import Optional

class AppointmentStatus(str, Enum):
    """Status of an appointment booking session."""
    PENDING = "pending"
    COLLECTING_INFO = "collecting_info"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

@dataclass
class AppointmentIntent:
    """Intent detection result for appointment booking requests."""
    is_appointment_request: bool
    confidence: float
    extracted_details: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    intent_type: str = "appointment_booking"  # specific appointment intent

@dataclass
class AppointmentData:
    """Complete appointment information."""
    title: str
    location: str
    date: datetime
    duration_minutes: int = 60
    attendee_emails: List[str] = field(default_factory=list)
    description: Optional[str] = None
    meet_link: Optional[str] = None
    calendar_event_id: Optional[str] = None
    organizer_email: Optional[str] = None
    confirmation_response: Optional[str] = None  # Track confirmation responses

@dataclass
class AppointmentSession:
    """Session state for appointment booking workflow."""
    session_id: str
    user_id: str
    status: AppointmentStatus
    collected_data: AppointmentData
    required_fields: List[str] = field(default_factory=lambda: ["title", "location", "date", "attendee_emails"])
    missing_fields: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ConfirmationUI:
    """UI configuration for appointment confirmation display."""
    appointment_card: Dict[str, Any] = field(default_factory=dict)
    action_buttons: List[Dict[str, Any]] = field(default_factory=list)
    styling: Dict[str, Any] = field(default_factory=dict)
    animations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AppointmentError:
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None

# ================================
# Performance Optimization Models
# ================================

class MessageType(str, Enum):
    """Types of user messages for classification."""
    GREETING = "greeting"
    THANK_YOU = "thank_you"
    HELP_REQUEST = "help_request"
    CONVERSATIONAL = "conversational"
    PROPERTY_SEARCH = "property_search"
    DIRECT_PROPERTY_QUERY = "direct_property_query"  # Simple queries like "top 3 cheap properties"
    APPOINTMENT_REQUEST = "appointment_request"
    UNKNOWN = "unknown"

class ProcessingStrategy(str, Enum):
    QUICK_RESPONSE = "quick_response"
    DIRECT_SEARCH = "direct_search"  # For direct property queries like "top 3 properties"
    PROPERTY_WORKFLOW = "property_workflow"
    APPOINTMENT_WORKFLOW = "appointment_workflow"
    FALLBACK_RESPONSE = "fallback_response"

@dataclass
class MessageClassification:
    """Result of message classification with confidence and processing strategy."""
    message_type: MessageType
    confidence: float
    processing_strategy: ProcessingStrategy
    estimated_response_time: float
    requires_index: bool
    detected_patterns: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None

@dataclass
class PerformanceMetrics:
    timestamp: dt.datetime
    operation_type: str
    duration_ms: float
    success: bool
    user_id: Optional[str] = None
    collection_name: Optional[str] = None
    query_pattern: Optional[str] = None
    resource_usage: Optional[Dict[str, float]] = None
    error_details: Optional[str] = None

    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}

@dataclass
class ConversationalContext:
    """Context for personalized conversational responses."""
    user_id: str
    user_name: Optional[str] = None
    previous_interactions: int = 0
    last_interaction_time: Optional[datetime] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_stage: str = "initial"

@dataclass
class WorkflowResponse:
    """Response from appointment workflow operations."""
    success: bool
    message: str
    session_id: Optional[str] = None
    step_name: str = ""
    next_step: Optional[str] = None
    ui_components: Optional[ConfirmationUI] = None
    appointment_data: Optional[AppointmentData] = None
    error_details: Optional[AppointmentError] = None

# MongoDB RAG Optimization Data Models
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import datetime as dt

# Performance and Health Monitoring Models
class DatabaseHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class PerformanceLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class DatabaseStatus:
    connection_available: bool
    ping_time_ms: float
    last_check: dt.datetime
    error_message: Optional[str] = None
    server_info: Optional[Dict[str, Any]] = None

@dataclass
class VectorStoreStatus:
    collections_healthy: bool
    total_collections: int
    storage_size_mb: float
    last_check: dt.datetime
    error_message: Optional[str] = None

@dataclass
class ConnectionPoolStatus:
    active_connections: int
    max_connections: int
    waiting_connections: int
    pool_utilization: float
    connection_errors: int

@dataclass
class HealthIssue:
    severity: str
    category: str
    message: str
    timestamp: dt.datetime
    suggested_action: Optional[str] = None

@dataclass
class HealthStatus:
    overall_status: DatabaseHealthStatus
    mongodb_status: DatabaseStatus
    chromadb_status: VectorStoreStatus
    connection_pool_status: ConnectionPoolStatus
    last_check: dt.datetime
    issues: List[HealthIssue]
    recommendations: List[str]

@dataclass
class OptimizationResult:
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvements: Dict[str, float]
    applied_optimizations: List[str]
    recommendations: List[str]
    timestamp: dt.datetime

@dataclass
class QueryPerformanceReport:
    total_queries: int
    avg_duration_ms: float
    slow_queries_count: int
    slow_query_threshold_ms: int
    query_patterns: Dict[str, int]
    recommendations: List[str]
    report_period: str

@dataclass
class CollectionStatsReport:
    collection_name: str
    document_count: int
    storage_size_mb: float
    avg_object_size_bytes: float
    index_count: int
    index_size_mb: float
    last_updated: dt.datetime

@dataclass
class SlowQuery:
    query_pattern: str
    avg_duration_ms: float
    execution_count: int
    last_seen: dt.datetime
    suggested_indexes: List[str]

@dataclass
class IndexOptimizationResult:
    collection_name: str
    created_indexes: List[str]
    dropped_indexes: List[str]
    optimization_suggestions: List[str]
    performance_impact: Dict[str, float]
    timestamp: dt.datetime

# RAG Performance Optimization Models
@dataclass
class ValidationResult:
    is_valid: bool
    issues_found: List[str]
    validation_time_ms: float
    recommendations: List[str]
    timestamp: dt.datetime

@dataclass
class StructureOptimizationResult:
    collection_name: str
    optimizations_applied: List[str]
    before_performance: Dict[str, float]
    after_performance: Dict[str, float]
    recommendations: List[str]
    timestamp: dt.datetime

@dataclass
class BenchmarkResult:
    operation_type: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_per_sec: float
    success_rate: float
    benchmark_duration_sec: float
    timestamp: dt.datetime

@dataclass
class CleanupResult:
    collections_cleaned: int
    documents_removed: int
    storage_freed_mb: float
    cleanup_duration_sec: float
    errors_encountered: List[str]
    timestamp: dt.datetime

# Concurrent Operations Management Models
class RequestPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QueryRequest:
    user_id: str
    query: str
    priority: RequestPriority
    timestamp: dt.datetime
    timeout_ms: int = 30000

@dataclass
class QueryResult:
    request_id: str
    success: bool
    duration_ms: float
    result_data: Optional[Any]
    error_message: Optional[str]
    timestamp: dt.datetime

@dataclass
class QueuePosition:
    position: int
    estimated_wait_time_ms: float
    queue_size: int
    priority: RequestPriority

@dataclass
class BalancedRequest:
    original_request: QueryRequest
    assigned_connection: str
    load_factor: float
    estimated_processing_time_ms: float

@dataclass
class ResourceUsageReport:
    cpu_usage_percent: float
    memory_usage_mb: float
    connection_pool_usage: float
    disk_io_ops_per_sec: float
    network_bandwidth_mbps: float
    timestamp: dt.datetime

# Security and Compliance Models
class SecurityStatus(str, Enum):
    SECURE = "secure"
    WARNING = "warning"
    VULNERABLE = "vulnerable"
    UNKNOWN = "unknown"

@dataclass
class EncryptedData:
    encrypted_content: str
    encryption_algorithm: str
    key_id: str
    timestamp: dt.datetime

@dataclass
class AccessAuditReport:
    total_access_attempts: int
    successful_accesses: int
    failed_accesses: int
    unique_users: int
    suspicious_activities: List[str]
    report_period: str
    timestamp: dt.datetime

@dataclass
class AnomalyReport:
    anomaly_type: str
    severity: str
    description: str
    affected_resources: List[str]
    detection_timestamp: dt.datetime
    recommended_actions: List[str]

@dataclass
class ComplianceReport:
    compliance_standard: str
    overall_score: float
    passed_checks: int
    failed_checks: int
    recommendations: List[str]
    next_review_date: dt.datetime
    timestamp: dt.datetime

# Recovery and Error Handling Models
@dataclass
class RecoveryResult:
    recovery_successful: bool
    recovery_actions_taken: List[str]
    recovery_time_ms: float
    remaining_issues: List[str]
    timestamp: dt.datetime

@dataclass
class TimeoutResult:
    query_optimized: bool
    timeout_handled: bool
    suggested_optimizations: List[str]
    new_timeout_ms: int
    timestamp: dt.datetime

@dataclass
class IndexRepairResult:
    indexes_repaired: List[str]
    indexes_recreated: List[str]
    repair_successful: bool
    repair_time_ms: float
    timestamp: dt.datetime

@dataclass
class DiskSpaceResult:
    cleanup_performed: bool
    space_freed_mb: float
    remaining_space_mb: float
    critical_threshold_reached: bool
    recommendations: List[str]
    timestamp: dt.datetime

@dataclass
class RepairResult:
    repair_successful: bool
    collections_repaired: List[str]
    data_integrity_restored: bool
    repair_duration_sec: float
    timestamp: dt.datetime

@dataclass
class ConsistencyResult:
    consistency_restored: bool
    embeddings_fixed: int
    missing_vectors_added: int
    inconsistencies_found: List[str]
    timestamp: dt.datetime

@dataclass
class StorageResult:
    storage_issue_resolved: bool
    backup_created: bool
    data_migrated: bool
    storage_optimized: bool
    timestamp: dt.datetime