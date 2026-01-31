"""
Conversation and session data models.
These models represent the core data structures for managing voicebot conversations.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages."""
    
    HINDI = "hi-IN"
    ENGLISH_INDIA = "en-IN"
    HINGLISH = "hi-en"  # Code-mixed Hindi-English
    BENGALI = "bn-IN"
    TAMIL = "ta-IN"


class Intent(str, Enum):
    """Tier-1 supported intents for Battery Smart."""
    
    # Swap related
    SWAP_HISTORY = "swap_history"
    SWAP_INVOICE = "swap_invoice"
    INVOICE_EXPLANATION = "invoice_explanation"
    
    # Station related
    NEAREST_STATION = "nearest_station"
    STATION_AVAILABILITY = "station_availability"
    STATION_DIRECTIONS = "station_directions"
    
    # Subscription related
    SUBSCRIPTION_STATUS = "subscription_status"
    SUBSCRIPTION_RENEWAL = "subscription_renewal"
    SUBSCRIPTION_PRICING = "subscription_pricing"
    PLAN_COMPARISON = "plan_comparison"
    
    # Account related
    LEAVE_INFORMATION = "leave_information"
    DSK_ACTIVATION = "dsk_activation"
    PROFILE_UPDATE = "profile_update"
    
    # General
    GREETING = "greeting"
    GOODBYE = "goodbye"
    HELP = "help"
    HUMAN_AGENT = "human_agent"  # Explicit request for human
    UNKNOWN = "unknown"
    OUT_OF_SCOPE = "out_of_scope"


class EntityType(str, Enum):
    """Entity types extracted from user utterances."""
    
    PHONE_NUMBER = "phone_number"
    DRIVER_ID = "driver_id"
    DATE = "date"
    DATE_RANGE = "date_range"
    LOCATION = "location"
    STATION_NAME = "station_name"
    SUBSCRIPTION_PLAN = "subscription_plan"
    AMOUNT = "amount"
    INVOICE_NUMBER = "invoice_number"
    BATTERY_ID = "battery_id"


class Entity(BaseModel):
    """Extracted entity from user utterance."""
    
    type: EntityType
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_pos: int | None = None
    end_pos: int | None = None
    normalized_value: Any | None = None  # Normalized/parsed value


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"


class Sentiment(BaseModel):
    """Sentiment analysis result."""
    
    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Sentiment score (0=very negative, 1=very positive)")
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: list[str] = Field(default_factory=list, description="Detected sentiment indicators")


class IntentClassification(BaseModel):
    """Intent classification result."""
    
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    sub_intent: str | None = None
    alternative_intents: list[tuple[Intent, float]] = Field(default_factory=list)


class NLUResult(BaseModel):
    """Complete NLU processing result."""
    
    intent: IntentClassification
    entities: list[Entity] = Field(default_factory=list)
    sentiment: Sentiment
    detected_language: Language
    original_text: str
    normalized_text: str  # Normalized/cleaned text
    is_code_mixed: bool = False  # Hindi-English mixing detected


class TurnRole(str, Enum):
    """Role in conversation turn."""
    
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


class ConversationTurn(BaseModel):
    """Single turn in conversation."""
    
    id: UUID = Field(default_factory=uuid4)
    role: TurnRole
    content: str
    audio_url: str | None = None  # S3 URL for audio
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # NLU results (for user turns)
    nlu_result: NLUResult | None = None
    
    # Bot response metadata
    response_source: str | None = None  # "rag", "tool", "fallback"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    
    # Metrics
    latency_ms: int | None = None
    confidence: float | None = None


class HandoffTrigger(str, Enum):
    """Reasons for triggering handoff."""
    
    LOW_CONFIDENCE = "low_confidence"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    SENTIMENT_DROP = "sentiment_drop"
    USER_REQUEST = "user_request"
    REPEATED_CLARIFICATION = "repeated_clarification"
    COMPLEX_QUERY = "complex_query"
    POLICY_VIOLATION = "policy_violation"
    TIMEOUT = "timeout"


class ConversationState(str, Enum):
    """Conversation lifecycle states."""
    
    ACTIVE = "active"
    AWAITING_INPUT = "awaiting_input"
    PROCESSING = "processing"
    HANDOFF_PENDING = "handoff_pending"
    HANDED_OFF = "handed_off"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ConversationMetrics(BaseModel):
    """Real-time conversation metrics for decision making."""
    
    # Confidence tracking
    confidence_trajectory: list[float] = Field(default_factory=list)
    average_confidence: float = 0.0
    min_confidence: float = 1.0
    confidence_trend: float = 0.0  # Positive = improving, negative = declining
    
    # Sentiment tracking
    sentiment_trajectory: list[float] = Field(default_factory=list)
    average_sentiment: float = 0.5
    sentiment_trend: float = 0.0
    
    # Repetition tracking
    clarification_count: int = 0
    repeated_intents: dict[str, int] = Field(default_factory=dict)
    unresolved_slots: list[str] = Field(default_factory=list)
    
    # Timing
    total_duration_seconds: float = 0.0
    average_response_latency_ms: float = 0.0
    
    # Resolution tracking
    intents_resolved: list[str] = Field(default_factory=list)
    intents_pending: list[str] = Field(default_factory=list)


class DriverInfo(BaseModel):
    """Driver/caller information."""
    
    driver_id: str | None = None
    phone_number: str
    name: str | None = None
    city: str | None = None
    preferred_language: Language = Language.HINGLISH
    subscription_plan: str | None = None
    is_verified: bool = False


class ConversationSession(BaseModel):
    """Complete conversation session."""
    
    id: UUID = Field(default_factory=uuid4)
    call_id: str  # Telephony call ID
    
    # Participant info
    driver: DriverInfo
    
    # State
    state: ConversationState = ConversationState.ACTIVE
    current_intent: Intent | None = None
    
    # Conversation history
    turns: list[ConversationTurn] = Field(default_factory=list)
    
    # Context/slots filled
    context: dict[str, Any] = Field(default_factory=dict)
    filled_slots: dict[str, Any] = Field(default_factory=dict)
    
    # Metrics
    metrics: ConversationMetrics = Field(default_factory=ConversationMetrics)
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    
    # Handoff
    handoff_triggered: bool = False
    handoff_trigger: HandoffTrigger | None = None
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn and update metrics."""
        self.turns.append(turn)
        
        if turn.role == TurnRole.USER and turn.nlu_result:
            # Update confidence trajectory
            self.metrics.confidence_trajectory.append(turn.nlu_result.intent.confidence)
            self.metrics.average_confidence = sum(self.metrics.confidence_trajectory) / len(
                self.metrics.confidence_trajectory
            )
            self.metrics.min_confidence = min(self.metrics.confidence_trajectory)
            
            # Update sentiment trajectory
            self.metrics.sentiment_trajectory.append(turn.nlu_result.sentiment.score)
            self.metrics.average_sentiment = sum(self.metrics.sentiment_trajectory) / len(
                self.metrics.sentiment_trajectory
            )
            
            # Calculate trends (using last 3 turns)
            if len(self.metrics.confidence_trajectory) >= 2:
                recent = self.metrics.confidence_trajectory[-3:]
                self.metrics.confidence_trend = recent[-1] - recent[0]
            
            if len(self.metrics.sentiment_trajectory) >= 2:
                recent = self.metrics.sentiment_trajectory[-3:]
                self.metrics.sentiment_trend = recent[-1] - recent[0]
    
    def get_conversation_summary(self) -> str:
        """Generate a brief conversation summary."""
        user_turns = [t for t in self.turns if t.role == TurnRole.USER]
        intents = [t.nlu_result.intent.intent.value for t in user_turns if t.nlu_result]
        
        return f"Conversation with {len(self.turns)} turns. " \
               f"Intents: {', '.join(set(intents))}. " \
               f"Avg confidence: {self.metrics.average_confidence:.2f}. " \
               f"Avg sentiment: {self.metrics.average_sentiment:.2f}."
