"""
Handoff data models for warm transfer to human agents.
This module defines the structure of handoff alerts and agent briefs.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.conversation import (
    ConversationSession,
    ConversationTurn,
    DriverInfo,
    HandoffTrigger,
    Intent,
    Language,
    SentimentLabel,
    TurnRole,
)


class HandoffPriority(str, Enum):
    """Priority level for handoff queue."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class HandoffStatus(str, Enum):
    """Status of handoff request."""
    
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ActionTaken(BaseModel):
    """Action taken by the bot during conversation."""
    
    action: str
    description: str
    timestamp: datetime
    result: str | None = None
    success: bool = True


class SuggestedAction(BaseModel):
    """Suggested action for the human agent."""
    
    action: str
    description: str
    priority: int = Field(ge=1, le=5, default=3)
    context: dict | None = None


class ConversationExcerpt(BaseModel):
    """Key excerpt from conversation for agent review."""
    
    turn_index: int
    role: TurnRole
    content: str
    timestamp: datetime
    is_key_moment: bool = False
    annotation: str | None = None


class HandoffSummary(BaseModel):
    """Auto-generated summary for agent handoff."""
    
    # Brief overview
    one_line_summary: str
    detailed_summary: str
    
    # Issue classification
    primary_issue: str
    secondary_issues: list[str] = Field(default_factory=list)
    
    # What was discussed
    topics_discussed: list[str] = Field(default_factory=list)
    
    # What the driver is stuck on
    stuck_on: str | None = None
    
    # Actions already taken by bot
    actions_taken: list[ActionTaken] = Field(default_factory=list)
    
    # Suggested next steps
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    
    # Key conversation excerpts
    key_excerpts: list[ConversationExcerpt] = Field(default_factory=list)


class ConfidenceTimeline(BaseModel):
    """Confidence score timeline for agent visibility."""
    
    timestamps: list[datetime] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)  # Notable events at each point
    
    @property
    def trend(self) -> str:
        """Get trend description."""
        if len(self.scores) < 2:
            return "insufficient_data"
        
        avg_first_half = sum(self.scores[:len(self.scores)//2]) / (len(self.scores)//2)
        avg_second_half = sum(self.scores[len(self.scores)//2:]) / (len(self.scores) - len(self.scores)//2)
        
        diff = avg_second_half - avg_first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"


class AgentMicroBrief(BaseModel):
    """
    Micro-brief displayed to agent when handoff is triggered.
    This is the key information agent sees at a glance.
    Based on the AGENT MICRO-BRIEF diagram.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # Driver info
    driver_name: str | None = None
    driver_phone_last_4: str
    driver_city: str | None = None
    driver_language: Language
    
    # Top entities (key data points)
    top_entities: dict[str, str] = Field(default_factory=dict)  # e.g., {"invoice_number": "INV123", "amount": "₹45"}
    
    # Actionable summary
    actionable_summary: str
    
    # Confidence timeline
    confidence_timeline: ConfidenceTimeline
    
    # Escalation reason
    escalation_reason: HandoffTrigger
    escalation_description: str
    
    # Suggested actions for quick resolution
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    
    # Key audio excerpt reference (timestamp in call)
    key_audio_timestamp_start: float | None = None  # seconds into call
    key_audio_timestamp_end: float | None = None
    key_audio_transcript: str | None = None
    
    # Sentiment
    current_sentiment: SentimentLabel
    sentiment_score: float


class HandoffAlert(BaseModel):
    """
    Complete handoff alert sent to agent/queue system.
    Based on the HANDOFF ALERT diagram in the requirements.
    """
    
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    call_id: str
    
    # Trigger info
    trigger: HandoffTrigger
    trigger_description: str
    
    # Driver info
    driver_info: DriverInfo
    
    # Conversation context
    intent_history: list[Intent] = Field(default_factory=list)
    current_intent: Intent | None = None
    
    # Sentiment
    sentiment: SentimentLabel
    sentiment_score: float = Field(ge=0.0, le=1.0)
    
    # Summary
    issue_summary: str
    detailed_summary: HandoffSummary
    
    # Actions
    actions_taken_by_bot: list[ActionTaken] = Field(default_factory=list)
    next_steps_for_agent: list[SuggestedAction] = Field(default_factory=list)
    
    # Agent brief
    micro_brief: AgentMicroBrief
    
    # Full conversation (for reference)
    conversation_turns: list[ConversationTurn] = Field(default_factory=list)
    
    # Status and assignment
    status: HandoffStatus = HandoffStatus.PENDING
    priority: HandoffPriority = HandoffPriority.MEDIUM
    assigned_agent_id: str | None = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_at: datetime | None = None
    resolved_at: datetime | None = None
    
    # Queue info
    queue_position: int | None = None
    estimated_wait_seconds: int | None = None

    @classmethod
    def from_conversation(
        cls,
        session: ConversationSession,
        trigger: HandoffTrigger,
        summary: HandoffSummary,
    ) -> "HandoffAlert":
        """Create handoff alert from conversation session."""
        
        # Determine priority based on trigger and metrics
        priority = HandoffPriority.MEDIUM
        if trigger == HandoffTrigger.USER_REQUEST:
            priority = HandoffPriority.HIGH
        elif trigger == HandoffTrigger.NEGATIVE_SENTIMENT:
            priority = HandoffPriority.HIGH
        elif session.metrics.average_sentiment < 0.3:
            priority = HandoffPriority.URGENT
        
        # Get current sentiment
        current_sentiment = SentimentLabel.NEUTRAL
        sentiment_score = 0.5
        if session.turns:
            last_user_turn = next(
                (t for t in reversed(session.turns) if t.role == TurnRole.USER and t.nlu_result),
                None
            )
            if last_user_turn and last_user_turn.nlu_result:
                current_sentiment = last_user_turn.nlu_result.sentiment.label
                sentiment_score = last_user_turn.nlu_result.sentiment.score
        
        # Build confidence timeline
        confidence_timeline = ConfidenceTimeline(
            timestamps=[t.timestamp for t in session.turns if t.role == TurnRole.USER],
            scores=session.metrics.confidence_trajectory,
            events=[]
        )
        
        # Get top entities from context
        top_entities = {}
        for key, value in session.filled_slots.items():
            if isinstance(value, (str, int, float)):
                top_entities[key] = str(value)
        
        # Create micro brief
        micro_brief = AgentMicroBrief(
            driver_name=session.driver.name,
            driver_phone_last_4=session.driver.phone_number[-4:],
            driver_city=session.driver.city,
            driver_language=session.driver.preferred_language,
            top_entities=top_entities,
            actionable_summary=summary.one_line_summary,
            confidence_timeline=confidence_timeline,
            escalation_reason=trigger,
            escalation_description=cls._get_trigger_description(trigger, session),
            suggested_actions=summary.suggested_actions[:3],  # Top 3
            current_sentiment=current_sentiment,
            sentiment_score=sentiment_score,
        )
        
        # Get intent history
        intent_history = []
        for turn in session.turns:
            if turn.role == TurnRole.USER and turn.nlu_result:
                intent_history.append(turn.nlu_result.intent.intent)
        
        return cls(
            conversation_id=session.id,
            call_id=session.call_id,
            trigger=trigger,
            trigger_description=cls._get_trigger_description(trigger, session),
            driver_info=session.driver,
            intent_history=intent_history,
            current_intent=session.current_intent,
            sentiment=current_sentiment,
            sentiment_score=sentiment_score,
            issue_summary=summary.one_line_summary,
            detailed_summary=summary,
            actions_taken_by_bot=summary.actions_taken,
            next_steps_for_agent=summary.suggested_actions,
            micro_brief=micro_brief,
            conversation_turns=session.turns,
            priority=priority,
        )
    
    @staticmethod
    def _get_trigger_description(trigger: HandoffTrigger, session: ConversationSession) -> str:
        """Generate human-readable trigger description."""
        descriptions = {
            HandoffTrigger.LOW_CONFIDENCE: f"Bot confidence dropped to {session.metrics.min_confidence:.0%}",
            HandoffTrigger.NEGATIVE_SENTIMENT: f"Driver sentiment is negative (score: {session.metrics.average_sentiment:.2f})",
            HandoffTrigger.SENTIMENT_DROP: f"Sentiment dropped significantly during conversation",
            HandoffTrigger.USER_REQUEST: "Driver explicitly requested human agent",
            HandoffTrigger.REPEATED_CLARIFICATION: f"Clarification requested {session.metrics.clarification_count} times",
            HandoffTrigger.COMPLEX_QUERY: "Query complexity exceeds Tier-1 scope",
            HandoffTrigger.POLICY_VIOLATION: "Policy-sensitive topic detected",
            HandoffTrigger.TIMEOUT: "Conversation timeout reached",
        }
        return descriptions.get(trigger, "Handoff triggered")
