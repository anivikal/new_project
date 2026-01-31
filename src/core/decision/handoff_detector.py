"""
Decision Layer for Handoff Detection.
Monitors confidence, sentiment, and other signals to determine when to escalate to human agent.
Based on the DECISION LAYER (USP) diagram.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import structlog

from src.config import get_settings
from src.models import (
    ConversationMetrics,
    ConversationSession,
    HandoffTrigger,
    Intent,
    NLUResult,
    SentimentLabel,
)

logger = structlog.get_logger(__name__)


class HandoffDecision(NamedTuple):
    """Result of handoff decision evaluation."""
    
    should_handoff: bool
    trigger: HandoffTrigger | None
    confidence: float  # Confidence in the decision
    reason: str
    risk_score: float  # 0-1, higher = more likely to need handoff


@dataclass
class DecisionSignals:
    """Signals used for handoff decision."""
    
    # Confidence signals
    current_confidence: float = 1.0
    min_confidence: float = 1.0
    confidence_trend: float = 0.0  # Negative = declining
    confidence_drops: int = 0  # Number of significant drops
    
    # Sentiment signals
    current_sentiment_score: float = 0.5
    sentiment_trend: float = 0.0
    negative_sentiment_count: int = 0
    frustration_detected: bool = False
    
    # Repetition signals
    clarification_count: int = 0
    same_intent_repeats: int = 0
    unanswered_questions: int = 0
    
    # Domain signals
    out_of_scope_detected: bool = False
    complex_query_detected: bool = False
    unresolved_slots: int = 0
    
    # User signals
    explicit_agent_request: bool = False
    
    # Time signals
    conversation_duration_seconds: float = 0.0
    silence_duration_seconds: float = 0.0


@dataclass
class HandoffPolicy:
    """Configurable policy for handoff decisions."""
    
    # Confidence thresholds
    confidence_threshold: float = 0.5  # Below this, consider handoff
    confidence_drop_threshold: float = 0.2  # Single turn drop threshold
    min_confidence_for_handoff: float = 0.3  # Below this, definitely handoff
    
    # Sentiment thresholds
    negative_sentiment_threshold: float = 0.35
    sentiment_drop_threshold: float = 0.2
    frustration_threshold: float = 0.25
    negative_turns_threshold: int = 2  # Consecutive negative turns
    
    # Repetition thresholds
    max_clarifications: int = 3
    max_same_intent_repeats: int = 2
    
    # Domain thresholds
    max_unresolved_slots: int = 3
    
    # Time thresholds
    max_conversation_duration_seconds: int = 600  # 10 minutes
    
    # Weight factors for risk score
    confidence_weight: float = 0.3
    sentiment_weight: float = 0.3
    repetition_weight: float = 0.2
    domain_weight: float = 0.2


class HandoffDecisionEngine:
    """
    Engine for making handoff decisions based on multiple signals.
    
    Implements the Decision Layer from the architecture:
    - Confidence Trajectory
    - Sentiment Trend
    - Repetition Counter
    - Domain Complexity / Unresolved Slots
    """
    
    def __init__(self, policy: HandoffPolicy | None = None) -> None:
        self.settings = get_settings()
        self.policy = policy or HandoffPolicy(
            confidence_threshold=self.settings.voicebot.handoff_confidence_threshold,
            negative_sentiment_threshold=self.settings.voicebot.sentiment_negative_threshold,
            sentiment_drop_threshold=self.settings.voicebot.sentiment_drop_threshold,
            max_clarifications=self.settings.voicebot.max_repetitions_before_handoff,
        )
    
    def evaluate(
        self,
        session: ConversationSession,
        current_nlu: NLUResult
    ) -> HandoffDecision:
        """
        Evaluate whether to trigger handoff based on conversation state.
        
        Args:
            session: Current conversation session
            current_nlu: NLU result from current turn
        
        Returns:
            HandoffDecision with recommendation
        """
        # Collect signals
        signals = self._collect_signals(session, current_nlu)
        
        # Log signals for debugging
        logger.debug(
            "handoff_signals_collected",
            session_id=str(session.id),
            signals=vars(signals)
        )
        
        # Check explicit triggers first (OR logic)
        explicit_decision = self._check_explicit_triggers(signals)
        if explicit_decision.should_handoff:
            return explicit_decision
        
        # Calculate risk score (weighted combination)
        risk_score = self._calculate_risk_score(signals)
        
        # High risk score triggers handoff
        if risk_score >= 0.7:
            trigger = self._determine_primary_trigger(signals)
            return HandoffDecision(
                should_handoff=True,
                trigger=trigger,
                confidence=risk_score,
                reason=f"High risk score ({risk_score:.2f}) - {trigger.value}",
                risk_score=risk_score
            )
        
        # Medium risk - consider handoff
        if risk_score >= 0.5:
            return HandoffDecision(
                should_handoff=False,
                trigger=None,
                confidence=risk_score,
                reason=f"Medium risk ({risk_score:.2f}) - monitoring",
                risk_score=risk_score
            )
        
        # Low risk - continue
        return HandoffDecision(
            should_handoff=False,
            trigger=None,
            confidence=1.0 - risk_score,
            reason="Conversation proceeding normally",
            risk_score=risk_score
        )
    
    def _collect_signals(
        self,
        session: ConversationSession,
        current_nlu: NLUResult
    ) -> DecisionSignals:
        """Collect all signals from conversation state."""
        metrics = session.metrics
        
        # Count negative sentiment turns
        negative_count = sum(
            1 for score in metrics.sentiment_trajectory
            if score < self.policy.negative_sentiment_threshold
        )
        
        # Count confidence drops
        confidence_drops = 0
        for i in range(1, len(metrics.confidence_trajectory)):
            if metrics.confidence_trajectory[i-1] - metrics.confidence_trajectory[i] > self.policy.confidence_drop_threshold:
                confidence_drops += 1
        
        # Check for same intent repeats
        same_intent_repeats = 0
        if session.current_intent:
            same_intent_repeats = metrics.repeated_intents.get(session.current_intent.value, 0)
        
        # Calculate conversation duration
        duration = (datetime.utcnow() - session.started_at).total_seconds()
        
        return DecisionSignals(
            current_confidence=current_nlu.intent.confidence,
            min_confidence=metrics.min_confidence,
            confidence_trend=metrics.confidence_trend,
            confidence_drops=confidence_drops,
            current_sentiment_score=current_nlu.sentiment.score,
            sentiment_trend=metrics.sentiment_trend,
            negative_sentiment_count=negative_count,
            frustration_detected=current_nlu.sentiment.label in [SentimentLabel.FRUSTRATED, SentimentLabel.CONFUSED],
            clarification_count=metrics.clarification_count,
            same_intent_repeats=same_intent_repeats,
            unanswered_questions=len(metrics.intents_pending),
            out_of_scope_detected=current_nlu.intent.intent == Intent.OUT_OF_SCOPE,
            complex_query_detected=len(current_nlu.entities) > 3,  # Multiple entities suggest complexity
            unresolved_slots=len(metrics.unresolved_slots),
            explicit_agent_request=current_nlu.intent.intent == Intent.HUMAN_AGENT,
            conversation_duration_seconds=duration,
        )
    
    def _check_explicit_triggers(self, signals: DecisionSignals) -> HandoffDecision:
        """Check for explicit handoff triggers (immediate handoff)."""
        
        # User explicitly requests human agent
        if signals.explicit_agent_request:
            return HandoffDecision(
                should_handoff=True,
                trigger=HandoffTrigger.USER_REQUEST,
                confidence=1.0,
                reason="User explicitly requested human agent",
                risk_score=1.0
            )
        
        # Confidence too low
        if signals.current_confidence < self.policy.min_confidence_for_handoff:
            return HandoffDecision(
                should_handoff=True,
                trigger=HandoffTrigger.LOW_CONFIDENCE,
                confidence=0.95,
                reason=f"Confidence critically low: {signals.current_confidence:.2f}",
                risk_score=0.9
            )
        
        # Frustrated user
        if signals.frustration_detected and signals.current_sentiment_score < self.policy.frustration_threshold:
            return HandoffDecision(
                should_handoff=True,
                trigger=HandoffTrigger.NEGATIVE_SENTIMENT,
                confidence=0.9,
                reason="User frustration detected",
                risk_score=0.85
            )
        
        # Too many clarifications
        if signals.clarification_count >= self.policy.max_clarifications:
            return HandoffDecision(
                should_handoff=True,
                trigger=HandoffTrigger.REPEATED_CLARIFICATION,
                confidence=0.85,
                reason=f"Repeated clarifications: {signals.clarification_count}",
                risk_score=0.8
            )
        
        # Conversation timeout
        if signals.conversation_duration_seconds >= self.policy.max_conversation_duration_seconds:
            return HandoffDecision(
                should_handoff=True,
                trigger=HandoffTrigger.TIMEOUT,
                confidence=0.9,
                reason="Conversation duration exceeded limit",
                risk_score=0.75
            )
        
        # Out of scope
        if signals.out_of_scope_detected:
            return HandoffDecision(
                should_handoff=True,
                trigger=HandoffTrigger.COMPLEX_QUERY,
                confidence=0.85,
                reason="Query out of Tier-1 scope",
                risk_score=0.8
            )
        
        # No explicit trigger
        return HandoffDecision(
            should_handoff=False,
            trigger=None,
            confidence=0.0,
            reason="",
            risk_score=0.0
        )
    
    def _calculate_risk_score(self, signals: DecisionSignals) -> float:
        """
        Calculate weighted risk score from all signals.
        
        Score ranges from 0 (no risk) to 1 (definite handoff needed).
        """
        # Confidence component
        confidence_risk = 0.0
        if signals.current_confidence < self.policy.confidence_threshold:
            confidence_risk = (self.policy.confidence_threshold - signals.current_confidence) / self.policy.confidence_threshold
        confidence_risk += signals.confidence_drops * 0.15
        confidence_risk = min(1.0, confidence_risk)
        
        # Sentiment component
        sentiment_risk = 0.0
        if signals.current_sentiment_score < self.policy.negative_sentiment_threshold:
            sentiment_risk = (self.policy.negative_sentiment_threshold - signals.current_sentiment_score) / self.policy.negative_sentiment_threshold
        if signals.frustration_detected:
            sentiment_risk += 0.3
        if signals.sentiment_trend < -self.policy.sentiment_drop_threshold:
            sentiment_risk += 0.2
        sentiment_risk += signals.negative_sentiment_count * 0.15
        sentiment_risk = min(1.0, sentiment_risk)
        
        # Repetition component
        repetition_risk = 0.0
        repetition_risk += signals.clarification_count / self.policy.max_clarifications
        repetition_risk += signals.same_intent_repeats / self.policy.max_same_intent_repeats
        repetition_risk = min(1.0, repetition_risk)
        
        # Domain component
        domain_risk = 0.0
        if signals.out_of_scope_detected:
            domain_risk = 0.8
        if signals.complex_query_detected:
            domain_risk += 0.2
        domain_risk += signals.unresolved_slots / (self.policy.max_unresolved_slots + 1)
        domain_risk = min(1.0, domain_risk)
        
        # Weighted combination
        risk_score = (
            confidence_risk * self.policy.confidence_weight +
            sentiment_risk * self.policy.sentiment_weight +
            repetition_risk * self.policy.repetition_weight +
            domain_risk * self.policy.domain_weight
        )
        
        return min(1.0, risk_score)
    
    def _determine_primary_trigger(self, signals: DecisionSignals) -> HandoffTrigger:
        """Determine the primary trigger from signals."""
        # Order by severity/impact
        if signals.frustration_detected:
            return HandoffTrigger.NEGATIVE_SENTIMENT
        if signals.current_sentiment_score < self.policy.negative_sentiment_threshold:
            return HandoffTrigger.SENTIMENT_DROP
        if signals.current_confidence < self.policy.confidence_threshold:
            return HandoffTrigger.LOW_CONFIDENCE
        if signals.clarification_count >= self.policy.max_clarifications - 1:
            return HandoffTrigger.REPEATED_CLARIFICATION
        if signals.out_of_scope_detected or signals.complex_query_detected:
            return HandoffTrigger.COMPLEX_QUERY
        
        return HandoffTrigger.LOW_CONFIDENCE


class ProactiveHandoffMonitor:
    """
    Monitors conversation for proactive handoff signals.
    Runs in background to predict handoff needs before they become critical.
    """
    
    def __init__(self) -> None:
        self.decision_engine = HandoffDecisionEngine()
        self.warning_threshold = 0.4  # Risk score that triggers warning
    
    def get_proactive_warning(
        self,
        session: ConversationSession,
        current_nlu: NLUResult
    ) -> dict | None:
        """
        Check for proactive warning signals.
        
        Returns warning info if conversation is heading toward handoff.
        """
        decision = self.decision_engine.evaluate(session, current_nlu)
        
        if not decision.should_handoff and decision.risk_score >= self.warning_threshold:
            # Conversation is at risk but not yet critical
            return {
                "warning": True,
                "risk_score": decision.risk_score,
                "reason": decision.reason,
                "recommended_actions": self._get_recommended_actions(decision.risk_score, current_nlu),
            }
        
        return None
    
    def _get_recommended_actions(self, risk_score: float, nlu: NLUResult) -> list[str]:
        """Get recommended actions to prevent escalation."""
        actions = []
        
        if nlu.intent.confidence < 0.7:
            actions.append("Clarify user intent with specific questions")
        
        if nlu.sentiment.label in [SentimentLabel.NEGATIVE, SentimentLabel.FRUSTRATED]:
            actions.append("Acknowledge user's concern empathetically")
        
        if nlu.sentiment.label == SentimentLabel.CONFUSED:
            actions.append("Provide clearer explanation with examples")
        
        if risk_score >= 0.5:
            actions.append("Offer option to connect with human agent")
        
        return actions
