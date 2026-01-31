"""
Handoff Summary Generator.
Generates comprehensive summaries for human agents during warm handoff.
"""

import json
import re
from datetime import datetime

import structlog

from src.config import get_settings
from src.models import (
    ActionTaken,
    ConversationExcerpt,
    ConversationSession,
    ConversationTurn,
    HandoffSummary,
    HandoffTrigger,
    Intent,
    SentimentLabel,
    SuggestedAction,
    TurnRole,
)

logger = structlog.get_logger(__name__)


class SummaryGenerator:
    """
    Generates human-readable summaries for agent handoff.
    Uses LLM for intelligent summarization.
    """
    
    SUMMARY_PROMPT = """You are summarizing a customer support conversation for a human agent at Battery Smart (EV battery swapping company).
The driver/rider was talking to an AI voicebot in Hindi/English mix.

Generate a concise handoff summary for the support agent.

CONVERSATION:
{conversation}

DRIVER INFO:
- Phone (last 4): {phone_last_4}
- City: {city}
- Language: {language}

METRICS:
- Total turns: {total_turns}
- Average confidence: {avg_confidence:.2%}
- Average sentiment: {avg_sentiment:.2f} (0=negative, 1=positive)
- Handoff trigger: {trigger}

Generate a JSON summary with:
{{
    "one_line_summary": "<50 words max, what's the driver's issue>",
    "detailed_summary": "<100 words max, full context>",
    "primary_issue": "<main issue category>",
    "secondary_issues": ["<other topics discussed>"],
    "stuck_on": "<what specifically is the driver stuck on, if any>",
    "suggested_actions": [
        {{"action": "<action name>", "description": "<what agent should do>", "priority": <1-5>}}
    ],
    "key_moments": [<turn indices of important moments>]
}}

Be concise and actionable. Focus on what the agent needs to know to help quickly."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
    
    async def _get_client(self):
        """Get or create Bedrock client."""
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            self._client = await session.client(
                "bedrock-runtime",
                region_name=self.settings.aws.region
            ).__aenter__()
        return self._client
    
    async def generate_summary(
        self,
        session: ConversationSession,
        trigger: HandoffTrigger
    ) -> HandoffSummary:
        """
        Generate comprehensive handoff summary.
        
        Args:
            session: The conversation session
            trigger: What triggered the handoff
        
        Returns:
            HandoffSummary with all details for agent
        """
        # Format conversation for prompt
        conversation_text = self._format_conversation(session.turns)
        
        # Try LLM summarization
        try:
            llm_summary = await self._generate_llm_summary(session, trigger, conversation_text)
            if llm_summary:
                return llm_summary
        except Exception as e:
            logger.warning("llm_summary_failed", error=str(e))
        
        # Fallback to rule-based summarization
        return self._generate_rule_based_summary(session, trigger)
    
    async def _generate_llm_summary(
        self,
        session: ConversationSession,
        trigger: HandoffTrigger,
        conversation_text: str
    ) -> HandoffSummary | None:
        """Generate summary using LLM."""
        client = await self._get_client()
        
        prompt = self.SUMMARY_PROMPT.format(
            conversation=conversation_text,
            phone_last_4=session.driver.phone_number[-4:],
            city=session.driver.city or "Unknown",
            language=session.driver.preferred_language.value,
            total_turns=len(session.turns),
            avg_confidence=session.metrics.average_confidence,
            avg_sentiment=session.metrics.average_sentiment,
            trigger=trigger.value
        )
        
        response = await client.invoke_model(
            modelId=self.settings.aws.bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.3,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        
        response_body = json.loads(response["body"].read())
        result_text = response_body["content"][0]["text"]
        
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            return None
        
        result = json.loads(json_match.group())
        
        # Build actions taken
        actions_taken = self._extract_actions_taken(session)
        
        # Build suggested actions
        suggested_actions = [
            SuggestedAction(
                action=sa["action"],
                description=sa["description"],
                priority=sa.get("priority", 3)
            )
            for sa in result.get("suggested_actions", [])
        ]
        
        # Build key excerpts
        key_moments = result.get("key_moments", [])
        excerpts = self._extract_key_excerpts(session.turns, key_moments)
        
        return HandoffSummary(
            one_line_summary=result.get("one_line_summary", "Driver needs assistance"),
            detailed_summary=result.get("detailed_summary", ""),
            primary_issue=result.get("primary_issue", "general_query"),
            secondary_issues=result.get("secondary_issues", []),
            stuck_on=result.get("stuck_on"),
            topics_discussed=self._extract_topics(session),
            actions_taken=actions_taken,
            suggested_actions=suggested_actions,
            key_excerpts=excerpts
        )
    
    def _generate_rule_based_summary(
        self,
        session: ConversationSession,
        trigger: HandoffTrigger
    ) -> HandoffSummary:
        """Generate summary using rules when LLM fails."""
        # Extract intents discussed
        intents = []
        for turn in session.turns:
            if turn.role == TurnRole.USER and turn.nlu_result:
                intents.append(turn.nlu_result.intent.intent)
        
        unique_intents = list(set(intents))
        
        # Determine primary issue
        primary_intent = unique_intents[0] if unique_intents else Intent.UNKNOWN
        primary_issue = self._intent_to_issue(primary_intent)
        
        # Generate one-line summary
        one_line = self._generate_one_liner(session, trigger, primary_issue)
        
        # Generate detailed summary
        detailed = self._generate_detailed_summary(session, trigger, unique_intents)
        
        # Extract actions taken
        actions_taken = self._extract_actions_taken(session)
        
        # Generate suggested actions based on trigger
        suggested_actions = self._generate_suggested_actions(trigger, session)
        
        # Extract key excerpts
        key_indices = self._identify_key_moments(session)
        excerpts = self._extract_key_excerpts(session.turns, key_indices)
        
        return HandoffSummary(
            one_line_summary=one_line,
            detailed_summary=detailed,
            primary_issue=primary_issue,
            secondary_issues=[self._intent_to_issue(i) for i in unique_intents[1:3]],
            stuck_on=self._identify_stuck_point(session),
            topics_discussed=[i.value for i in unique_intents],
            actions_taken=actions_taken,
            suggested_actions=suggested_actions,
            key_excerpts=excerpts
        )
    
    def _format_conversation(self, turns: list[ConversationTurn]) -> str:
        """Format conversation turns for prompt."""
        lines = []
        for i, turn in enumerate(turns):
            role = "Driver" if turn.role == TurnRole.USER else "Bot"
            lines.append(f"[Turn {i+1}] {role}: {turn.content}")
            
            if turn.role == TurnRole.USER and turn.nlu_result:
                lines.append(f"  -> Intent: {turn.nlu_result.intent.intent.value} "
                           f"(confidence: {turn.nlu_result.intent.confidence:.0%})")
                lines.append(f"  -> Sentiment: {turn.nlu_result.sentiment.label.value} "
                           f"(score: {turn.nlu_result.sentiment.score:.2f})")
        
        return "\n".join(lines)
    
    def _intent_to_issue(self, intent: Intent) -> str:
        """Map intent to human-readable issue category."""
        mapping = {
            Intent.SWAP_HISTORY: "Swap history inquiry",
            Intent.SWAP_INVOICE: "Invoice request",
            Intent.INVOICE_EXPLANATION: "Invoice/billing confusion",
            Intent.NEAREST_STATION: "Station location",
            Intent.STATION_AVAILABILITY: "Battery availability",
            Intent.SUBSCRIPTION_STATUS: "Subscription status",
            Intent.SUBSCRIPTION_RENEWAL: "Subscription renewal",
            Intent.SUBSCRIPTION_PRICING: "Pricing inquiry",
            Intent.PLAN_COMPARISON: "Plan comparison",
            Intent.LEAVE_INFORMATION: "Leave/pause request",
            Intent.DSK_ACTIVATION: "Account activation",
            Intent.HUMAN_AGENT: "Requested human agent",
            Intent.UNKNOWN: "Unclear query",
            Intent.OUT_OF_SCOPE: "Out of scope query",
        }
        return mapping.get(intent, intent.value)
    
    def _generate_one_liner(
        self,
        session: ConversationSession,
        trigger: HandoffTrigger,
        primary_issue: str
    ) -> str:
        """Generate one-line summary."""
        trigger_context = {
            HandoffTrigger.LOW_CONFIDENCE: "Bot couldn't understand",
            HandoffTrigger.NEGATIVE_SENTIMENT: "Driver frustrated",
            HandoffTrigger.SENTIMENT_DROP: "Driver becoming upset",
            HandoffTrigger.USER_REQUEST: "Driver requested human",
            HandoffTrigger.REPEATED_CLARIFICATION: "Multiple clarifications needed",
            HandoffTrigger.COMPLEX_QUERY: "Complex query",
            HandoffTrigger.TIMEOUT: "Conversation timeout",
        }
        
        context = trigger_context.get(trigger, "Escalation needed")
        
        return f"{context} - {primary_issue}. Avg sentiment: {session.metrics.average_sentiment:.2f}"
    
    def _generate_detailed_summary(
        self,
        session: ConversationSession,
        trigger: HandoffTrigger,
        intents: list[Intent]
    ) -> str:
        """Generate detailed summary."""
        parts = []
        
        # Conversation overview
        parts.append(f"Conversation had {len(session.turns)} turns over "
                    f"{(datetime.utcnow() - session.started_at).seconds} seconds.")
        
        # Topics discussed
        if intents:
            topics = [self._intent_to_issue(i) for i in intents[:3]]
            parts.append(f"Topics: {', '.join(topics)}.")
        
        # Sentiment journey
        if session.metrics.sentiment_trajectory:
            start_sentiment = session.metrics.sentiment_trajectory[0]
            end_sentiment = session.metrics.sentiment_trajectory[-1]
            if end_sentiment < start_sentiment - 0.2:
                parts.append("Sentiment declined during conversation.")
            elif end_sentiment < 0.4:
                parts.append("Driver sentiment is negative.")
        
        # Trigger explanation
        parts.append(f"Handoff triggered due to: {trigger.value}.")
        
        # Filled slots
        if session.filled_slots:
            slots_str = ", ".join(f"{k}: {v}" for k, v in list(session.filled_slots.items())[:3])
            parts.append(f"Known info: {slots_str}.")
        
        return " ".join(parts)
    
    def _extract_actions_taken(self, session: ConversationSession) -> list[ActionTaken]:
        """Extract actions bot has already taken."""
        actions = []
        
        for turn in session.turns:
            if turn.role == TurnRole.BOT and turn.tool_calls:
                for tool_call in turn.tool_calls:
                    actions.append(ActionTaken(
                        action=tool_call.get("tool", "unknown"),
                        description=f"Called {tool_call.get('tool')} with {tool_call.get('args', {})}",
                        timestamp=turn.timestamp,
                        result=str(tool_call.get("result", {}))[:100],
                        success=tool_call.get("result", {}).get("success", True)
                    ))
        
        return actions
    
    def _generate_suggested_actions(
        self,
        trigger: HandoffTrigger,
        session: ConversationSession
    ) -> list[SuggestedAction]:
        """Generate suggested actions for agent."""
        actions = []
        
        # Base actions based on trigger
        if trigger == HandoffTrigger.NEGATIVE_SENTIMENT:
            actions.append(SuggestedAction(
                action="acknowledge_frustration",
                description="Start by acknowledging the driver's frustration empathetically",
                priority=1
            ))
        
        if trigger == HandoffTrigger.LOW_CONFIDENCE:
            actions.append(SuggestedAction(
                action="clarify_intent",
                description="Ask the driver to explain their issue in simple terms",
                priority=1
            ))
        
        # Intent-specific actions
        if session.current_intent == Intent.INVOICE_EXPLANATION:
            actions.append(SuggestedAction(
                action="verify_invoice",
                description="Pull up the invoice and verify GST calculation",
                priority=2
            ))
            actions.append(SuggestedAction(
                action="check_refund",
                description="Check if any refund is applicable",
                priority=3
            ))
        
        if session.current_intent in [Intent.SUBSCRIPTION_STATUS, Intent.SUBSCRIPTION_RENEWAL]:
            actions.append(SuggestedAction(
                action="verify_subscription",
                description="Verify subscription status in CRM",
                priority=2
            ))
        
        # Default actions
        if not actions:
            actions.append(SuggestedAction(
                action="review_history",
                description="Review driver's recent activity and issues",
                priority=2
            ))
        
        return actions[:5]  # Limit to 5 actions
    
    def _identify_key_moments(self, session: ConversationSession) -> list[int]:
        """Identify key moments in conversation."""
        key_indices = []
        
        for i, turn in enumerate(session.turns):
            if turn.role != TurnRole.USER or not turn.nlu_result:
                continue
            
            # Low confidence moment
            if turn.nlu_result.intent.confidence < 0.5:
                key_indices.append(i)
            
            # Negative sentiment moment
            if turn.nlu_result.sentiment.score < 0.35:
                key_indices.append(i)
            
            # Intent change moment
            if i > 0:
                prev_user_turn = next(
                    (t for t in reversed(session.turns[:i]) if t.role == TurnRole.USER and t.nlu_result),
                    None
                )
                if prev_user_turn and prev_user_turn.nlu_result:
                    if turn.nlu_result.intent.intent != prev_user_turn.nlu_result.intent.intent:
                        key_indices.append(i)
        
        return list(set(key_indices))[:5]
    
    def _extract_key_excerpts(
        self,
        turns: list[ConversationTurn],
        indices: list[int]
    ) -> list[ConversationExcerpt]:
        """Extract key conversation excerpts."""
        excerpts = []
        
        for i in indices:
            if i < len(turns):
                turn = turns[i]
                annotation = None
                
                if turn.nlu_result:
                    if turn.nlu_result.sentiment.score < 0.35:
                        annotation = "Negative sentiment detected"
                    elif turn.nlu_result.intent.confidence < 0.5:
                        annotation = "Low confidence"
                
                excerpts.append(ConversationExcerpt(
                    turn_index=i,
                    role=turn.role,
                    content=turn.content,
                    timestamp=turn.timestamp,
                    is_key_moment=True,
                    annotation=annotation
                ))
        
        return excerpts
    
    def _identify_stuck_point(self, session: ConversationSession) -> str | None:
        """Identify what the driver is stuck on."""
        # Check for unresolved slots
        if session.metrics.unresolved_slots:
            return f"Missing information: {', '.join(session.metrics.unresolved_slots)}"
        
        # Check for repeated intent
        for intent, count in session.metrics.repeated_intents.items():
            if count >= 2:
                return f"Repeated questions about {intent}"
        
        # Check last user turn
        last_user = next(
            (t for t in reversed(session.turns) if t.role == TurnRole.USER),
            None
        )
        if last_user and last_user.nlu_result:
            if last_user.nlu_result.sentiment.label == SentimentLabel.CONFUSED:
                return "Driver seems confused about the process"
        
        return None
    
    def _extract_topics(self, session: ConversationSession) -> list[str]:
        """Extract topics discussed."""
        topics = set()
        
        for turn in session.turns:
            if turn.role == TurnRole.USER and turn.nlu_result:
                topics.add(turn.nlu_result.intent.intent.value)
        
        return list(topics)
