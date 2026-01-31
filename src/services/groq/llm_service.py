"""
Groq LLM Service for intelligent response generation.
Handles context-aware conversations in Hindi/Hinglish using Groq's fast inference.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import structlog

from src.config import get_settings
from src.models import (
    ConversationSession,
    ConversationTurn,
    Intent,
    IntentClassification,
    Language,
    Sentiment,
    SentimentLabel,
    TurnRole,
)

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    
    text: str
    confidence: float
    tokens_used: int
    model: str


class GroqLLMService:
    """
    Groq LLM service for response generation, intent classification,
    sentiment analysis, and entity extraction.
    Uses Groq's fast inference for low-latency responses.
    """
    
    BASE_URL = "https://api.groq.com/openai/v1"
    
    SYSTEM_PROMPT = """You are a helpful AI assistant for Battery Smart, India's largest EV battery swapping network.
You help drivers/riders with their queries in a friendly, conversational manner.

Key guidelines:
1. Respond naturally in Hinglish (Hindi-English mix) or pure Hindi based on user's language
2. Be concise - drivers are usually in a hurry
3. Be empathetic when users are frustrated
4. Always confirm understanding before taking action
5. If you can't help, acknowledge it and offer to connect with human support

Battery Smart services you can help with:
- Swap history and invoices
- Finding nearest stations
- Checking battery availability
- Subscription status and renewal
- Leave/pause requests
- DSK activation

Common terms:
- Swap = Battery exchange at station
- DSK = Driver Service Kit (new driver activation)
- Hub/Station = Battery swapping location
- GST = 18% tax on services

Always be polite, use "aap" (formal you) when addressing drivers."""

    RESPONSE_PROMPT = """Generate a helpful response for this Battery Smart driver query.

Conversation so far:
{history}

Current query: {query}
Detected intent: {intent}
Extracted entities: {entities}

Generate a natural response in {language}.
Be concise (max 2-3 sentences).
If you need more information, ask a specific clarifying question.

Response:"""

    INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for Battery Smart's driver support system.
Battery Smart operates India's largest battery-swapping network for electric vehicles.

Classify the following driver query into one of these intents:
- swap_history: Asking about past battery swaps, swap records, history
- swap_invoice: Requesting invoice/bill for swaps
- invoice_explanation: Asking to explain charges, GST, extra amounts on invoice
- nearest_station: Finding nearest battery swap station
- station_availability: Checking if battery is available at station
- subscription_status: Checking current subscription/plan status
- subscription_renewal: Wanting to renew subscription
- subscription_pricing: Asking about plan prices/costs
- plan_comparison: Comparing different plans
- leave_information: Taking leave/pause from subscription
- dsk_activation: Activating new DSK/account
- human_agent: Explicitly requesting human support
- greeting: Simple greeting (hi, hello, namaste, etc.)
- goodbye: Ending conversation
- help: Asking what the bot can do
- out_of_scope: Query not related to Battery Smart services
- unknown: Cannot determine intent

The query may be in Hindi, English, or Hinglish (mixed Hindi-English).

Query: "{query}"

Respond in JSON format only:
{{"intent": "<intent_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    SENTIMENT_ANALYSIS_PROMPT = """Analyze the sentiment of this Battery Smart driver support query.
The driver is contacting support about battery swapping services.
The text may be in Hindi, English, or Hinglish (mixed).

Consider:
1. Explicit emotional words
2. Tone and urgency
3. Signs of frustration (repetition, impatience, complaints)
4. Signs of confusion (questions, uncertainty)
5. Politeness level

Query: "{query}"

Classify as one of: positive, neutral, negative, frustrated, confused

Respond in JSON format only:
{{"label": "<sentiment_label>", "score": <0.0-1.0 where 0=very negative, 1=very positive>, "confidence": <0.0-1.0>, "indicators": ["<indicator1>", "<indicator2>"]}}"""

    ENTITY_EXTRACTION_PROMPT = """Extract entities from this Battery Smart driver support query.
The query may be in Hindi, English, or Hinglish (mixed).

Entity types to extract:
- phone_number: Phone numbers
- amount: Money amounts in rupees
- date: Dates or time periods
- invoice_number: Invoice or bill numbers
- station_name: Battery swap station names
- location: Places, areas, landmarks
- subscription_plan: Plan names (daily, weekly, monthly)
- vehicle_id: Vehicle registration numbers

Query: "{query}"

Respond in JSON format only:
{{"entities": [{{"type": "<entity_type>", "value": "<extracted_value>", "confidence": <0.0-1.0>}}]}}"""

    SUMMARY_PROMPT = """Generate a concise agent handoff summary for this Battery Smart driver support conversation.
The summary should help the human agent quickly understand the situation.

Conversation:
{conversation}

Driver Info:
- Phone: {phone}
- Language: {language}

Metrics:
- Turns: {turns}
- Duration: {duration}s
- Trigger: {trigger}

Generate a JSON summary:
{{
    "one_line_summary": "<max 100 chars quick summary>",
    "primary_issue": "<main problem>",
    "actions_taken": ["<action1>", "<action2>"],
    "suggested_actions": ["<suggestion1>", "<suggestion2>"],
    "key_moments": ["<important_turn1>", "<important_turn2>"],
    "sentiment_trajectory": "<positive/declining/stable/negative>"
}}"""

    def __init__(self, api_key: str | None = None) -> None:
        self.settings = get_settings()
        self.api_key = api_key or self.settings.groq.api_key
        self.model = self.settings.groq.model_id
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    async def _call_groq(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> dict:
        """Make a call to Groq API."""
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("groq_api_error", status=e.response.status_code, detail=str(e))
            raise
        except Exception as e:
            logger.error("groq_request_failed", error=str(e))
            raise
    
    async def generate_response(
        self,
        session: ConversationSession,
        intent: Intent,
        entities: dict[str, Any],
        tool_result: dict | None = None
    ) -> LLMResponse:
        """
        Generate a contextual response using Groq LLM.
        
        Args:
            session: Current conversation session
            intent: Detected user intent
            entities: Extracted entities
            tool_result: Result from tool call if any
        
        Returns:
            Generated response
        """
        # Build conversation history
        history = self._build_history(session.turns[-6:])
        
        # Determine target language
        language = "Hinglish" if session.driver.preferred_language == Language.HINGLISH else \
                   "Hindi" if session.driver.preferred_language == Language.HINDI else "English"
        
        # Build prompt
        prompt = self.RESPONSE_PROMPT.format(
            history=history,
            query=session.turns[-1].content if session.turns else "",
            intent=intent.value,
            entities=json.dumps(entities, default=str),
            language=language
        )
        
        # Add tool result context
        if tool_result:
            prompt += f"\n\nTool result to incorporate: {json.dumps(tool_result, default=str)}"
        
        try:
            result = await self._call_groq(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            
            logger.info(
                "groq_response_generated",
                intent=intent.value,
                response_length=len(text),
                tokens=tokens
            )
            
            return LLMResponse(
                text=text.strip(),
                confidence=0.9,
                tokens_used=tokens,
                model=self.model
            )
            
        except Exception as e:
            logger.error("groq_generation_failed", error=str(e))
            return LLMResponse(
                text=self._get_fallback_response(intent, language),
                confidence=0.5,
                tokens_used=0,
                model="fallback"
            )
    
    async def classify_intent(
        self,
        text: str,
        language: Language,
        context: dict | None = None
    ) -> IntentClassification:
        """Classify intent using Groq LLM."""
        try:
            prompt = self.INTENT_CLASSIFICATION_PROMPT.format(query=text)
            
            if context:
                context_str = f"\nConversation context: {json.dumps(context)}"
                prompt = prompt.replace('Query:', f'{context_str}\n\nQuery:')
            
            result = await self._call_groq(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = result["choices"][0]["message"]["content"]
            
            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Map to Intent enum
            intent_str = parsed.get("intent", "unknown")
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN
            
            logger.info(
                "groq_intent_classified",
                intent=intent.value,
                confidence=parsed.get("confidence", 0.7)
            )
            
            return IntentClassification(
                intent=intent,
                confidence=float(parsed.get("confidence", 0.7)),
                alternative_intents=[]
            )
            
        except Exception as e:
            logger.error("groq_classification_failed", error=str(e))
            return IntentClassification(
                intent=Intent.UNKNOWN,
                confidence=0.3,
                alternative_intents=[]
            )
    
    async def analyze_sentiment(
        self,
        text: str,
        language: Language,
        context: list[str] | None = None
    ) -> Sentiment:
        """Analyze sentiment using Groq LLM."""
        try:
            prompt = self.SENTIMENT_ANALYSIS_PROMPT.format(query=text)
            
            if context:
                context_section = "\nPrevious conversation turns:\n" + "\n".join(
                    f"- {turn}" for turn in context[-3:]
                )
                prompt = prompt.replace('Query:', f'{context_section}\n\nQuery:')
            
            result = await self._call_groq(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = result["choices"][0]["message"]["content"]
            
            # Parse JSON
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Map label
            label_map = {
                "positive": SentimentLabel.POSITIVE,
                "neutral": SentimentLabel.NEUTRAL,
                "negative": SentimentLabel.NEGATIVE,
                "frustrated": SentimentLabel.FRUSTRATED,
                "confused": SentimentLabel.CONFUSED,
            }
            
            label = label_map.get(parsed.get("label", "neutral").lower(), SentimentLabel.NEUTRAL)
            
            logger.info(
                "groq_sentiment_analyzed",
                label=label.value,
                score=parsed.get("score", 0.5)
            )
            
            return Sentiment(
                label=label,
                score=float(parsed.get("score", 0.5)),
                confidence=float(parsed.get("confidence", 0.8)),
                indicators=parsed.get("indicators", [])
            )
            
        except Exception as e:
            logger.error("groq_sentiment_failed", error=str(e))
            return Sentiment(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                confidence=0.3,
                indicators=[]
            )
    
    async def extract_entities(self, text: str, language: Language) -> list[dict]:
        """Extract entities using Groq LLM."""
        try:
            prompt = self.ENTITY_EXTRACTION_PROMPT.format(query=text)
            
            result = await self._call_groq(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = result["choices"][0]["message"]["content"]
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                entities = parsed.get("entities", [])
                logger.info("groq_entities_extracted", count=len(entities))
                return entities
            
            return []
            
        except Exception as e:
            logger.error("groq_entity_extraction_failed", error=str(e))
            return []
    
    async def generate_summary(
        self,
        session: ConversationSession,
        trigger: str
    ) -> dict:
        """Generate handoff summary using Groq LLM."""
        try:
            # Build conversation text
            conversation = "\n".join([
                f"{'Driver' if t.role == TurnRole.USER else 'Bot'}: {t.content}"
                for t in session.turns
            ])
            
            prompt = self.SUMMARY_PROMPT.format(
                conversation=conversation,
                phone=session.driver.phone_number or "Unknown",
                language=session.driver.preferred_language.value,
                turns=len(session.turns),
                duration=int((session.turns[-1].timestamp - session.turns[0].timestamp).total_seconds()) if len(session.turns) > 1 else 0,
                trigger=trigger
            )
            
            result = await self._call_groq(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = result["choices"][0]["message"]["content"]
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                summary = json.loads(json_match.group())
                logger.info("groq_summary_generated")
                return summary
            
            raise ValueError("No JSON in response")
            
        except Exception as e:
            logger.error("groq_summary_failed", error=str(e))
            # Return fallback summary
            return {
                "one_line_summary": f"Handoff triggered: {trigger}",
                "primary_issue": "Unable to determine",
                "actions_taken": [],
                "suggested_actions": ["Review conversation history"],
                "key_moments": [],
                "sentiment_trajectory": "unknown"
            }
    
    async def generate_stream(
        self,
        session: ConversationSession,
        intent: Intent,
        entities: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Stream response generation for lower latency."""
        history = self._build_history(session.turns[-6:])
        language = "Hinglish"
        
        prompt = self.RESPONSE_PROMPT.format(
            history=history,
            query=session.turns[-1].content if session.turns else "",
            intent=intent.value,
            entities=json.dumps(entities, default=str),
            language=language
        )
        
        try:
            client = await self._get_client()
            
            async with client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error("groq_stream_failed", error=str(e))
            yield self._get_fallback_response(intent, "Hinglish")
    
    def _build_history(self, turns: list[ConversationTurn]) -> str:
        """Build conversation history string."""
        history_parts = []
        
        for turn in turns:
            role = "Driver" if turn.role == TurnRole.USER else "Bot"
            history_parts.append(f"{role}: {turn.content}")
        
        return "\n".join(history_parts) if history_parts else "No previous conversation."
    
    def _get_fallback_response(self, intent: Intent, language: str) -> str:
        """Get fallback response when LLM fails."""
        fallbacks = {
            Intent.SWAP_HISTORY: "Main aapki swap history check kar raha hoon. Ek moment please.",
            Intent.NEAREST_STATION: "Main aapke paas ka station dhundh raha hoon.",
            Intent.SUBSCRIPTION_STATUS: "Main aapki subscription check kar raha hoon.",
            Intent.HUMAN_AGENT: "Main aapko human agent se connect kar raha hoon.",
            Intent.GREETING: "Namaste! Main Battery Smart ka AI assistant hoon. Aaj main aapki kaise help kar sakta hoon?",
            Intent.HELP: "Main aapki in cheezon mein madad kar sakta hoon: Swap history, Nearest station, Subscription status, Invoice help. Kya chahiye?",
        }
        
        return fallbacks.get(intent, "Main aapki request process kar raha hoon. Please wait.")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


# Singleton instance
_groq_service: GroqLLMService | None = None


def get_groq_service() -> GroqLLMService:
    """Get Groq service instance."""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqLLMService()
    return _groq_service
