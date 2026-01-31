"""
AWS Bedrock LLM Service for intelligent response generation.
Handles context-aware conversations in Hindi/Hinglish.
"""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import structlog

from src.config import get_settings
from src.models import ConversationSession, ConversationTurn, Intent, Language, TurnRole

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    
    text: str
    confidence: float
    tokens_used: int
    model: str


class BedrockLLMService:
    """
    AWS Bedrock LLM service for response generation.
    Uses Claude for high-quality Hinglish conversations.
    """
    
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
    
    async def generate_response(
        self,
        session: ConversationSession,
        intent: Intent,
        entities: dict[str, Any],
        tool_result: dict | None = None
    ) -> LLMResponse:
        """
        Generate a contextual response using LLM.
        
        Args:
            session: Current conversation session
            intent: Detected user intent
            entities: Extracted entities
            tool_result: Result from tool call if any
        
        Returns:
            Generated response
        """
        # Build conversation history
        history = self._build_history(session.turns[-6:])  # Last 6 turns for context
        
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
            client = await self._get_client()
            
            response = await client.invoke_model(
                modelId=self.settings.aws.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "system": self.SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            response_body = json.loads(response["body"].read())
            text = response_body["content"][0]["text"]
            
            # Extract token usage
            usage = response_body.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            
            logger.info(
                "llm_response_generated",
                intent=intent.value,
                response_length=len(text),
                tokens=tokens
            )
            
            return LLMResponse(
                text=text.strip(),
                confidence=0.9,
                tokens_used=tokens,
                model=self.settings.aws.bedrock_model_id
            )
            
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            # Return fallback response
            return LLMResponse(
                text=self._get_fallback_response(intent, language),
                confidence=0.5,
                tokens_used=0,
                model="fallback"
            )
    
    async def generate_stream(
        self,
        session: ConversationSession,
        intent: Intent,
        entities: dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        Stream response generation for lower latency.
        
        Yields text chunks as they're generated.
        """
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
            
            response = await client.invoke_model_with_response_stream(
                modelId=self.settings.aws.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "system": self.SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            async for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    text = chunk.get("delta", {}).get("text", "")
                    if text:
                        yield text
        
        except Exception as e:
            logger.error("llm_stream_failed", error=str(e))
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
        }
        
        return fallbacks.get(intent, "Main aapki request process kar raha hoon. Please wait.")


class ConversationRAG:
    """
    RAG (Retrieval Augmented Generation) for Battery Smart knowledge.
    Retrieves relevant context for better responses.
    """
    
    # Static knowledge base (in production, use vector DB)
    KNOWLEDGE_BASE = {
        "gst": """GST (Goods and Services Tax) is 18% on battery swap services.
        It's calculated on the base swap price.
        Example: If swap price is ₹50, GST = ₹9, Total = ₹59""",
        
        "subscription": """Battery Smart subscription plans:
        - Daily: ₹49/day (unlimited swaps)
        - Weekly: ₹299/week (unlimited swaps)
        - Monthly: ₹999/month (unlimited swaps)
        All plans include GST.""",
        
        "leave": """Leave/Pause feature:
        - Pause subscription for up to 7 days per month
        - Request via app or call
        - Remaining days added back when resumed""",
        
        "dsk_activation": """DSK (Driver Service Kit) activation:
        1. Complete registration on app
        2. Submit documents (Aadhaar, license)
        3. Visit nearest hub for kit collection
        4. First swap activated after verification""",
        
        "refund": """Refund policy:
        - Overcharges refunded within 5-7 working days
        - Refund to original payment method
        - Contact support with invoice number for faster processing""",
    }
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._embedding_client = None
    
    async def retrieve_context(self, query: str, intent: Intent) -> str:
        """Retrieve relevant context for the query."""
        # Simple keyword-based retrieval for MVP
        # In production, use embeddings and vector search
        
        relevant_docs = []
        query_lower = query.lower()
        
        # Check for keywords
        keyword_mapping = {
            "gst": ["gst", "tax", "18%", "extra charge"],
            "subscription": ["plan", "subscription", "daily", "weekly", "monthly", "validity"],
            "leave": ["leave", "pause", "chutti", "holiday"],
            "dsk_activation": ["dsk", "activation", "new driver", "registration"],
            "refund": ["refund", "money back", "wapas", "return"],
        }
        
        for key, keywords in keyword_mapping.items():
            if any(kw in query_lower for kw in keywords):
                relevant_docs.append(self.KNOWLEDGE_BASE[key])
        
        # Also check based on intent
        intent_mapping = {
            Intent.INVOICE_EXPLANATION: ["gst", "refund"],
            Intent.SUBSCRIPTION_STATUS: ["subscription"],
            Intent.SUBSCRIPTION_PRICING: ["subscription"],
            Intent.LEAVE_INFORMATION: ["leave"],
            Intent.DSK_ACTIVATION: ["dsk_activation"],
        }
        
        intent_keys = intent_mapping.get(intent, [])
        for key in intent_keys:
            if key in self.KNOWLEDGE_BASE and self.KNOWLEDGE_BASE[key] not in relevant_docs:
                relevant_docs.append(self.KNOWLEDGE_BASE[key])
        
        return "\n\n".join(relevant_docs) if relevant_docs else ""


# Singleton instances
_llm_service: BedrockLLMService | None = None
_rag_service: ConversationRAG | None = None


def get_llm_service() -> BedrockLLMService:
    """Get LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = BedrockLLMService()
    return _llm_service


def get_rag_service() -> ConversationRAG:
    """Get RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = ConversationRAG()
    return _rag_service
