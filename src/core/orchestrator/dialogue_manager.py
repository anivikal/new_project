"""
Dialogue Manager - Central orchestrator for conversation flow.
Manages state, context, slot filling, and response generation.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import structlog

from src.config import get_settings
from src.core.decision import HandoffDecision, HandoffDecisionEngine
from src.core.nlu import NLUPipeline, get_nlu_pipeline
from src.models import (
    ConversationSession,
    ConversationState,
    ConversationTurn,
    DriverInfo,
    Entity,
    HandoffTrigger,
    Intent,
    Language,
    NLUResult,
    TurnRole,
)

# Import Groq service for LLM-powered response generation
try:
    from src.services.groq.llm_service import get_groq_service
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class SlotDefinition:
    """Definition of a slot to be filled for an intent."""
    
    name: str
    entity_type: str
    required: bool = True
    prompt_hindi: str = ""
    prompt_english: str = ""
    validation: Callable[[Any], bool] | None = None


@dataclass
class IntentConfig:
    """Configuration for handling an intent."""
    
    intent: Intent
    required_slots: list[SlotDefinition] = field(default_factory=list)
    tool_name: str | None = None  # Tool to call when slots are filled
    response_template_hindi: str = ""
    response_template_english: str = ""
    max_clarifications: int = 2


# Intent configurations for Tier-1 use cases
INTENT_CONFIGS = {
    Intent.SWAP_HISTORY: IntentConfig(
        intent=Intent.SWAP_HISTORY,
        required_slots=[
            SlotDefinition(
                name="date_range",
                entity_type="date_range",
                required=False,
                prompt_hindi="Kitne din ka history chahiye? (Last 7 days, 30 days, etc.)",
                prompt_english="How many days of history do you need?",
            ),
        ],
        tool_name="get_swap_history",
        response_template_hindi="Aapke {count} swaps hue hain. {details}",
        response_template_english="You have {count} swaps. {details}",
    ),
    Intent.SWAP_INVOICE: IntentConfig(
        intent=Intent.SWAP_INVOICE,
        required_slots=[
            SlotDefinition(
                name="invoice_number",
                entity_type="invoice_number",
                required=False,
                prompt_hindi="Kaunsi invoice chahiye? Date ya invoice number bataiye.",
                prompt_english="Which invoice do you need? Please provide date or invoice number.",
            ),
        ],
        tool_name="get_invoice",
    ),
    Intent.INVOICE_EXPLANATION: IntentConfig(
        intent=Intent.INVOICE_EXPLANATION,
        required_slots=[
            SlotDefinition(
                name="invoice_number",
                entity_type="invoice_number",
                required=False,
            ),
        ],
        tool_name="explain_invoice",
    ),
    Intent.NEAREST_STATION: IntentConfig(
        intent=Intent.NEAREST_STATION,
        required_slots=[
            SlotDefinition(
                name="location",
                entity_type="location",
                required=False,
                prompt_hindi="Aap kahan hain? Area ya landmark bataiye.",
                prompt_english="Where are you? Please share your area or landmark.",
            ),
        ],
        tool_name="find_nearest_station",
    ),
    Intent.STATION_AVAILABILITY: IntentConfig(
        intent=Intent.STATION_AVAILABILITY,
        required_slots=[
            SlotDefinition(
                name="station_name",
                entity_type="station_name",
                required=False,
                prompt_hindi="Kaunse station ki availability chahiye?",
                prompt_english="Which station's availability do you want to check?",
            ),
        ],
        tool_name="check_availability",
    ),
    Intent.SUBSCRIPTION_STATUS: IntentConfig(
        intent=Intent.SUBSCRIPTION_STATUS,
        required_slots=[],
        tool_name="get_subscription_status",
    ),
    Intent.SUBSCRIPTION_RENEWAL: IntentConfig(
        intent=Intent.SUBSCRIPTION_RENEWAL,
        required_slots=[],
        tool_name="initiate_renewal",
    ),
    Intent.SUBSCRIPTION_PRICING: IntentConfig(
        intent=Intent.SUBSCRIPTION_PRICING,
        required_slots=[],
        tool_name="get_pricing",
    ),
    Intent.PLAN_COMPARISON: IntentConfig(
        intent=Intent.PLAN_COMPARISON,
        required_slots=[],
        tool_name="compare_plans",
    ),
    Intent.LEAVE_INFORMATION: IntentConfig(
        intent=Intent.LEAVE_INFORMATION,
        required_slots=[
            SlotDefinition(
                name="leave_dates",
                entity_type="date_range",
                required=True,
                prompt_hindi="Kitne din ka leave chahiye? Start aur end date bataiye.",
                prompt_english="How many days of leave do you need?",
            ),
        ],
        tool_name="process_leave",
    ),
    Intent.DSK_ACTIVATION: IntentConfig(
        intent=Intent.DSK_ACTIVATION,
        required_slots=[
            SlotDefinition(
                name="driver_id",
                entity_type="driver_id",
                required=True,
                prompt_hindi="Aapka driver ID kya hai?",
                prompt_english="What is your driver ID?",
            ),
        ],
        tool_name="activate_dsk",
    ),
}


@dataclass
class DialogueState:
    """Current state of dialogue for slot filling and flow control."""
    
    current_intent: Intent | None = None
    pending_slots: list[str] = field(default_factory=list)
    filled_slots: dict[str, Any] = field(default_factory=dict)
    clarification_count: int = 0
    awaiting_confirmation: bool = False
    confirmation_data: dict | None = None


class ResponseGenerator:
    """Generates natural responses in Hindi/Hinglish."""
    
    GREETING_RESPONSES = {
        Language.HINDI: "नमस्ते! मैं Battery Smart का AI assistant हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?",
        Language.HINGLISH: "Namaste! Main Battery Smart ka AI assistant hoon. Aaj main aapki kaise help kar sakta hoon?",
        Language.ENGLISH_INDIA: "Hello! I'm Battery Smart's AI assistant. How can I help you today?",
    }
    
    CLARIFICATION_RESPONSES = {
        Language.HINDI: "जी, मैं सुन रहा हूँ। आप बताइए - swap history, nearest station, ya subscription के बारे में पूछना है?",
        Language.HINGLISH: "Ji, main sun raha hoon. Aap bataiye - swap history, nearest station, ya subscription ke baare mein poochna hai?",
        Language.ENGLISH_INDIA: "Yes, I'm listening. Please tell me - do you want to ask about swap history, nearest station, or subscription?",
    }
    
    # Empathetic responses for when user seems frustrated
    EMPATHY_RESPONSES = {
        Language.HINDI: "मैं समझ सकता हूँ। मैं आपकी मदद करने के लिए यहाँ हूँ। बताइए क्या परेशानी है?",
        Language.HINGLISH: "Main samajh sakta hoon. Main aapki madad karne ke liye yahan hoon. Bataiye kya pareshani hai?",
        Language.ENGLISH_INDIA: "I understand. I'm here to help you. Please tell me what's the issue?",
    }
    
    HELP_RESPONSES = {
        Language.HINDI: "मैं आपकी इन चीज़ों में मदद कर सकता हूँ: 1) Swap history 2) Nearest station 3) Subscription status 4) Invoice help। क्या चाहिए?",
        Language.HINGLISH: "Main aapki in cheezon mein madad kar sakta hoon: 1) Swap history 2) Nearest station 3) Subscription status 4) Invoice help. Kya chahiye?",
        Language.ENGLISH_INDIA: "I can help you with: 1) Swap history 2) Nearest station 3) Subscription status 4) Invoice help. What do you need?",
    }
    
    HANDOFF_RESPONSES = {
        Language.HINDI: "मैं आपको हमारे customer care executive से connect कर रहा हूँ। कृपया थोड़ा इंतज़ार करें।",
        Language.HINGLISH: "Main aapko humare customer care executive se connect kar raha hoon. Please thoda wait karein.",
        Language.ENGLISH_INDIA: "I'm connecting you with our customer care executive. Please wait a moment.",
    }
    
    GOODBYE_RESPONSES = {
        Language.HINDI: "धन्यवाद! Battery Smart को choose करने के लिए। अगर कोई और help चाहिए तो ज़रूर बताइए।",
        Language.HINGLISH: "Thank you! Battery Smart choose karne ke liye. Agar koi aur help chahiye toh zaroor bataiye.",
        Language.ENGLISH_INDIA: "Thank you for choosing Battery Smart! Let us know if you need any other help.",
    }
    
    def generate_greeting(self, language: Language) -> str:
        """Generate greeting response."""
        return self.GREETING_RESPONSES.get(language, self.GREETING_RESPONSES[Language.HINGLISH])
    
    def generate_clarification(self, language: Language, context: str = "") -> str:
        """Generate clarification request."""
        base = self.CLARIFICATION_RESPONSES.get(language, self.CLARIFICATION_RESPONSES[Language.HINGLISH])
        if context:
            base += f" {context}"
        return base
    
    def generate_empathy(self, language: Language) -> str:
        """Generate empathetic response for frustrated users."""
        return self.EMPATHY_RESPONSES.get(language, self.EMPATHY_RESPONSES[Language.HINGLISH])
    
    def generate_help(self, language: Language) -> str:
        """Generate help/menu response."""
        return self.HELP_RESPONSES.get(language, self.HELP_RESPONSES[Language.HINGLISH])
    
    def generate_handoff_message(self, language: Language) -> str:
        """Generate handoff transition message."""
        return self.HANDOFF_RESPONSES.get(language, self.HANDOFF_RESPONSES[Language.HINGLISH])
    
    def generate_goodbye(self, language: Language) -> str:
        """Generate goodbye response."""
        return self.GOODBYE_RESPONSES.get(language, self.GOODBYE_RESPONSES[Language.HINGLISH])
    
    def generate_slot_prompt(self, slot: SlotDefinition, language: Language) -> str:
        """Generate prompt for missing slot."""
        if language in [Language.HINDI, Language.HINGLISH]:
            return slot.prompt_hindi or slot.prompt_english
        return slot.prompt_english or slot.prompt_hindi
    
    def format_response(self, template: str, **kwargs) -> str:
        """Format response template with values."""
        try:
            return template.format(**kwargs)
        except KeyError:
            return template


class DialogueManager:
    """
    Central dialogue manager orchestrating conversation flow.
    
    Responsibilities:
    1. Process user input through NLU
    2. Manage conversation state and context
    3. Handle slot filling for intents
    4. Generate appropriate responses using LLM
    5. Integrate with decision layer for handoff
    """
    
    def __init__(
        self,
        nlu_pipeline: NLUPipeline | None = None,
        decision_engine: HandoffDecisionEngine | None = None,
    ) -> None:
        self.settings = get_settings()
        self.nlu_pipeline = nlu_pipeline or get_nlu_pipeline()
        self.decision_engine = decision_engine or HandoffDecisionEngine()
        self.response_generator = ResponseGenerator()
        
        # Initialize Groq LLM service for dynamic response generation
        self._groq_service = None
        if GROQ_AVAILABLE:
            try:
                self._groq_service = get_groq_service()
                logger.info("groq_service_initialized_for_dialogue")
            except Exception as e:
                logger.warning("groq_service_init_failed", error=str(e))
        
        # Session storage (in production, use Redis)
        self._sessions: dict[str, ConversationSession] = {}
        self._dialogue_states: dict[str, DialogueState] = {}
    
    async def create_session(
        self,
        call_id: str,
        phone_number: str,
        driver_info: DriverInfo | None = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        driver = driver_info or DriverInfo(phone_number=phone_number)
        
        session = ConversationSession(
            call_id=call_id,
            driver=driver,
        )
        
        self._sessions[call_id] = session
        self._dialogue_states[call_id] = DialogueState()
        
        logger.info(
            "session_created",
            call_id=call_id,
            session_id=str(session.id)
        )
        
        return session
    
    async def get_session(self, call_id: str) -> ConversationSession | None:
        """Get existing session."""
        return self._sessions.get(call_id)
    
    async def process_turn(
        self,
        call_id: str,
        user_text: str,
        audio_url: str | None = None
    ) -> tuple[str, HandoffDecision | None]:
        """
        Process a user turn and generate response.
        
        Args:
            call_id: Call/session identifier
            user_text: Transcribed user speech
            audio_url: Optional URL to audio recording
        
        Returns:
            Tuple of (response_text, handoff_decision if triggered)
        """
        session = await self.get_session(call_id)
        if not session:
            raise ValueError(f"Session not found: {call_id}")
        
        dialogue_state = self._dialogue_states.get(call_id, DialogueState())
        
        start_time = datetime.utcnow()
        
        # 1. Process through NLU
        previous_turns = [
            t.content for t in session.turns
            if t.role == TurnRole.USER
        ][-5:]
        
        nlu_result = await self.nlu_pipeline.process(
            user_text,
            context=session.filled_slots,
            previous_turns=previous_turns
        )
        
        # 2. Create user turn
        user_turn = ConversationTurn(
            role=TurnRole.USER,
            content=user_text,
            audio_url=audio_url,
            nlu_result=nlu_result,
            confidence=nlu_result.intent.confidence,
        )
        session.add_turn(user_turn)
        
        # 3. Check for handoff
        handoff_decision = self.decision_engine.evaluate(session, nlu_result)
        
        if handoff_decision.should_handoff:
            response = self.response_generator.generate_handoff_message(
                session.driver.preferred_language
            )
            session.state = ConversationState.HANDOFF_PENDING
            session.handoff_triggered = True
            session.handoff_trigger = handoff_decision.trigger
            
            # Create bot turn
            bot_turn = ConversationTurn(
                role=TurnRole.BOT,
                content=response,
                latency_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            )
            session.turns.append(bot_turn)
            
            return response, handoff_decision
        
        # 4. Handle intent
        response, tool_calls = await self._handle_intent(
            session,
            dialogue_state,
            nlu_result
        )
        
        # 5. Create bot turn
        bot_turn = ConversationTurn(
            role=TurnRole.BOT,
            content=response,
            latency_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            tool_calls=tool_calls,
        )
        session.turns.append(bot_turn)
        
        # 6. Update session state
        self._dialogue_states[call_id] = dialogue_state
        
        return response, None
    
    async def _handle_intent(
        self,
        session: ConversationSession,
        dialogue_state: DialogueState,
        nlu_result: NLUResult
    ) -> tuple[str, list[dict]]:
        """Handle intent and generate response using LLM."""
        intent = nlu_result.intent.intent
        language = session.driver.preferred_language
        tool_calls = []
        
        # Get user's original query for context
        user_query = session.turns[-1].content if session.turns else ""
        
        # Build conversation context for LLM
        conversation_context = self._build_conversation_context(session)
        
        # Try to generate LLM response for all intents
        if self._groq_service:
            try:
                # Handle special intents with LLM-generated responses
                if intent == Intent.GREETING:
                    llm_response = await self._generate_llm_response(
                        user_query=user_query,
                        intent=intent,
                        context=conversation_context,
                        language=language
                    )
                    return llm_response, []
                
                if intent == Intent.GOODBYE:
                    session.state = ConversationState.COMPLETED
                    llm_response = await self._generate_llm_response(
                        user_query=user_query,
                        intent=intent,
                        context=conversation_context,
                        language=language
                    )
                    return llm_response, []
                
                if intent == Intent.HELP:
                    llm_response = await self._generate_llm_response(
                        user_query=user_query,
                        intent=intent,
                        context=conversation_context,
                        language=language
                    )
                    return llm_response, []
                
                if intent == Intent.UNKNOWN or intent == Intent.OUT_OF_SCOPE:
                    dialogue_state.clarification_count += 1
                    session.metrics.clarification_count = dialogue_state.clarification_count
                    
                    # Generate contextual response using LLM
                    llm_response = await self._generate_llm_response(
                        user_query=user_query,
                        intent=intent,
                        context=conversation_context,
                        language=language,
                        sentiment=nlu_result.sentiment.label.value if nlu_result.sentiment else "neutral"
                    )
                    return llm_response, []
                    
            except Exception as e:
                logger.warning("llm_response_generation_failed", error=str(e))
                # Fall back to template responses
        
        # Fallback: Handle special intents with templates
        if intent == Intent.GREETING:
            return self.response_generator.generate_greeting(language), []
        
        if intent == Intent.GOODBYE:
            session.state = ConversationState.COMPLETED
            return self.response_generator.generate_goodbye(language), []
        
        if intent == Intent.HELP:
            return self.response_generator.generate_help(language), []
        
        if intent == Intent.UNKNOWN or intent == Intent.OUT_OF_SCOPE:
            dialogue_state.clarification_count += 1
            session.metrics.clarification_count = dialogue_state.clarification_count
            
            if nlu_result.sentiment and nlu_result.sentiment.label.value in ['negative', 'frustrated']:
                return self.response_generator.generate_empathy(language), []
            
            if dialogue_state.clarification_count == 1:
                return self.response_generator.generate_help(language), []
            
            return self.response_generator.generate_clarification(language), []
        
        # Get intent configuration
        intent_config = INTENT_CONFIGS.get(intent)
        if not intent_config:
            # Fallback for unconfigured intents
            return self._generate_fallback_response(intent, language), []
        
        # Update current intent
        if dialogue_state.current_intent != intent:
            dialogue_state.current_intent = intent
            dialogue_state.pending_slots = [
                slot.name for slot in intent_config.required_slots
                if slot.required
            ]
            dialogue_state.filled_slots = {}
        
        session.current_intent = intent
        
        # Extract and fill slots from entities
        for entity in nlu_result.entities:
            for slot in intent_config.required_slots:
                if entity.type.value == slot.entity_type:
                    dialogue_state.filled_slots[slot.name] = entity.normalized_value or entity.value
                    if slot.name in dialogue_state.pending_slots:
                        dialogue_state.pending_slots.remove(slot.name)
        
        # Update session slots
        session.filled_slots.update(dialogue_state.filled_slots)
        
        # Check if we have all required slots
        if dialogue_state.pending_slots:
            # Need more information
            missing_slot_name = dialogue_state.pending_slots[0]
            missing_slot = next(
                (s for s in intent_config.required_slots if s.name == missing_slot_name),
                None
            )
            if missing_slot:
                session.metrics.unresolved_slots = dialogue_state.pending_slots.copy()
                return self.response_generator.generate_slot_prompt(missing_slot, language), []
        
        # All slots filled - execute tool/action
        if intent_config.tool_name:
            tool_result = await self._execute_tool(
                intent_config.tool_name,
                dialogue_state.filled_slots,
                session
            )
            tool_calls.append({
                "tool": intent_config.tool_name,
                "args": dialogue_state.filled_slots,
                "result": tool_result
            })
            
            # Generate response based on tool result using LLM
            response = await self._generate_tool_response(
                intent_config,
                tool_result,
                language,
                user_query=user_query
            )
            
            # Mark intent as resolved
            session.metrics.intents_resolved.append(intent.value)
            
            # Reset dialogue state for this intent
            dialogue_state.current_intent = None
            dialogue_state.pending_slots = []
            dialogue_state.filled_slots = {}
            
            return response, tool_calls
        
        return self._generate_fallback_response(intent, language), tool_calls
    
    async def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        session: ConversationSession
    ) -> dict:
        """
        Execute a tool/action.
        
        In production, this would call actual CRM/backend APIs.
        For MVP, returns mock data.
        """
        # This is where you'd integrate with actual services
        # For now, return mock responses
        
        mock_responses = {
            "get_swap_history": {
                "success": True,
                "count": 15,
                "details": "Last swap was 2 hours ago at Andheri station.",
            },
            "get_invoice": {
                "success": True,
                "invoice_number": "INV-2024-001234",
                "amount": 150,
                "gst": 27,
                "total": 177,
            },
            "explain_invoice": {
                "success": True,
                "breakdown": "Base charge: ₹150, GST (18%): ₹27, Total: ₹177",
            },
            "find_nearest_station": {
                "success": True,
                "station": "Andheri West Hub",
                "distance": "1.2 km",
                "batteries_available": 8,
            },
            "check_availability": {
                "success": True,
                "available": True,
                "count": 12,
            },
            "get_subscription_status": {
                "success": True,
                "plan": "Monthly Premium",
                "valid_till": "2024-02-28",
                "swaps_remaining": 45,
            },
            "get_pricing": {
                "success": True,
                "plans": [
                    {"name": "Daily", "price": 49},
                    {"name": "Weekly", "price": 299},
                    {"name": "Monthly", "price": 999},
                ],
            },
        }
        
        return mock_responses.get(tool_name, {"success": False, "error": "Unknown tool"})
    
    async def _generate_tool_response(
        self,
        intent_config: IntentConfig,
        tool_result: dict,
        language: Language,
        user_query: str = ""
    ) -> str:
        """Generate response from tool result using LLM."""
        if not tool_result.get("success"):
            return self._generate_error_response(language)
        
        # Try to use LLM for natural response generation
        if self._groq_service:
            try:
                # Build prompt with tool result data
                data_context = f"Data: {tool_result}"
                llm_response = await self._generate_llm_response(
                    user_query=user_query,
                    intent=intent_config.intent,
                    context={"tool_result": tool_result},
                    language=language
                )
                return llm_response
            except Exception as e:
                logger.warning("llm_tool_response_failed", error=str(e))
        
        # Fallback to templates
        templates = {
            Intent.SWAP_HISTORY: {
                Language.HINGLISH: "Aapke total {count} swaps hue hain. {details}",
                Language.HINDI: "आपके कुल {count} स्वैप हुए हैं। {details}",
                Language.ENGLISH_INDIA: "You have {count} total swaps. {details}",
            },
            Intent.SUBSCRIPTION_STATUS: {
                Language.HINGLISH: "Aapka {plan} plan active hai. Valid till {valid_till}. {swaps_remaining} swaps remaining.",
                Language.HINDI: "आपका {plan} प्लान एक्टिव है। {valid_till} तक valid है।",
                Language.ENGLISH_INDIA: "Your {plan} plan is active until {valid_till}.",
            },
            Intent.NEAREST_STATION: {
                Language.HINGLISH: "Aapke paas {station} hai, sirf {distance} door. Wahan {batteries_available} batteries available hain.",
                Language.HINDI: "आपके पास {station} है, सिर्फ {distance} दूर।",
                Language.ENGLISH_INDIA: "The nearest station is {station}, {distance} away with {batteries_available} batteries available.",
            },
        }
        
        intent_templates = templates.get(intent_config.intent, {})
        template = intent_templates.get(language, intent_templates.get(Language.HINGLISH, ""))
        
        if template:
            try:
                return template.format(**tool_result)
            except KeyError:
                pass
        
        # Fallback
        return f"Done! {tool_result}"
    
    def _build_conversation_context(self, session: ConversationSession) -> dict:
        """Build conversation context for LLM."""
        # Get last 5 turns for context
        recent_turns = []
        for turn in session.turns[-10:]:
            recent_turns.append({
                "role": "user" if turn.role == TurnRole.USER else "assistant",
                "content": turn.content
            })
        
        return {
            "conversation_history": recent_turns,
            "driver_name": session.driver.name or "Driver",
            "current_intent": session.current_intent.value if session.current_intent else None,
            "filled_slots": session.filled_slots,
        }
    
    async def _generate_llm_response(
        self,
        user_query: str,
        intent: Intent,
        context: dict,
        language: Language,
        sentiment: str = "neutral"
    ) -> str:
        """Generate natural response using Groq LLM."""
        if not self._groq_service:
            raise RuntimeError("Groq service not available")
        
        # Use the Groq service's generate_response method
        response = await self._groq_service.generate_response(
            user_query=user_query,
            intent=intent,
            entities=[],  # Entities already processed
            language=language,
            context=context
        )
        
        return response
    
    def _generate_help_response(self, language: Language) -> str:
        """Generate help menu response."""
        responses = {
            Language.HINGLISH: """Main aapki in cheezon mein help kar sakta hoon:
1. Swap history aur invoice
2. Nearest station dhundhna
3. Subscription status aur renewal
4. Leave request
5. DSK activation

Kya chahiye aapko?""",
            Language.HINDI: """मैं आपकी इन चीज़ों में मदद कर सकता हूँ:
1. स्वैप हिस्ट्री और इनवॉइस
2. नज़दीकी स्टेशन
3. सब्सक्रिप्शन स्टेटस
4. लीव रिक्वेस्ट
5. DSK एक्टिवेशन""",
            Language.ENGLISH_INDIA: """I can help you with:
1. Swap history and invoices
2. Finding nearest station
3. Subscription status and renewal
4. Leave requests
5. DSK activation

What do you need?""",
        }
        return responses.get(language, responses[Language.HINGLISH])
    
    def _generate_fallback_response(self, intent: Intent, language: Language) -> str:
        """Generate fallback response for unhandled intents."""
        return f"I understand you're asking about {intent.value}. Let me help you with that."
    
    def _generate_error_response(self, language: Language) -> str:
        """Generate error response."""
        responses = {
            Language.HINGLISH: "Sorry, kuch technical problem hai. Kya aap thodi der baad try kar sakte hain?",
            Language.HINDI: "माफ़ कीजिए, कुछ तकनीकी समस्या है।",
            Language.ENGLISH_INDIA: "Sorry, there's a technical issue. Please try again later.",
        }
        return responses.get(language, responses[Language.HINGLISH])
    
    async def end_session(self, call_id: str) -> None:
        """End and cleanup a session."""
        session = await self.get_session(call_id)
        if session:
            session.state = ConversationState.COMPLETED
            session.ended_at = datetime.utcnow()
            
            logger.info(
                "session_ended",
                call_id=call_id,
                duration_seconds=(session.ended_at - session.started_at).total_seconds(),
                turns=len(session.turns)
            )
            
            # In production, persist to database before cleanup
            # For now, just log
            
            # Cleanup
            del self._sessions[call_id]
            if call_id in self._dialogue_states:
                del self._dialogue_states[call_id]
