"""
Voice API routes for real-time audio communication.
Handles WebSocket connections for streaming audio and conversation management.
"""

import asyncio
import base64
import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.asr import TranscriberFactory, VADProcessor
from src.core.handoff import get_handoff_manager
from src.core.nlu import get_nlu_pipeline
from src.core.orchestrator import DialogueManager
from src.core.tts import SynthesizerFactory
from src.models import ConversationState, DriverInfo, HandoffTrigger, Language

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/voice", tags=["Voice"])


# Request/Response Models
class StartCallRequest(BaseModel):
    """Request to start a new call session."""
    
    phone_number: str = Field(..., description="Caller phone number")
    call_id: str | None = Field(default=None, description="External call ID")
    driver_id: str | None = Field(default=None, description="Known driver ID")
    preferred_language: str = Field(default="hi-en", description="Preferred language code")


class StartCallResponse(BaseModel):
    """Response with session details."""
    
    session_id: str
    call_id: str
    greeting_text: str
    greeting_audio_base64: str | None = None


class TextInputRequest(BaseModel):
    """Request for text-based conversation (testing/fallback)."""
    
    call_id: str
    text: str


class ConversationResponse(BaseModel):
    """Response from conversation turn."""
    
    response_text: str
    response_audio_base64: str | None = None
    intent: str | None = None
    confidence: float | None = None
    sentiment: str | None = None
    handoff_triggered: bool = False
    handoff_info: dict | None = None
    session_state: str


class EndCallRequest(BaseModel):
    """Request to end a call session."""
    
    call_id: str
    reason: str = "completed"


# WebSocket message types
class WSMessageType:
    """WebSocket message types."""
    
    AUDIO = "audio"
    TEXT = "text"
    TRANSCRIPT = "transcript"
    RESPONSE = "response"
    HANDOFF = "handoff"
    ERROR = "error"
    END = "end"
    PING = "ping"
    PONG = "pong"


# Connection manager for WebSocket sessions
class ConnectionManager:
    """Manages WebSocket connections for voice sessions."""
    
    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self.call_data: dict[str, dict] = {}
    
    async def connect(self, call_id: str, websocket: WebSocket) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self.active_connections[call_id] = websocket
        logger.info("websocket_connected", call_id=call_id)
    
    def disconnect(self, call_id: str) -> None:
        """Remove a WebSocket connection."""
        if call_id in self.active_connections:
            del self.active_connections[call_id]
            logger.info("websocket_disconnected", call_id=call_id)
    
    async def send_message(self, call_id: str, message: dict) -> None:
        """Send message to specific connection."""
        if call_id in self.active_connections:
            await self.active_connections[call_id].send_json(message)
    
    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connections."""
        for connection in self.active_connections.values():
            await connection.send_json(message)


# Initialize components
manager = ConnectionManager()
settings = get_settings()

# Singleton dialogue manager
_dialogue_manager: DialogueManager | None = None


def get_dialogue_manager() -> DialogueManager:
    """Get or create dialogue manager singleton."""
    global _dialogue_manager
    if _dialogue_manager is None:
        _dialogue_manager = DialogueManager()
    return _dialogue_manager


@router.post("/start", response_model=StartCallResponse)
async def start_call(request: StartCallRequest):
    """
    Start a new voice call session.
    
    This creates a conversation session and returns a greeting.
    """
    call_id = request.call_id or str(uuid4())
    
    # Map language code
    lang_map = {
        "hi-IN": Language.HINDI,
        "hi-en": Language.HINGLISH,
        "en-IN": Language.ENGLISH_INDIA,
        "bn-IN": Language.BENGALI,
    }
    language = lang_map.get(request.preferred_language, Language.HINGLISH)
    
    # Create driver info
    driver_info = DriverInfo(
        phone_number=request.phone_number,
        driver_id=request.driver_id,
        preferred_language=language
    )
    
    # Create session
    dialogue_manager = get_dialogue_manager()
    session = await dialogue_manager.create_session(
        call_id=call_id,
        phone_number=request.phone_number,
        driver_info=driver_info
    )
    
    # Generate greeting
    greeting_text = dialogue_manager.response_generator.generate_greeting(language)
    
    # Synthesize greeting audio
    greeting_audio_base64 = None
    try:
        synthesizer = SynthesizerFactory.create("polly")
        result = await synthesizer.synthesize(greeting_text, language)
        greeting_audio_base64 = base64.b64encode(result.audio_data).decode()
    except Exception as e:
        logger.warning("greeting_synthesis_failed", error=str(e))
    
    logger.info(
        "call_started",
        call_id=call_id,
        session_id=str(session.id),
        language=language.value
    )
    
    return StartCallResponse(
        session_id=str(session.id),
        call_id=call_id,
        greeting_text=greeting_text,
        greeting_audio_base64=greeting_audio_base64
    )


@router.post("/message", response_model=ConversationResponse)
async def process_text_message(request: TextInputRequest):
    """
    Process a text message (for testing or text fallback).
    
    In production, most interactions will use the WebSocket endpoint.
    """
    dialogue_manager = get_dialogue_manager()
    
    session = await dialogue_manager.get_session(request.call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Process the turn
    response_text, handoff_decision = await dialogue_manager.process_turn(
        call_id=request.call_id,
        user_text=request.text
    )
    
    # Synthesize response audio
    response_audio_base64 = None
    try:
        synthesizer = SynthesizerFactory.create("polly")
        result = await synthesizer.synthesize(
            response_text,
            session.driver.preferred_language
        )
        response_audio_base64 = base64.b64encode(result.audio_data).decode()
    except Exception as e:
        logger.warning("response_synthesis_failed", error=str(e))
    
    # Get handoff info if triggered
    handoff_info = None
    if handoff_decision and handoff_decision.should_handoff:
        handoff_manager = get_handoff_manager()
        alert = await handoff_manager.initiate_handoff(
            session,
            handoff_decision.trigger
        )
        handoff_info = {
            "alert_id": str(alert.id),
            "trigger": alert.trigger.value,
            "priority": alert.priority.value,
            "queue_position": alert.queue_position,
        }
    
    # Get last NLU result
    last_turn = session.turns[-2] if len(session.turns) >= 2 else None  # -2 because -1 is bot response
    intent = None
    confidence = None
    sentiment = None
    
    if last_turn and last_turn.nlu_result:
        intent = last_turn.nlu_result.intent.intent.value
        confidence = last_turn.nlu_result.intent.confidence
        sentiment = last_turn.nlu_result.sentiment.label.value
    
    return ConversationResponse(
        response_text=response_text,
        response_audio_base64=response_audio_base64,
        intent=intent,
        confidence=confidence,
        sentiment=sentiment,
        handoff_triggered=handoff_decision.should_handoff if handoff_decision else False,
        handoff_info=handoff_info,
        session_state=session.state.value
    )


@router.post("/end")
async def end_call(request: EndCallRequest):
    """End a voice call session."""
    dialogue_manager = get_dialogue_manager()
    
    session = await dialogue_manager.get_session(request.call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if there's a pending handoff to cancel
    if session.handoff_triggered:
        handoff_manager = get_handoff_manager()
        await handoff_manager.cancel_handoff(request.call_id, request.reason)
    
    await dialogue_manager.end_session(request.call_id)
    
    # Disconnect WebSocket if connected
    manager.disconnect(request.call_id)
    
    return {"status": "ended", "call_id": request.call_id}


@router.websocket("/stream/{call_id}")
async def voice_stream(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Protocol:
    1. Client connects with call_id
    2. Client sends audio chunks (base64 encoded PCM)
    3. Server streams back transcriptions and responses
    
    Message format:
    {
        "type": "audio|text|end",
        "data": "<base64 audio or text>",
        "timestamp": <unix timestamp>
    }
    
    Response format:
    {
        "type": "transcript|response|handoff|error",
        "data": {...},
        "timestamp": <unix timestamp>
    }
    """
    await manager.connect(call_id, websocket)
    
    dialogue_manager = get_dialogue_manager()
    session = await dialogue_manager.get_session(call_id)
    
    if not session:
        await websocket.send_json({
            "type": WSMessageType.ERROR,
            "data": {"error": "Session not found. Call /voice/start first."},
            "timestamp": datetime.utcnow().timestamp()
        })
        await websocket.close()
        return
    
    # Initialize components
    transcriber = TranscriberFactory.create("whisper")  # or "aws" for AWS Transcribe
    synthesizer = SynthesizerFactory.create("polly")
    vad = VADProcessor(aggressiveness=settings.voicebot.vad_aggressiveness)
    
    # Audio buffer for streaming transcription
    audio_buffer = bytearray()
    last_speech_time = datetime.utcnow()
    silence_threshold_ms = settings.voicebot.silence_timeout_ms
    
    try:
        while True:
            # Receive message
            try:
                raw_data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # Connection timeout
                )
                message = json.loads(raw_data)
            except asyncio.TimeoutError:
                # Send ping
                await websocket.send_json({
                    "type": WSMessageType.PING,
                    "timestamp": datetime.utcnow().timestamp()
                })
                continue
            except json.JSONDecodeError:
                continue
            
            msg_type = message.get("type")
            msg_data = message.get("data")
            
            if msg_type == WSMessageType.PING:
                await websocket.send_json({
                    "type": WSMessageType.PONG,
                    "timestamp": datetime.utcnow().timestamp()
                })
                continue
            
            if msg_type == WSMessageType.END:
                break
            
            if msg_type == WSMessageType.AUDIO:
                # Decode audio chunk
                try:
                    audio_chunk = base64.b64decode(msg_data)
                except Exception:
                    continue
                
                # Check for speech
                is_speech = vad.is_speech(audio_chunk)
                
                if is_speech:
                    audio_buffer.extend(audio_chunk)
                    last_speech_time = datetime.utcnow()
                else:
                    # Check for end of utterance
                    silence_duration = (datetime.utcnow() - last_speech_time).total_seconds() * 1000
                    
                    if silence_duration >= silence_threshold_ms and len(audio_buffer) > 0:
                        # Process complete utterance
                        try:
                            # Transcribe
                            transcription = await transcriber.transcribe(
                                bytes(audio_buffer),
                                session.driver.preferred_language
                            )
                            
                            if transcription.text.strip():
                                # Send transcript
                                await websocket.send_json({
                                    "type": WSMessageType.TRANSCRIPT,
                                    "data": {
                                        "text": transcription.text,
                                        "confidence": transcription.confidence,
                                        "language": transcription.language.value
                                    },
                                    "timestamp": datetime.utcnow().timestamp()
                                })
                                
                                # Process through dialogue manager
                                response_text, handoff_decision = await dialogue_manager.process_turn(
                                    call_id=call_id,
                                    user_text=transcription.text
                                )
                                
                                # Synthesize response
                                audio_result = await synthesizer.synthesize(
                                    response_text,
                                    session.driver.preferred_language
                                )
                                
                                # Send response
                                await websocket.send_json({
                                    "type": WSMessageType.RESPONSE,
                                    "data": {
                                        "text": response_text,
                                        "audio": base64.b64encode(audio_result.audio_data).decode()
                                    },
                                    "timestamp": datetime.utcnow().timestamp()
                                })
                                
                                # Handle handoff
                                if handoff_decision and handoff_decision.should_handoff:
                                    handoff_manager = get_handoff_manager()
                                    alert = await handoff_manager.initiate_handoff(
                                        session,
                                        handoff_decision.trigger
                                    )
                                    
                                    await websocket.send_json({
                                        "type": WSMessageType.HANDOFF,
                                        "data": {
                                            "alert_id": str(alert.id),
                                            "trigger": alert.trigger.value,
                                            "priority": alert.priority.value,
                                            "summary": alert.issue_summary,
                                            "queue_position": alert.queue_position
                                        },
                                        "timestamp": datetime.utcnow().timestamp()
                                    })
                            
                        except Exception as e:
                            logger.error("processing_error", error=str(e))
                            await websocket.send_json({
                                "type": WSMessageType.ERROR,
                                "data": {"error": "Processing failed"},
                                "timestamp": datetime.utcnow().timestamp()
                            })
                        
                        # Clear buffer
                        audio_buffer = bytearray()
            
            elif msg_type == WSMessageType.TEXT:
                # Direct text input (for testing)
                response_text, handoff_decision = await dialogue_manager.process_turn(
                    call_id=call_id,
                    user_text=msg_data
                )
                
                audio_result = await synthesizer.synthesize(
                    response_text,
                    session.driver.preferred_language
                )
                
                await websocket.send_json({
                    "type": WSMessageType.RESPONSE,
                    "data": {
                        "text": response_text,
                        "audio": base64.b64encode(audio_result.audio_data).decode()
                    },
                    "timestamp": datetime.utcnow().timestamp()
                })
                
                if handoff_decision and handoff_decision.should_handoff:
                    handoff_manager = get_handoff_manager()
                    alert = await handoff_manager.initiate_handoff(
                        session,
                        handoff_decision.trigger
                    )
                    
                    await websocket.send_json({
                        "type": WSMessageType.HANDOFF,
                        "data": {
                            "alert_id": str(alert.id),
                            "trigger": alert.trigger.value,
                        },
                        "timestamp": datetime.utcnow().timestamp()
                    })
    
    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected", call_id=call_id)
    except Exception as e:
        logger.error("websocket_error", call_id=call_id, error=str(e))
    finally:
        manager.disconnect(call_id)
        
        # End session if still active
        session = await dialogue_manager.get_session(call_id)
        if session and session.state == ConversationState.ACTIVE:
            await dialogue_manager.end_session(call_id)


@router.get("/session/{call_id}")
async def get_session_status(call_id: str):
    """Get current session status."""
    dialogue_manager = get_dialogue_manager()
    session = await dialogue_manager.get_session(call_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": str(session.id),
        "call_id": session.call_id,
        "state": session.state.value,
        "turns": len(session.turns),
        "current_intent": session.current_intent.value if session.current_intent else None,
        "handoff_triggered": session.handoff_triggered,
        "metrics": {
            "average_confidence": session.metrics.average_confidence,
            "average_sentiment": session.metrics.average_sentiment,
            "clarification_count": session.metrics.clarification_count,
        },
        "started_at": session.started_at.isoformat(),
        "duration_seconds": (datetime.utcnow() - session.started_at).total_seconds()
    }
