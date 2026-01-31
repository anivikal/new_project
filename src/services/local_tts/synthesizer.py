"""
Local Text-to-Speech (TTS) module using gTTS (Google Text-to-Speech).
Provides free TTS as an alternative to AWS Polly.
"""

import asyncio
import hashlib
import io
import re
from dataclasses import dataclass
from typing import AsyncIterator

import structlog

from src.config import get_settings
from src.models import Language

logger = structlog.get_logger(__name__)


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis."""
    
    audio_data: bytes
    audio_format: str  # mp3, wav
    sample_rate: int
    duration_ms: int | None = None
    voice_id: str | None = None


class LocalTTSSynthesizer:
    """
    Local TTS synthesizer using gTTS (Google Text-to-Speech).
    Free alternative to AWS Polly for Hindi/Hinglish.
    """
    
    # Language mapping for gTTS
    LANGUAGE_MAP = {
        Language.HINDI: "hi",
        Language.HINGLISH: "hi",  # gTTS handles Hinglish reasonably well with Hindi
        Language.ENGLISH_INDIA: "en-in",
        Language.BENGALI: "bn",
        Language.TAMIL: "ta",
    }
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: dict[str, bytes] = {}  # Simple in-memory cache
        self._cache_max_size = 500
    
    def _get_cache_key(self, text: str, lang: str) -> str:
        """Generate cache key for text."""
        content = f"{text}:{lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def synthesize(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> SynthesisResult:
        """
        Synthesize speech using gTTS.
        
        Args:
            text: Text to synthesize
            language: Target language
            voice_id: Ignored for gTTS (no voice selection)
        
        Returns:
            SynthesisResult with MP3 audio data
        """
        from gtts import gTTS
        
        # Map language
        lang_code = self.LANGUAGE_MAP.get(language, "hi")
        
        # Check cache
        cache_key = self._get_cache_key(text, lang_code)
        if cache_key in self._cache:
            logger.debug("tts_cache_hit", cache_key=cache_key[:8])
            return SynthesisResult(
                audio_data=self._cache[cache_key],
                audio_format="mp3",
                sample_rate=24000,
                voice_id="gtts"
            )
        
        # Process text for better pronunciation
        processed_text = self._process_text(text, language)
        
        logger.info(
            "synthesizing_speech_gtts",
            text_length=len(text),
            language=lang_code
        )
        
        try:
            # Run gTTS in thread pool (it's blocking)
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                processed_text,
                lang_code
            )
            
            # Cache the result
            self._cache[cache_key] = audio_data
            
            # Limit cache size
            if len(self._cache) > self._cache_max_size:
                # Remove oldest entries (FIFO-ish)
                keys_to_remove = list(self._cache.keys())[:100]
                for key in keys_to_remove:
                    del self._cache[key]
            
            return SynthesisResult(
                audio_data=audio_data,
                audio_format="mp3",
                sample_rate=24000,
                voice_id="gtts"
            )
            
        except Exception as e:
            logger.error("gtts_synthesis_failed", error=str(e))
            raise
    
    def _synthesize_sync(self, text: str, lang_code: str) -> bytes:
        """Synchronous gTTS synthesis."""
        from gtts import gTTS
        
        # Create gTTS object
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to bytes
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
    
    def _process_text(self, text: str, language: Language) -> str:
        """Process text for better TTS pronunciation."""
        processed = text
        
        # Handle currency
        processed = re.sub(r'₹\s*(\d+)', r'\1 rupees', processed)
        processed = re.sub(r'Rs\.?\s*(\d+)', r'\1 rupees', processed)
        
        # Handle abbreviations
        abbreviations = {
            "GST": "G S T",
            "ID": "I D", 
            "DSK": "D S K",
            "OTP": "O T P",
            "EV": "E V",
            "km": "kilometers",
            "hrs": "hours",
            "mins": "minutes",
        }
        
        for abbr, expansion in abbreviations.items():
            processed = re.sub(rf'\b{abbr}\b', expansion, processed, flags=re.IGNORECASE)
        
        return processed
    
    async def synthesize_stream(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesized speech.
        
        For gTTS, we synthesize in chunks for longer text.
        """
        if len(text) <= 300:
            result = await self.synthesize(text, language, voice_id)
            yield result.audio_data
            return
        
        # Split into sentences for streaming
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        
        for sentence in sentences:
            if sentence.strip():
                result = await self.synthesize(sentence, language, voice_id)
                yield result.audio_data


class PyttsxSynthesizer:
    """
    Alternative local TTS using pyttsx3.
    Works offline but quality may vary.
    """
    
    def __init__(self) -> None:
        self._engine = None
    
    def _get_engine(self):
        """Lazy load pyttsx3 engine."""
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            # Set properties for Hindi-like pronunciation
            self._engine.setProperty('rate', 150)  # Speed
            self._engine.setProperty('volume', 0.9)  # Volume
        return self._engine
    
    async def synthesize(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> SynthesisResult:
        """Synthesize using pyttsx3."""
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        return SynthesisResult(
            audio_data=audio_data,
            audio_format="wav",
            sample_rate=22050,
            voice_id="pyttsx3"
        )
    
    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous pyttsx3 synthesis."""
        engine = self._get_engine()
        
        # Save to temporary file then read
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def synthesize_stream(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech."""
        result = await self.synthesize(text, language, voice_id)
        yield result.audio_data


class HybridTTSSynthesizer:
    """
    Hybrid TTS that tries multiple backends.
    Falls back gracefully if one fails.
    """
    
    def __init__(self) -> None:
        self._gtts = LocalTTSSynthesizer()
        self._pyttsx = None  # Lazy load
        self.settings = get_settings()
    
    async def synthesize(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> SynthesisResult:
        """
        Synthesize using best available backend.
        
        Priority:
        1. gTTS (better quality, requires internet)
        2. pyttsx3 (offline fallback)
        """
        # Try gTTS first
        try:
            return await self._gtts.synthesize(text, language, voice_id)
        except Exception as e:
            logger.warning("gtts_failed_trying_pyttsx", error=str(e))
        
        # Fallback to pyttsx3
        try:
            if self._pyttsx is None:
                self._pyttsx = PyttsxSynthesizer()
            return await self._pyttsx.synthesize(text, language, voice_id)
        except Exception as e:
            logger.error("all_tts_backends_failed", error=str(e))
            # Return empty audio
            return SynthesisResult(
                audio_data=b"",
                audio_format="mp3",
                sample_rate=24000,
                voice_id="none"
            )
    
    async def synthesize_stream(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech."""
        try:
            async for chunk in self._gtts.synthesize_stream(text, language, voice_id):
                yield chunk
        except Exception as e:
            logger.warning("gtts_stream_failed", error=str(e))
            result = await self.synthesize(text, language, voice_id)
            yield result.audio_data


# Pre-defined phrases for caching
PRESYNTHESIZE_PHRASES = {
    Language.HINGLISH: [
        "Namaste! Main Battery Smart ka AI assistant hoon.",
        "Aapki kaise help kar sakta hoon?",
        "Please thoda wait karein.",
        "Main aapko human agent se connect kar raha hoon.",
        "Kya aur kuch help chahiye?",
        "Thank you! Battery Smart choose karne ke liye.",
        "Sorry, main samajh nahi paya. Kya aap dobara bata sakte hain?",
        "Main samajh sakta hoon. Main aapki madad karne ke liye yahan hoon.",
    ],
    Language.HINDI: [
        "नमस्ते! मैं Battery Smart का AI assistant हूँ।",
        "आपकी कैसे मदद कर सकता हूँ?",
        "कृपया थोड़ा इंतज़ार करें।",
        "धन्यवाद!",
    ],
}


async def presynthesize_common_phrases(synthesizer: LocalTTSSynthesizer) -> dict[str, bytes]:
    """Pre-synthesize common phrases for faster response."""
    cache = {}
    
    for language, phrases in PRESYNTHESIZE_PHRASES.items():
        for phrase in phrases:
            try:
                result = await synthesizer.synthesize(phrase, language)
                cache_key = f"{phrase[:20]}:{language.value}"
                cache[cache_key] = result.audio_data
                logger.debug("presynthesized_phrase", phrase=phrase[:30])
            except Exception as e:
                logger.warning("presynthesize_failed", phrase=phrase[:30], error=str(e))
    
    return cache


# Singleton instance
_local_tts: LocalTTSSynthesizer | None = None


def get_local_tts() -> LocalTTSSynthesizer:
    """Get local TTS instance."""
    global _local_tts
    if _local_tts is None:
        _local_tts = LocalTTSSynthesizer()
    return _local_tts
