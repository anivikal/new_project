"""
Text-to-Speech (TTS) module for generating Hindi/Hinglish audio responses.
Uses AWS Polly for high-quality neural voices.
"""

import asyncio
import hashlib
import io
import os
import re
from abc import ABC, abstractmethod
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
    audio_format: str  # mp3, pcm, ogg_vorbis
    sample_rate: int
    duration_ms: int | None = None
    voice_id: str | None = None


@dataclass
class SSMLTag:
    """SSML tag for speech customization."""
    
    tag: str
    attributes: dict | None = None
    content: str = ""


class BaseSynthesizer(ABC):
    """Abstract base class for TTS synthesis."""
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> SynthesisResult:
        """Synthesize speech from text."""
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech."""
        pass


class SSMLBuilder:
    """Build SSML markup for enhanced speech synthesis."""
    
    def __init__(self) -> None:
        self.parts: list[str] = []
    
    def add_text(self, text: str) -> "SSMLBuilder":
        """Add plain text."""
        # Escape XML special characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        self.parts.append(text)
        return self
    
    def add_break(self, time_ms: int = 500) -> "SSMLBuilder":
        """Add pause/break."""
        self.parts.append(f'<break time="{time_ms}ms"/>')
        return self
    
    def add_emphasis(self, text: str, level: str = "moderate") -> "SSMLBuilder":
        """Add emphasized text. Level: strong, moderate, reduced."""
        self.parts.append(f'<emphasis level="{level}">{text}</emphasis>')
        return self
    
    def add_prosody(
        self,
        text: str,
        rate: str | None = None,
        pitch: str | None = None,
        volume: str | None = None
    ) -> "SSMLBuilder":
        """Add prosody control."""
        attrs = []
        if rate:
            attrs.append(f'rate="{rate}"')
        if pitch:
            attrs.append(f'pitch="{pitch}"')
        if volume:
            attrs.append(f'volume="{volume}"')
        
        attr_str = " ".join(attrs)
        self.parts.append(f'<prosody {attr_str}>{text}</prosody>')
        return self
    
    def add_say_as(self, text: str, interpret_as: str) -> "SSMLBuilder":
        """Add say-as for special interpretation (number, date, etc.)."""
        self.parts.append(f'<say-as interpret-as="{interpret_as}">{text}</say-as>')
        return self
    
    def add_phoneme(self, text: str, phoneme: str, alphabet: str = "ipa") -> "SSMLBuilder":
        """Add phonetic pronunciation."""
        self.parts.append(f'<phoneme alphabet="{alphabet}" ph="{phoneme}">{text}</phoneme>')
        return self
    
    def add_lang(self, text: str, lang: str) -> "SSMLBuilder":
        """Wrap text in language tag for code-switching."""
        self.parts.append(f'<lang xml:lang="{lang}">{text}</lang>')
        return self
    
    def build(self) -> str:
        """Build final SSML document."""
        content = "".join(self.parts)
        return f'<speak>{content}</speak>'


class HinglishTextProcessor:
    """
    Process Hinglish text for optimal TTS synthesis.
    Handles Hindi-English code switching and pronunciation.
    """
    
    # Words that should be pronounced in English
    ENGLISH_WORDS = {
        "okay", "ok", "yes", "no", "sorry", "please", "thank", "thanks",
        "battery", "swap", "station", "plan", "subscription", "invoice",
        "hub", "available", "history", "status", "premium", "basic",
        "monthly", "weekly", "daily", "driver", "id", "number",
        "gst", "amount", "total", "charge", "refund"
    }
    
    # Hindi words with pronunciation hints
    PRONUNCIATION_MAP = {
        "namaste": "nuh-MUSS-tay",
        "dhanyavaad": "DHUN-ya-vaad",
        "kripya": "KRIP-ya",
    }
    
    def process_for_tts(self, text: str, language: Language) -> str:
        """
        Process text for TTS synthesis.
        
        Handles:
        - Code-switching markers
        - Pronunciation improvements
        - Number formatting
        """
        processed = text
        
        # Handle numbers (speak as words for Hindi)
        if language in [Language.HINDI, Language.HINGLISH]:
            processed = self._process_numbers(processed)
        
        # Handle currency
        processed = self._process_currency(processed)
        
        # Handle abbreviations
        processed = self._process_abbreviations(processed)
        
        return processed
    
    def build_ssml(self, text: str, language: Language) -> str:
        """Build SSML with proper language tags for code-switching."""
        builder = SSMLBuilder()
        
        # For Hinglish, identify English words and wrap appropriately
        if language == Language.HINGLISH:
            words = text.split()
            current_segment = []
            current_lang = "hi-IN"
            
            for word in words:
                word_lower = word.lower().rstrip(".,!?")
                
                # Check if English word
                is_english = (
                    word_lower in self.ENGLISH_WORDS or
                    word_lower.isascii() and len(word_lower) > 2
                )
                
                target_lang = "en-IN" if is_english else "hi-IN"
                
                if target_lang != current_lang and current_segment:
                    # Flush current segment
                    segment_text = " ".join(current_segment)
                    if current_lang == "en-IN":
                        builder.add_lang(segment_text, "en-IN")
                    else:
                        builder.add_text(segment_text + " ")
                    current_segment = []
                
                current_segment.append(word)
                current_lang = target_lang
            
            # Flush remaining
            if current_segment:
                segment_text = " ".join(current_segment)
                if current_lang == "en-IN":
                    builder.add_lang(segment_text, "en-IN")
                else:
                    builder.add_text(segment_text)
            
            return builder.build()
        
        # For pure Hindi or English, simpler processing
        builder.add_text(text)
        return builder.build()
    
    def _process_numbers(self, text: str) -> str:
        """Process numbers for Hindi pronunciation."""
        # Handle currency amounts specially
        def replace_number(match):
            num = match.group()
            # Keep as is for TTS to handle
            return num
        
        return re.sub(r'\d+', replace_number, text)
    
    def _process_currency(self, text: str) -> str:
        """Process currency symbols."""
        # Replace ₹ with "rupees" for clearer pronunciation
        text = re.sub(r'₹\s*(\d+)', r'\1 rupees', text)
        text = re.sub(r'Rs\.?\s*(\d+)', r'\1 rupees', text)
        return text
    
    def _process_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
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
            text = re.sub(rf'\b{abbr}\b', expansion, text, flags=re.IGNORECASE)
        
        return text


class AWSPollySynthesizer(BaseSynthesizer):
    """AWS Polly-based TTS synthesizer."""
    
    # Voice mapping for different languages
    VOICE_MAP = {
        Language.HINDI: "Aditi",       # Neural Hindi voice
        Language.HINGLISH: "Kajal",    # Neural Hindi voice (better for Hinglish)
        Language.ENGLISH_INDIA: "Kajal",  # Indian English
        Language.BENGALI: "Aditi",     # Fallback to Hindi
        Language.TAMIL: "Aditi",       # Fallback to Hindi
    }
    
    # Neural voice engines
    NEURAL_VOICES = {"Kajal", "Aditi"}
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self.text_processor = HinglishTextProcessor()
        self._client = None
        self._cache: dict[str, bytes] = {}  # Simple in-memory cache
    
    async def _get_client(self):
        """Get or create Polly client."""
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            self._client = await session.client(
                "polly",
                region_name=self.settings.aws.region
            ).__aenter__()
        return self._client
    
    def _get_cache_key(self, text: str, voice_id: str) -> str:
        """Generate cache key for text."""
        content = f"{text}:{voice_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def synthesize(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> SynthesisResult:
        """Synthesize speech using AWS Polly."""
        # Select voice
        voice = voice_id or self.VOICE_MAP.get(language, "Kajal")
        
        # Check cache
        cache_key = self._get_cache_key(text, voice)
        if cache_key in self._cache:
            logger.debug("tts_cache_hit", cache_key=cache_key)
            return SynthesisResult(
                audio_data=self._cache[cache_key],
                audio_format="mp3",
                sample_rate=24000,
                voice_id=voice
            )
        
        # Process text
        processed_text = self.text_processor.process_for_tts(text, language)
        
        # Build SSML if needed
        use_ssml = language == Language.HINGLISH or "<" in text
        if use_ssml:
            text_content = self.text_processor.build_ssml(processed_text, language)
            text_type = "ssml"
        else:
            text_content = processed_text
            text_type = "text"
        
        # Determine engine
        engine = "neural" if voice in self.NEURAL_VOICES else "standard"
        
        logger.info(
            "synthesizing_speech",
            text_length=len(text),
            voice=voice,
            engine=engine,
            language=language.value
        )
        
        try:
            client = await self._get_client()
            
            response = await client.synthesize_speech(
                Text=text_content,
                TextType=text_type,
                OutputFormat="mp3",
                VoiceId=voice,
                Engine=engine,
                LanguageCode=self._get_language_code(language),
                SampleRate="24000"
            )
            
            # Read audio stream
            audio_stream = response["AudioStream"]
            audio_data = await audio_stream.read()
            
            # Cache the result
            self._cache[cache_key] = audio_data
            
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries
                keys = list(self._cache.keys())[:100]
                for key in keys:
                    del self._cache[key]
            
            return SynthesisResult(
                audio_data=audio_data,
                audio_format="mp3",
                sample_rate=24000,
                voice_id=voice
            )
            
        except Exception as e:
            logger.error("polly_synthesis_failed", error=str(e))
            raise
    
    async def synthesize_stream(
        self,
        text: str,
        language: Language,
        voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech for low-latency playback."""
        # For Polly, we synthesize in chunks for longer text
        # Short text can be synthesized in one go
        
        if len(text) <= 500:
            result = await self.synthesize(text, language, voice_id)
            yield result.audio_data
            return
        
        # Split into sentences for streaming
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        
        for sentence in sentences:
            if sentence.strip():
                result = await self.synthesize(sentence, language, voice_id)
                yield result.audio_data
    
    def _get_language_code(self, language: Language) -> str:
        """Map to Polly language code."""
        mapping = {
            Language.HINDI: "hi-IN",
            Language.HINGLISH: "hi-IN",
            Language.ENGLISH_INDIA: "en-IN",
            Language.BENGALI: "hi-IN",  # Fallback
            Language.TAMIL: "hi-IN",    # Fallback
        }
        return mapping.get(language, "hi-IN")


class TTSCache:
    """
    Caching layer for TTS results.
    In production, use Redis or S3 for persistence.
    """
    
    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self._cache: dict[str, tuple[bytes, float]] = {}  # key -> (audio, timestamp)
    
    def get(self, key: str) -> bytes | None:
        """Get cached audio."""
        if key in self._cache:
            return self._cache[key][0]
        return None
    
    def set(self, key: str, audio: bytes) -> None:
        """Cache audio data."""
        import time
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            # Remove oldest 10%
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            for k in sorted_keys[:self.max_size // 10]:
                del self._cache[k]
        
        self._cache[key] = (audio, time.time())
    
    @staticmethod
    def make_key(text: str, voice: str, language: str) -> str:
        """Generate cache key."""
        content = f"{text}:{voice}:{language}"
        return hashlib.sha256(content.encode()).hexdigest()


class SynthesizerFactory:
    """Factory for creating TTS synthesizer instances."""
    
    @staticmethod
    def create(provider: str = "polly") -> BaseSynthesizer:
        """Create a synthesizer instance."""
        if provider == "polly":
            return AWSPollySynthesizer()
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")


# Pre-synthesize common responses for lower latency
PRESYNTHESIZE_PHRASES = {
    Language.HINGLISH: [
        "Namaste! Main Battery Smart ka AI assistant hoon.",
        "Aapki kaise help kar sakta hoon?",
        "Please thoda wait karein.",
        "Main aapko human agent se connect kar raha hoon.",
        "Kya aur kuch help chahiye?",
        "Thank you! Battery Smart choose karne ke liye.",
        "Sorry, main samajh nahi paya. Kya aap dobara bata sakte hain?",
    ],
    Language.HINDI: [
        "नमस्ते! मैं Battery Smart का AI assistant हूँ।",
        "आपकी कैसे मदद कर सकता हूँ?",
        "कृपया थोड़ा इंतज़ार करें।",
        "धन्यवाद!",
    ],
}


async def presynthesize_common_phrases(synthesizer: BaseSynthesizer) -> dict[str, bytes]:
    """Pre-synthesize common phrases for faster response."""
    cache = {}
    
    for language, phrases in PRESYNTHESIZE_PHRASES.items():
        for phrase in phrases:
            try:
                result = await synthesizer.synthesize(phrase, language)
                cache_key = TTSCache.make_key(phrase, "default", language.value)
                cache[cache_key] = result.audio_data
            except Exception as e:
                logger.warning("presynthesize_failed", phrase=phrase[:30], error=str(e))
    
    return cache
