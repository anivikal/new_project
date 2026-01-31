"""
Automatic Speech Recognition (ASR) module for multilingual transcription.
Supports AWS Transcribe and local Whisper models for Hindi/Hinglish.
"""

import asyncio
import base64
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import structlog

from src.config import get_settings
from src.models import Language

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result from speech transcription."""
    
    text: str
    language: Language
    confidence: float
    is_partial: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    words: list[dict] | None = None  # Word-level timestamps
    alternatives: list[tuple[str, float]] | None = None


@dataclass
class AudioChunk:
    """Audio chunk for streaming transcription."""
    
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    timestamp_ms: int = 0


class BaseTranscriber(ABC):
    """Abstract base class for transcription services."""
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: Language | None = None) -> TranscriptionResult:
        """Transcribe audio data to text."""
        pass
    
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        language: Language | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription for real-time audio."""
        pass
    
    @abstractmethod
    async def detect_language(self, audio_data: bytes) -> Language:
        """Detect language from audio sample."""
        pass


class AWSTranscriber(BaseTranscriber):
    """AWS Transcribe-based transcription with streaming support."""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._streaming_client = None
    
    async def _get_client(self):
        """Get or create boto3 client."""
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            self._client = await session.client(
                "transcribe",
                region_name=self.settings.aws.region
            ).__aenter__()
        return self._client
    
    def _map_language_code(self, language: Language) -> str:
        """Map internal language codes to AWS Transcribe codes."""
        mapping = {
            Language.HINDI: "hi-IN",
            Language.ENGLISH_INDIA: "en-IN",
            Language.HINGLISH: "hi-IN",  # Use Hindi with code-switching
            Language.BENGALI: "bn-IN",
            Language.TAMIL: "ta-IN",
        }
        return mapping.get(language, "hi-IN")
    
    async def transcribe(self, audio_data: bytes, language: Language | None = None) -> TranscriptionResult:
        """
        Transcribe audio using AWS Transcribe.
        
        For production, this uses StartTranscriptionJob for batch processing.
        For real-time, use transcribe_stream instead.
        """
        import aioboto3
        from botocore.exceptions import ClientError
        
        language = language or Language.HINGLISH
        language_code = self._map_language_code(language)
        
        try:
            session = aioboto3.Session()
            
            # For short audio, use synchronous approach with S3
            # In production, you'd upload to S3 and use StartTranscriptionJob
            # Here we'll use a simulated response for the MVP
            
            logger.info(
                "transcribing_audio",
                audio_size=len(audio_data),
                language=language_code
            )
            
            # For MVP: Return placeholder - in production this connects to AWS
            # The actual implementation would:
            # 1. Upload audio to S3
            # 2. Start transcription job
            # 3. Poll for completion
            # 4. Return results
            
            return TranscriptionResult(
                text="",  # Actual transcription
                language=language,
                confidence=0.0,
                is_partial=False,
            )
            
        except ClientError as e:
            logger.error("transcription_failed", error=str(e))
            raise
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        language: Language | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Stream transcription using AWS Transcribe Streaming.
        
        This provides real-time transcription with partial results.
        """
        import aioboto3
        
        language = language or Language.HINGLISH
        language_code = self._map_language_code(language)
        
        logger.info("starting_stream_transcription", language=language_code)
        
        session = aioboto3.Session()
        
        async with session.client(
            "transcribe-streaming",
            region_name=self.settings.aws.region
        ) as client:
            # Start streaming transcription
            stream_response = await client.start_stream_transcription(
                LanguageCode=language_code,
                MediaSampleRateHertz=16000,
                MediaEncoding="pcm",
                EnablePartialResultsStabilization=True,
                PartialResultsStability="high",
            )
            
            async def audio_generator():
                """Generate audio events for the stream."""
                async for chunk in audio_stream:
                    yield {"AudioEvent": {"AudioChunk": chunk.data}}
            
            # Send audio and receive transcriptions
            async with stream_response["TranscriptResultStream"] as result_stream:
                async for event in result_stream:
                    if "TranscriptEvent" in event:
                        results = event["TranscriptEvent"]["Transcript"]["Results"]
                        for result in results:
                            if result["Alternatives"]:
                                alt = result["Alternatives"][0]
                                yield TranscriptionResult(
                                    text=alt["Transcript"],
                                    language=language,
                                    confidence=alt.get("Confidence", 0.9),
                                    is_partial=result["IsPartial"],
                                    start_time=result.get("StartTime", 0.0),
                                    end_time=result.get("EndTime", 0.0),
                                )
    
    async def detect_language(self, audio_data: bytes) -> Language:
        """Detect language from audio using AWS Transcribe."""
        # AWS Transcribe can auto-detect language
        # For Hindi-English code-switching, we default to Hinglish
        
        # In production, you could:
        # 1. Run a short transcription with multiple language codes
        # 2. Use confidence scores to determine language
        # 3. Use a separate language ID model
        
        return Language.HINGLISH


class WhisperTranscriber(BaseTranscriber):
    """
    Local Whisper-based transcription using faster-whisper.
    Better for Hinglish code-switching scenarios.
    """
    
    def __init__(self, model_size: str = "medium") -> None:
        self.settings = get_settings()
        self.model_size = model_size
        self._model = None
    
    def _get_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                
                # Use CUDA if available, otherwise CPU
                self._model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16"
                )
            except Exception:
                # Fall back to CPU
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8"
                )
        return self._model
    
    async def transcribe(self, audio_data: bytes, language: Language | None = None) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        import soundfile as sf
        
        # Convert bytes to numpy array
        audio_io = io.BytesIO(audio_data)
        audio_array, sample_rate = sf.read(audio_io)
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(
                audio_array.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=16000
            )
        
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_array,
            language
        )
        
        return result
    
    def _transcribe_sync(self, audio_array: np.ndarray, language: Language | None) -> TranscriptionResult:
        """Synchronous transcription."""
        model = self._get_model()
        
        # Map language for Whisper
        whisper_lang = "hi" if language in [Language.HINDI, Language.HINGLISH] else None
        
        segments, info = model.transcribe(
            audio_array,
            language=whisper_lang,
            beam_size=5,
            best_of=5,
            task="transcribe"
        )
        
        # Collect segments
        text_parts = []
        words = []
        
        for segment in segments:
            text_parts.append(segment.text)
            if segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
        
        # Detect if code-mixed
        detected_lang = self._detect_language_from_text(" ".join(text_parts))
        
        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            language=detected_lang,
            confidence=info.language_probability if info.language_probability else 0.9,
            is_partial=False,
            words=words if words else None,
        )
    
    def _detect_language_from_text(self, text: str) -> Language:
        """Detect language from transcribed text."""
        from langdetect import detect_langs
        
        try:
            detections = detect_langs(text)
            
            # Check for Hindi-English mix
            has_hindi = any(d.lang == "hi" for d in detections)
            has_english = any(d.lang == "en" for d in detections)
            
            if has_hindi and has_english:
                return Language.HINGLISH
            elif has_hindi:
                return Language.HINDI
            elif has_english:
                return Language.ENGLISH_INDIA
            
            # Check for other languages
            primary = detections[0].lang if detections else "hi"
            lang_map = {
                "hi": Language.HINDI,
                "en": Language.ENGLISH_INDIA,
                "bn": Language.BENGALI,
                "ta": Language.TAMIL,
            }
            return lang_map.get(primary, Language.HINGLISH)
            
        except Exception:
            return Language.HINGLISH
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        language: Language | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Stream transcription using buffered Whisper.
        
        Whisper doesn't natively support streaming, so we buffer
        and transcribe in chunks.
        """
        buffer = io.BytesIO()
        buffer_duration_ms = 0
        min_chunk_duration_ms = 1000  # Transcribe every 1 second
        
        async for chunk in audio_stream:
            buffer.write(chunk.data)
            buffer_duration_ms += len(chunk.data) * 1000 // (chunk.sample_rate * 2)  # 16-bit audio
            
            if buffer_duration_ms >= min_chunk_duration_ms:
                # Transcribe buffer
                buffer.seek(0)
                audio_data = buffer.read()
                
                if len(audio_data) > 0:
                    result = await self.transcribe(audio_data, language)
                    result.is_partial = True
                    yield result
                
                # Reset buffer
                buffer = io.BytesIO()
                buffer_duration_ms = 0
        
        # Transcribe remaining audio
        buffer.seek(0)
        audio_data = buffer.read()
        if len(audio_data) > 0:
            result = await self.transcribe(audio_data, language)
            result.is_partial = False
            yield result
    
    async def detect_language(self, audio_data: bytes) -> Language:
        """Detect language from audio sample."""
        result = await self.transcribe(audio_data)
        return result.language


class TranscriberFactory:
    """Factory for creating transcriber instances."""
    
    @staticmethod
    def create(provider: str = "aws") -> BaseTranscriber:
        """Create a transcriber instance."""
        if provider == "aws":
            return AWSTranscriber()
        elif provider == "whisper":
            return WhisperTranscriber()
        else:
            raise ValueError(f"Unknown transcriber provider: {provider}")


# Voice Activity Detection helper
class VADProcessor:
    """Voice Activity Detection processor using WebRTC VAD."""
    
    def __init__(self, aggressiveness: int = 2) -> None:
        """
        Initialize VAD.
        
        Args:
            aggressiveness: 0-3, higher = more aggressive filtering
        """
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000
        self.frame_duration_ms = 30  # 10, 20, or 30 ms
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech."""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception:
            return True  # Assume speech on error
    
    def process_audio(self, audio_data: bytes) -> tuple[bytes, list[tuple[int, int]]]:
        """
        Process audio and return speech segments.
        
        Returns:
            Tuple of (speech_only_audio, list of (start_ms, end_ms) segments)
        """
        frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 16-bit
        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        
        speech_frames = []
        speech_segments = []
        in_speech = False
        segment_start = 0
        
        for i, frame in enumerate(frames):
            if len(frame) < frame_size:
                continue
            
            is_speech = self.is_speech(frame)
            
            if is_speech and not in_speech:
                in_speech = True
                segment_start = i * self.frame_duration_ms
            elif not is_speech and in_speech:
                in_speech = False
                speech_segments.append((segment_start, i * self.frame_duration_ms))
            
            if is_speech:
                speech_frames.append(frame)
        
        # Handle ongoing speech at end
        if in_speech:
            speech_segments.append((segment_start, len(frames) * self.frame_duration_ms))
        
        return b"".join(speech_frames), speech_segments
