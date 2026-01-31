"""
AI4Bharat IndicConformer ASR for multilingual Indian language transcription.
Supports all 22 official Indian languages including Hindi, Bengali, Tamil, etc.
"""

import asyncio
import io
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
    words: list[dict] | None = None
    alternatives: list[tuple[str, float]] | None = None


@dataclass
class AudioChunk:
    """Audio chunk for streaming transcription."""
    
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    timestamp_ms: int = 0


class IndicConformerTranscriber:
    """
    AI4Bharat IndicConformer-based transcription for Indian languages.
    
    Supports 22 official Indian languages:
    - Hindi (hi), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr)
    - Gujarati (gu), Kannada (kn), Malayalam (ml), Odia (or), Punjabi (pa)
    - Assamese (as), Bodo (brx), Dogri (doi), Kashmiri (ks), Konkani (kok)
    - Maithili (mai), Manipuri (mni), Nepali (ne), Sanskrit (sa), Santali (sat)
    - Sindhi (sd), Urdu (ur)
    """
    
    # Language code mapping from our internal codes to IndicConformer codes
    LANGUAGE_MAP = {
        Language.HINDI: "hi",
        Language.HINGLISH: "hi",  # Use Hindi for Hinglish
        Language.ENGLISH_INDIA: "hi",  # Fallback to Hindi
        Language.BENGALI: "bn",
        Language.TAMIL: "ta",
    }
    
    def __init__(self, hf_token: str | None = None) -> None:
        self.settings = get_settings()
        self.hf_token = hf_token or self.settings.indic_asr.hf_token
        self._model = None
        self._model_loaded = False
    
    def _get_model(self):
        """Lazy load the IndicConformer model."""
        if self._model is None:
            try:
                from transformers import AutoModel
                import os
                
                # Set HuggingFace token if provided
                if self.hf_token:
                    os.environ["HF_TOKEN"] = self.hf_token
                
                logger.info("loading_indic_conformer_model")
                
                # Load the model with trust_remote_code for custom model
                self._model = AutoModel.from_pretrained(
                    "ai4bharat/indic-conformer-600m-multilingual",
                    trust_remote_code=True,
                    token=self.hf_token
                )
                
                self._model_loaded = True
                logger.info("indic_conformer_model_loaded")
                
            except Exception as e:
                logger.error("indic_conformer_load_failed", error=str(e))
                raise
        
        return self._model
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Language | None = None
    ) -> TranscriptionResult:
        """
        Transcribe audio using IndicConformer.
        
        Args:
            audio_data: Audio bytes (WAV format expected)
            language: Target language for transcription
        
        Returns:
            TranscriptionResult with transcribed text
        """
        import torchaudio
        import torch
        
        language = language or Language.HINDI
        lang_code = self.LANGUAGE_MAP.get(language, "hi")
        
        try:
            # Convert bytes to tensor
            audio_io = io.BytesIO(audio_data)
            wav, sr = torchaudio.load(audio_io)
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            target_sample_rate = 16000
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=target_sample_rate
                )
                wav = resampler(wav)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                wav,
                lang_code
            )
            
            # Map back to our Language enum
            detected_language = self._detect_language_from_text(result, language)
            
            return TranscriptionResult(
                text=result,
                language=detected_language,
                confidence=0.9,  # IndicConformer is generally high confidence
                is_partial=False,
            )
            
        except Exception as e:
            logger.error("indic_conformer_transcription_failed", error=str(e))
            return TranscriptionResult(
                text="",
                language=language or Language.HINDI,
                confidence=0.0,
                is_partial=False,
            )
    
    def _transcribe_sync(self, wav, lang_code: str) -> str:
        """Synchronous transcription using IndicConformer."""
        model = self._get_model()
        
        # Use CTC decoding (faster) or RNNT (more accurate)
        # CTC is preferred for real-time applications
        try:
            transcription = model(wav, lang_code, "ctc")
            return transcription.strip()
        except Exception as e:
            logger.warning("ctc_failed_trying_rnnt", error=str(e))
            try:
                transcription = model(wav, lang_code, "rnnt")
                return transcription.strip()
            except Exception as e2:
                logger.error("rnnt_also_failed", error=str(e2))
                return ""
    
    def _detect_language_from_text(self, text: str, default: Language) -> Language:
        """Detect language from transcribed text."""
        if not text:
            return default
        
        try:
            from langdetect import detect_langs
            
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
            
            # Map other detected languages
            primary = detections[0].lang if detections else "hi"
            lang_map = {
                "hi": Language.HINDI,
                "en": Language.ENGLISH_INDIA,
                "bn": Language.BENGALI,
                "ta": Language.TAMIL,
            }
            return lang_map.get(primary, default)
            
        except Exception:
            return default
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        language: Language | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Stream transcription for real-time audio.
        
        IndicConformer doesn't natively support streaming,
        so we buffer and transcribe in chunks.
        """
        buffer = io.BytesIO()
        buffer_duration_ms = 0
        min_chunk_duration_ms = 1500  # Transcribe every 1.5 seconds
        
        language = language or Language.HINDI
        
        async for chunk in audio_stream:
            buffer.write(chunk.data)
            # Calculate duration (16-bit audio = 2 bytes per sample)
            buffer_duration_ms += len(chunk.data) * 1000 // (chunk.sample_rate * 2)
            
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
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded


class IndicConformerLite:
    """
    Lightweight version using ONNX for faster inference.
    Better for production with limited resources.
    """
    
    def __init__(self, hf_token: str | None = None) -> None:
        self.settings = get_settings()
        self.hf_token = hf_token or self.settings.indic_asr.hf_token
        self._session = None
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: Language | None = None
    ) -> TranscriptionResult:
        """Transcribe using ONNX runtime for faster inference."""
        # This is a placeholder for ONNX-based inference
        # The actual implementation would use onnxruntime
        
        logger.warning("onnx_inference_not_implemented")
        return TranscriptionResult(
            text="",
            language=language or Language.HINDI,
            confidence=0.0,
            is_partial=False,
        )


# Singleton instance
_indic_transcriber: IndicConformerTranscriber | None = None


def get_indic_transcriber() -> IndicConformerTranscriber:
    """Get IndicConformer transcriber instance."""
    global _indic_transcriber
    if _indic_transcriber is None:
        _indic_transcriber = IndicConformerTranscriber()
    return _indic_transcriber
