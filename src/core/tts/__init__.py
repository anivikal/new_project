"""Text-to-Speech (TTS) module."""

from src.core.tts.synthesizer import (
    AWSPollySynthesizer,
    BaseSynthesizer,
    HinglishTextProcessor,
    SSMLBuilder,
    SynthesisResult,
    SynthesizerFactory,
    TTSCache,
    presynthesize_common_phrases,
)

__all__ = [
    "AWSPollySynthesizer",
    "BaseSynthesizer",
    "HinglishTextProcessor",
    "SSMLBuilder",
    "SynthesisResult",
    "SynthesizerFactory",
    "TTSCache",
    "presynthesize_common_phrases",
]
