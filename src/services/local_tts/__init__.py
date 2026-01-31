"""Local TTS Service using gTTS as AWS Polly alternative."""
from src.services.local_tts.synthesizer import LocalTTSSynthesizer, get_local_tts

__all__ = ["LocalTTSSynthesizer", "get_local_tts"]
