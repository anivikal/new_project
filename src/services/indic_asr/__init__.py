"""AI4Bharat IndicConformer ASR Service for Indian Languages."""
from src.services.indic_asr.transcriber import IndicConformerTranscriber, get_indic_transcriber

__all__ = ["IndicConformerTranscriber", "get_indic_transcriber"]
