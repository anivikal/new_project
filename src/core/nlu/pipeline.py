"""
NLU Pipeline that combines intent classification, entity extraction, and sentiment analysis.
"""

import re
from dataclasses import dataclass

import structlog

from src.config import get_settings
from src.core.nlu.entity_extractor import EntityExtractorFactory, HybridEntityExtractor
from src.core.nlu.intent_classifier import HybridIntentClassifier, IntentClassifierFactory
from src.core.nlu.sentiment_analyzer import HybridSentimentAnalyzer, SentimentAnalyzerFactory
from src.models import Language, NLUResult

logger = structlog.get_logger(__name__)


@dataclass
class NLUConfig:
    """Configuration for NLU pipeline."""
    
    intent_classifier_type: str = "hybrid"
    entity_extractor_type: str = "hybrid"
    sentiment_analyzer_type: str = "hybrid"
    
    # Text preprocessing
    normalize_text: bool = True
    transliterate_hindi: bool = False  # Convert Devanagari to Roman


class TextNormalizer:
    """Text normalization and preprocessing for Hinglish."""
    
    # Common Hinglish abbreviations and corrections
    CORRECTIONS = {
        "kya": "kya",
        "hai": "hai",
        "nhi": "nahi",
        "nai": "nahi",
        "h": "hai",
        "kr": "kar",
        "krna": "karna",
        "krte": "karte",
        "m": "main",
        "me": "main",
        "k": "ka",
        "ki": "ki",
        "ke": "ke",
        "toh": "to",
        "bhi": "bhi",
        "abhi": "abhi",
        "wo": "wo",
        "ye": "ye",
        "kb": "kab",
        "kaise": "kaise",
        "kyu": "kyun",
        "q": "kyun",
        "thx": "thanks",
        "pls": "please",
        "plz": "please",
    }
    
    # Number words to digits
    NUMBER_WORDS = {
        "ek": "1", "do": "2", "teen": "3", "char": "4", "paanch": "5",
        "chhe": "6", "saat": "7", "aath": "8", "nau": "9", "das": "10",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    }
    
    def normalize(self, text: str) -> str:
        """Normalize text for better NLU processing."""
        # Lowercase
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Expand common abbreviations
        words = normalized.split()
        expanded_words = [self.CORRECTIONS.get(w, w) for w in words]
        normalized = ' '.join(expanded_words)
        
        # Normalize repeated characters (e.g., "pleaseee" -> "please")
        normalized = re.sub(r'(.)\1{2,}', r'\1\1', normalized)
        
        # Normalize punctuation
        normalized = re.sub(r'[!]{2,}', '!', normalized)
        normalized = re.sub(r'[?]{2,}', '?', normalized)
        
        return normalized
    
    def detect_code_mixing(self, text: str) -> bool:
        """Detect if text has Hindi-English code mixing."""
        # Check for Devanagari characters
        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
        
        # Check for Latin characters (excluding punctuation)
        has_latin = any(c.isalpha() and ord(c) < 128 for c in text)
        
        # Check for common Hindi words in Roman script
        hindi_roman_words = [
            "kya", "hai", "nahi", "haan", "theek", "accha",
            "kaise", "kyun", "kab", "kahan", "kaun",
            "mera", "tumhara", "uska", "hamara",
            "chahiye", "hoga", "tha", "thi",
        ]
        has_hindi_roman = any(word in text.lower() for word in hindi_roman_words)
        
        return (has_devanagari and has_latin) or (has_latin and has_hindi_roman)


class LanguageDetector:
    """Detect language from text."""
    
    def detect(self, text: str) -> Language:
        """Detect primary language of text."""
        # Check for Devanagari script
        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        
        total_chars = devanagari_chars + latin_chars
        if total_chars == 0:
            return Language.HINGLISH
        
        devanagari_ratio = devanagari_chars / total_chars
        
        # Use langdetect for more accurate detection
        try:
            from langdetect import detect_langs
            
            detections = detect_langs(text)
            
            # Check for code mixing
            has_hindi = any(d.lang == "hi" for d in detections)
            has_english = any(d.lang == "en" for d in detections)
            
            if has_hindi and has_english:
                return Language.HINGLISH
            elif has_hindi or devanagari_ratio > 0.5:
                return Language.HINDI
            elif has_english:
                return Language.ENGLISH_INDIA
            
            # Check for other Indian languages
            primary = detections[0].lang if detections else "hi"
            lang_map = {
                "hi": Language.HINDI,
                "en": Language.ENGLISH_INDIA,
                "bn": Language.BENGALI,
                "ta": Language.TAMIL,
            }
            return lang_map.get(primary, Language.HINGLISH)
            
        except Exception:
            # Fallback based on character ratio
            if devanagari_ratio > 0.8:
                return Language.HINDI
            elif devanagari_ratio > 0.2:
                return Language.HINGLISH
            else:
                return Language.ENGLISH_INDIA


class NLUPipeline:
    """
    Main NLU pipeline combining all components.
    
    Flow:
    1. Text normalization
    2. Language detection
    3. Intent classification
    4. Entity extraction
    5. Sentiment analysis
    6. Result aggregation
    """
    
    def __init__(self, config: NLUConfig | None = None) -> None:
        self.config = config or NLUConfig()
        self.settings = get_settings()
        
        # Initialize components
        self.normalizer = TextNormalizer()
        self.language_detector = LanguageDetector()
        self.intent_classifier = IntentClassifierFactory.create(self.config.intent_classifier_type)
        self.entity_extractor = EntityExtractorFactory.create(self.config.entity_extractor_type)
        self.sentiment_analyzer = SentimentAnalyzerFactory.create(self.config.sentiment_analyzer_type)
    
    async def process(
        self,
        text: str,
        context: dict | None = None,
        previous_turns: list[str] | None = None
    ) -> NLUResult:
        """
        Process text through the NLU pipeline.
        
        Args:
            text: User input text
            context: Conversation context (filled slots, current intent, etc.)
            previous_turns: Previous user utterances for sentiment context
        
        Returns:
            Complete NLU result
        """
        logger.info("nlu_processing_start", text_length=len(text))
        
        # 1. Normalize text
        original_text = text
        if self.config.normalize_text:
            normalized_text = self.normalizer.normalize(text)
        else:
            normalized_text = text
        
        # 2. Detect language
        language = self.language_detector.detect(normalized_text)
        is_code_mixed = self.normalizer.detect_code_mixing(original_text)
        
        # 3. Run classification, extraction, and analysis in parallel
        import asyncio
        
        intent_task = self.intent_classifier.classify(normalized_text, language, context)
        entities_task = self.entity_extractor.extract(normalized_text, language)
        sentiment_task = self.sentiment_analyzer.analyze(normalized_text, language, previous_turns)
        
        intent, entities, sentiment = await asyncio.gather(
            intent_task,
            entities_task,
            sentiment_task
        )
        
        logger.info(
            "nlu_processing_complete",
            intent=intent.intent.value,
            confidence=intent.confidence,
            entities_count=len(entities),
            sentiment=sentiment.label.value,
            language=language.value
        )
        
        return NLUResult(
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            detected_language=language,
            original_text=original_text,
            normalized_text=normalized_text,
            is_code_mixed=is_code_mixed
        )
    
    async def process_with_history(
        self,
        text: str,
        conversation_history: list[dict]
    ) -> NLUResult:
        """
        Process text with full conversation history for better context.
        
        Args:
            text: Current user input
            conversation_history: List of previous turns with 'role' and 'content'
        
        Returns:
            NLU result
        """
        # Extract context from history
        context = {}
        previous_turns = []
        
        for turn in conversation_history:
            if turn.get("role") == "user":
                previous_turns.append(turn.get("content", ""))
            
            # Extract slot values from history
            if turn.get("filled_slots"):
                context.update(turn["filled_slots"])
            
            # Track current intent
            if turn.get("intent"):
                context["previous_intent"] = turn["intent"]
        
        return await self.process(text, context, previous_turns[-5:])  # Last 5 turns


# Singleton instance for efficiency
_pipeline_instance: NLUPipeline | None = None


def get_nlu_pipeline(config: NLUConfig | None = None) -> NLUPipeline:
    """Get or create NLU pipeline instance."""
    global _pipeline_instance
    
    if _pipeline_instance is None or config is not None:
        _pipeline_instance = NLUPipeline(config)
    
    return _pipeline_instance
