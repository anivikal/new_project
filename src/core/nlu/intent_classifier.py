"""
Intent Classification for Battery Smart Tier-1 queries.
Uses AWS Bedrock LLM for robust Hinglish intent detection.
"""

import json
import re
from abc import ABC, abstractmethod

import structlog

from src.config import get_settings
from src.models import Intent, IntentClassification, Language

logger = structlog.get_logger(__name__)


# Intent patterns for fallback classification
INTENT_PATTERNS = {
    Intent.SWAP_HISTORY: [
        r"swap\s*(history|record|list)",
        r"(kitne|कितने)\s*(swap|स्वैप)",
        r"(pichle|पिछले)\s*(swap|स्वैप)",
        r"(mera|मेरा)\s*(swap|स्वैप)\s*(record|history)",
        r"battery\s*(badli|change)\s*(history|record)",
    ],
    Intent.SWAP_INVOICE: [
        r"(invoice|bill|receipt)\s*(chahiye|dikhao|bhejo)",
        r"(इनवॉइस|बिल)\s*(चाहिए|दिखाओ|भेजो)",
        r"(swap|स्वैप)\s*(ka|की|का)\s*(bill|invoice|receipt)",
    ],
    Intent.INVOICE_EXPLANATION: [
        r"(invoice|bill)\s*(samjhao|explain|breakdown)",
        r"(charge|amount)\s*(kya|kyun|why)",
        r"(extra|zyada)\s*(paisa|charge|amount)",
        r"gst\s*(kya|kitna|charge)",
        r"(समझाओ|explain)\s*(invoice|bill)",
    ],
    Intent.NEAREST_STATION: [
        r"(nearest|paas|nazdeek|नज़दीक)\s*(station|hub)",
        r"(station|hub)\s*(kahan|where|kidhar)",
        r"(battery|बैटरी)\s*(kahan|कहाँ)\s*(milegi|मिलेगी)",
        r"swap\s*(kahan|कहाँ)",
    ],
    Intent.STATION_AVAILABILITY: [
        r"(battery|बैटरी)\s*(available|hai|है)",
        r"(station|hub)\s*(khula|open|band|closed)",
        r"(kitni|कितनी)\s*(battery|बैटरी)\s*(available|hai|है)",
        r"(slot|स्लॉट)\s*(available|hai|है)",
    ],
    Intent.SUBSCRIPTION_STATUS: [
        r"(subscription|plan)\s*(status|kya|hai)",
        r"(mera|मेरा)\s*(plan|subscription)",
        r"(plan|प्लान)\s*(active|चालू|expire)",
        r"(validity|वैलिडिटी)",
    ],
    Intent.SUBSCRIPTION_RENEWAL: [
        r"(renew|renewal|नवीनीकरण)",
        r"(plan|subscription)\s*(renew|extend|बढ़ाओ)",
        r"(dobara|फिर से)\s*(subscribe|लेना)",
    ],
    Intent.SUBSCRIPTION_PRICING: [
        r"(plan|subscription)\s*(price|cost|kitna)",
        r"(kitna|कितना)\s*(paisa|पैसा|rupee|रुपये)",
        r"(monthly|daily|weekly)\s*(charge|rate)",
        r"(pricing|rates|दाम)",
    ],
    Intent.PLAN_COMPARISON: [
        r"(plan|plans)\s*(compare|difference|कौनसा)",
        r"(best|सबसे अच्छा)\s*(plan|subscription)",
        r"(kon|कौन)\s*(sa|सा)\s*(plan|subscription)",
    ],
    Intent.LEAVE_INFORMATION: [
        r"(leave|छुट्टी)\s*(lena|chahiye|चाहिए)",
        r"(pause|रोकना)\s*(subscription|plan)",
        r"(temporary|कुछ दिन)\s*(band|बंद)",
        r"chutti\s*(mark|karna)",
    ],
    Intent.DSK_ACTIVATION: [
        r"(dsk|DSK)\s*(activation|activate|चालू)",
        r"(new|naya)\s*(dsk|DSK)",
        r"(account|खाता)\s*(activate|चालू)",
        r"(registration|register)",
    ],
    Intent.HUMAN_AGENT: [
        r"(agent|executive|insaan|इंसान)\s*(se|से)\s*(baat|बात)",
        r"(human|person)\s*(connect|transfer)",
        r"(customer\s*care|support)\s*(se|से)",
        r"(kisi|किसी)\s*(se|से)\s*(baat|बात)",
        r"(manager|supervisor)",
    ],
    Intent.GREETING: [
        r"^(hello|hi|hey|namaste|namaskar|नमस्ते|नमस्कार)$",
        r"^(good\s*(morning|afternoon|evening))$",
    ],
    Intent.GOODBYE: [
        r"^(bye|goodbye|alvida|धन्यवाद|thanks|thank you)$",
        r"(call\s*end|disconnect)",
    ],
    Intent.HELP: [
        r"^(help|madad|मदद|sahayata)$",
        r"(kya\s*kar\s*sakte|क्या कर सकते)",
        r"(options|menu)",
    ],
}


class BaseIntentClassifier(ABC):
    """Abstract base class for intent classification."""
    
    @abstractmethod
    async def classify(self, text: str, language: Language, context: dict | None = None) -> IntentClassification:
        """Classify intent from text."""
        pass


class PatternIntentClassifier(BaseIntentClassifier):
    """
    Pattern-based intent classifier for fast, local classification.
    Used as fallback or for simple cases.
    """
    
    def __init__(self) -> None:
        self.patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }
    
    async def classify(self, text: str, language: Language, context: dict | None = None) -> IntentClassification:
        """Classify intent using regex patterns."""
        text_lower = text.lower().strip()
        
        matches: list[tuple[Intent, int]] = []
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    matches.append((intent, len(pattern.pattern)))
                    break
        
        if matches:
            # Sort by pattern length (longer = more specific)
            matches.sort(key=lambda x: x[1], reverse=True)
            best_intent = matches[0][0]
            
            # Calculate confidence based on match quality
            confidence = min(0.85, 0.6 + (matches[0][1] / 100))
            
            alternatives = [
                (m[0], 0.5) for m in matches[1:3]
            ] if len(matches) > 1 else []
            
            return IntentClassification(
                intent=best_intent,
                confidence=confidence,
                alternative_intents=alternatives
            )
        
        # No pattern matched
        return IntentClassification(
            intent=Intent.UNKNOWN,
            confidence=0.3,
            alternative_intents=[]
        )


class BedrockIntentClassifier(BaseIntentClassifier):
    """
    AWS Bedrock-based intent classifier using Claude.
    Provides robust Hinglish understanding.
    """
    
    CLASSIFICATION_PROMPT = """You are an intent classifier for Battery Smart's driver support system.
Battery Smart operates India's largest battery-swapping network for electric vehicles.

Classify the following driver query into one of these intents:
- swap_history: Asking about past battery swaps, swap records, history
- swap_invoice: Requesting invoice/bill for swaps
- invoice_explanation: Asking to explain charges, GST, extra amounts on invoice
- nearest_station: Finding nearest battery swap station
- station_availability: Checking if battery is available at station
- subscription_status: Checking current subscription/plan status
- subscription_renewal: Wanting to renew subscription
- subscription_pricing: Asking about plan prices/costs
- plan_comparison: Comparing different plans
- leave_information: Taking leave/pause from subscription
- dsk_activation: Activating new DSK/account
- human_agent: Explicitly requesting human support
- greeting: Simple greeting
- goodbye: Ending conversation
- help: Asking what the bot can do
- out_of_scope: Query not related to Battery Smart services
- unknown: Cannot determine intent

The query may be in Hindi, English, or Hinglish (mixed Hindi-English).

Query: "{query}"

Respond in JSON format:
{{
    "intent": "<intent_name>",
    "confidence": <0.0-1.0>,
    "sub_intent": "<optional sub-intent>",
    "reasoning": "<brief explanation>",
    "alternative_intents": [["<intent>", <confidence>]]
}}"""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
    
    async def _get_client(self):
        """Get or create Bedrock client."""
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            self._client = await session.client(
                "bedrock-runtime",
                region_name=self.settings.aws.region
            ).__aenter__()
        return self._client
    
    async def classify(self, text: str, language: Language, context: dict | None = None) -> IntentClassification:
        """Classify intent using Bedrock Claude."""
        try:
            client = await self._get_client()
            
            prompt = self.CLASSIFICATION_PROMPT.format(query=text)
            
            # Add context if available
            if context:
                context_str = f"\nConversation context: {json.dumps(context)}"
                prompt = prompt.replace('Query:', f'{context_str}\n\nQuery:')
            
            response = await client.invoke_model(
                modelId=self.settings.aws.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            response_body = json.loads(response["body"].read())
            result_text = response_body["content"][0]["text"]
            
            # Parse JSON from response
            try:
                # Find JSON in response
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            except json.JSONDecodeError:
                logger.warning("failed_to_parse_intent_json", response=result_text)
                # Fall back to pattern classifier
                fallback = PatternIntentClassifier()
                return await fallback.classify(text, language, context)
            
            # Map to Intent enum
            intent_str = result.get("intent", "unknown")
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN
            
            # Parse alternatives
            alternatives = []
            for alt in result.get("alternative_intents", []):
                if isinstance(alt, list) and len(alt) == 2:
                    try:
                        alt_intent = Intent(alt[0])
                        alternatives.append((alt_intent, float(alt[1])))
                    except (ValueError, TypeError):
                        pass
            
            return IntentClassification(
                intent=intent,
                confidence=float(result.get("confidence", 0.7)),
                sub_intent=result.get("sub_intent"),
                alternative_intents=alternatives
            )
            
        except Exception as e:
            logger.error("bedrock_classification_failed", error=str(e))
            # Fall back to pattern classifier
            fallback = PatternIntentClassifier()
            return await fallback.classify(text, language, context)


class HybridIntentClassifier(BaseIntentClassifier):
    """
    Hybrid classifier combining pattern matching and LLM.
    Uses patterns for high-confidence matches, LLM for ambiguous cases.
    """
    
    def __init__(self) -> None:
        self.pattern_classifier = PatternIntentClassifier()
        self.llm_classifier = BedrockIntentClassifier()
        self.settings = get_settings()
    
    async def classify(self, text: str, language: Language, context: dict | None = None) -> IntentClassification:
        """
        Classify using hybrid approach:
        1. Try pattern matching first
        2. If confidence >= threshold, use pattern result
        3. Otherwise, use LLM classifier
        """
        # First try pattern matching (fast)
        pattern_result = await self.pattern_classifier.classify(text, language, context)
        
        # If high confidence pattern match, use it
        if pattern_result.confidence >= 0.8 and pattern_result.intent != Intent.UNKNOWN:
            logger.debug(
                "using_pattern_classification",
                intent=pattern_result.intent.value,
                confidence=pattern_result.confidence
            )
            return pattern_result
        
        # Otherwise use LLM for better understanding
        logger.debug("using_llm_classification", pattern_confidence=pattern_result.confidence)
        llm_result = await self.llm_classifier.classify(text, language, context)
        
        # If LLM also uncertain, consider pattern alternative
        if llm_result.confidence < 0.5 and pattern_result.confidence > llm_result.confidence:
            return pattern_result
        
        return llm_result


class IntentClassifierFactory:
    """Factory for creating intent classifier instances."""
    
    @staticmethod
    def create(classifier_type: str = "hybrid") -> BaseIntentClassifier:
        """Create an intent classifier."""
        classifiers = {
            "pattern": PatternIntentClassifier,
            "bedrock": BedrockIntentClassifier,
            "hybrid": HybridIntentClassifier,
        }
        
        classifier_class = classifiers.get(classifier_type)
        if not classifier_class:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        return classifier_class()
