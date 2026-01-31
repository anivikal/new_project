"""
Sentiment Analysis for Battery Smart driver conversations.
Detects negative sentiment, frustration, and confusion for handoff triggers.
"""

import re
from abc import ABC, abstractmethod

import structlog

from src.config import get_settings
from src.models import Language, Sentiment, SentimentLabel

logger = structlog.get_logger(__name__)


# Sentiment indicators for Hinglish
NEGATIVE_INDICATORS = [
    # Hindi negative expressions
    r"(गुस्सा|gussa|angry|frustrated)",
    r"(बेकार|bekar|useless|waste)",
    r"(परेशान|pareshan|pareshaan|troubled)",
    r"(problem|dikkat|दिक्कत|समस्या)",
    r"(galat|गलत|wrong|incorrect)",
    r"(nahi\s*samajh|नहीं\s*समझ|not\s*understand)",
    r"(kya\s*bakwas|क्या\s*बकवास|nonsense)",
    r"(time\s*waste|टाइम\s*वेस्ट)",
    r"(cheat|धोखा|fraud)",
    r"(loot|लूट|overcharge)",
    r"(complaint|शिकायत)",
    r"(worst|सबसे\s*खराब)",
    r"(terrible|horrible|awful)",
    r"(kab\s*tak|कब\s*तक|how\s*long)",
    r"(itna\s*time|इतना\s*टाइम|so\s*much\s*time)",
    r"(haar\s*gaya|हार\s*गया|give\s*up)",
    r"(bore|बोर)",
    r"(pagal|पागल|crazy)",
]

FRUSTRATION_INDICATORS = [
    # Repeated questions/requests
    r"(phir\s*se|फिर\s*से|again)",
    r"(kitni\s*baar|कितनी\s*बार|how\s*many\s*times)",
    r"(bol\s*raha|बोल\s*रहा|already\s*told)",
    r"(sun\s*nahi|सुन\s*नहीं|not\s*listening)",
    r"(kuch\s*nahi\s*hota|कुछ\s*नहीं\s*होता|nothing\s*happens)",
    r"(same\s*(question|problem|issue))",
    r"(still\s*not|abhi\s*bhi\s*nahi|अभी\s*भी\s*नहीं)",
    # Exclamations
    r"(arre|अरे|hey|oh)",
    r"(yaar|यार|bhai|भाई)",
    r"(!{2,})",  # Multiple exclamation marks
    r"(\?{2,})",  # Multiple question marks
    # Impatience
    r"(jaldi|जल्दी|quickly|fast)",
    r"(urgent|जरूरी|emergency)",
]

CONFUSION_INDICATORS = [
    r"(samajh\s*nahi|समझ\s*नहीं|don't\s*understand)",
    r"(kya\s*matlab|क्या\s*मतलब|what\s*do\s*you\s*mean)",
    r"(confuse|confused|कंफ्यूज)",
    r"(clear\s*nahi|क्लियर\s*नहीं|not\s*clear)",
    r"(kaise|कैसे|how\s*to)",
    r"(kyun|क्यों|why)",
    r"(explain|समझाओ)",
    r"(what\s*is|kya\s*hai|क्या\s*है)",
]

POSITIVE_INDICATORS = [
    r"(thank|thanks|धन्यवाद|shukriya|शुक्रिया)",
    r"(good|अच्छा|accha|badhiya|बढ़िया)",
    r"(great|wonderful|amazing)",
    r"(helpful|मददगार)",
    r"(solve|solved|हल)",
    r"(perfect|सही)",
    r"(happy|खुश|khush)",
    r"(samajh\s*gaya|समझ\s*गया|understood)",
    r"(ho\s*gaya|हो\s*गया|done)",
    r"(theek|ठीक|okay|ok)",
]


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analysis."""
    
    @abstractmethod
    async def analyze(self, text: str, language: Language, context: list[str] | None = None) -> Sentiment:
        """Analyze sentiment of text."""
        pass


class RuleSentimentAnalyzer(BaseSentimentAnalyzer):
    """Rule-based sentiment analyzer for fast, explainable results."""
    
    def __init__(self) -> None:
        self.negative_patterns = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_INDICATORS]
        self.frustration_patterns = [re.compile(p, re.IGNORECASE) for p in FRUSTRATION_INDICATORS]
        self.confusion_patterns = [re.compile(p, re.IGNORECASE) for p in CONFUSION_INDICATORS]
        self.positive_patterns = [re.compile(p, re.IGNORECASE) for p in POSITIVE_INDICATORS]
    
    async def analyze(self, text: str, language: Language, context: list[str] | None = None) -> Sentiment:
        """Analyze sentiment using rule-based patterns."""
        indicators = []
        
        # Count matches
        negative_count = sum(1 for p in self.negative_patterns if p.search(text))
        frustration_count = sum(1 for p in self.frustration_patterns if p.search(text))
        confusion_count = sum(1 for p in self.confusion_patterns if p.search(text))
        positive_count = sum(1 for p in self.positive_patterns if p.search(text))
        
        # Collect matched indicators
        for p in self.negative_patterns:
            match = p.search(text)
            if match:
                indicators.append(f"negative:{match.group()}")
        
        for p in self.frustration_patterns:
            match = p.search(text)
            if match:
                indicators.append(f"frustration:{match.group()}")
        
        for p in self.confusion_patterns:
            match = p.search(text)
            if match:
                indicators.append(f"confusion:{match.group()}")
        
        for p in self.positive_patterns:
            match = p.search(text)
            if match:
                indicators.append(f"positive:{match.group()}")
        
        # Determine sentiment label and score
        total_negative = negative_count + frustration_count
        
        if frustration_count >= 2 or (negative_count >= 2 and frustration_count >= 1):
            label = SentimentLabel.FRUSTRATED
            score = max(0.1, 0.4 - (frustration_count * 0.1) - (negative_count * 0.05))
        elif confusion_count >= 2:
            label = SentimentLabel.CONFUSED
            score = 0.4
        elif total_negative > positive_count and total_negative >= 1:
            label = SentimentLabel.NEGATIVE
            score = max(0.2, 0.5 - (total_negative * 0.1))
        elif positive_count > total_negative and positive_count >= 1:
            label = SentimentLabel.POSITIVE
            score = min(0.9, 0.6 + (positive_count * 0.1))
        else:
            label = SentimentLabel.NEUTRAL
            score = 0.5
        
        # Adjust based on context (previous turns)
        if context:
            context_negative = sum(
                1 for turn in context[-3:]  # Last 3 turns
                for p in self.negative_patterns + self.frustration_patterns
                if p.search(turn)
            )
            if context_negative >= 2:
                score = max(0.15, score - 0.15)
                if label == SentimentLabel.NEUTRAL:
                    label = SentimentLabel.NEGATIVE
        
        # Calculate confidence
        total_indicators = len(indicators)
        confidence = min(0.95, 0.6 + (total_indicators * 0.1))
        
        return Sentiment(
            label=label,
            score=score,
            confidence=confidence,
            indicators=indicators[:5]  # Limit to top 5
        )


class BedrockSentimentAnalyzer(BaseSentimentAnalyzer):
    """AWS Bedrock-based sentiment analysis for nuanced understanding."""
    
    ANALYSIS_PROMPT = """Analyze the sentiment of this Battery Smart driver support query.
The driver is contacting support about battery swapping services.
The text may be in Hindi, English, or Hinglish (mixed).

Consider:
1. Explicit emotional words
2. Tone and urgency
3. Signs of frustration (repetition, impatience, complaints)
4. Signs of confusion (questions, uncertainty)
5. Politeness level

Query: "{query}"

{context_section}

Classify as one of: positive, neutral, negative, frustrated, confused

Respond in JSON:
{{
    "label": "<sentiment_label>",
    "score": <0.0-1.0 where 0=very negative, 1=very positive>,
    "confidence": <0.0-1.0>,
    "indicators": ["<detected_indicator_1>", "<detected_indicator_2>"],
    "reasoning": "<brief explanation>"
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
    
    async def analyze(self, text: str, language: Language, context: list[str] | None = None) -> Sentiment:
        """Analyze sentiment using Bedrock LLM."""
        import json
        
        try:
            client = await self._get_client()
            
            # Build context section
            context_section = ""
            if context:
                context_section = "Previous conversation turns:\n" + "\n".join(
                    f"- {turn}" for turn in context[-3:]
                )
            
            prompt = self.ANALYSIS_PROMPT.format(
                query=text,
                context_section=context_section
            )
            
            response = await client.invoke_model(
                modelId=self.settings.aws.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            
            response_body = json.loads(response["body"].read())
            result_text = response_body["content"][0]["text"]
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON in response")
            
            # Map label
            label_map = {
                "positive": SentimentLabel.POSITIVE,
                "neutral": SentimentLabel.NEUTRAL,
                "negative": SentimentLabel.NEGATIVE,
                "frustrated": SentimentLabel.FRUSTRATED,
                "confused": SentimentLabel.CONFUSED,
            }
            
            label = label_map.get(result.get("label", "neutral").lower(), SentimentLabel.NEUTRAL)
            
            return Sentiment(
                label=label,
                score=float(result.get("score", 0.5)),
                confidence=float(result.get("confidence", 0.8)),
                indicators=result.get("indicators", [])
            )
            
        except Exception as e:
            logger.error("bedrock_sentiment_analysis_failed", error=str(e))
            # Fall back to rule-based
            fallback = RuleSentimentAnalyzer()
            return await fallback.analyze(text, language, context)


class HybridSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Combines rule-based and LLM sentiment analysis.
    Rules for speed, LLM for nuanced cases.
    """
    
    def __init__(self) -> None:
        self.rule_analyzer = RuleSentimentAnalyzer()
        self.llm_analyzer = BedrockSentimentAnalyzer()
        self.settings = get_settings()
    
    async def analyze(self, text: str, language: Language, context: list[str] | None = None) -> Sentiment:
        """Analyze sentiment using hybrid approach."""
        # Always run rule-based first (fast)
        rule_result = await self.rule_analyzer.analyze(text, language, context)
        
        # For borderline cases or frustration, use LLM for confirmation
        needs_llm = (
            rule_result.label in [SentimentLabel.FRUSTRATED, SentimentLabel.NEGATIVE] or
            rule_result.confidence < 0.7 or
            0.35 <= rule_result.score <= 0.65  # Borderline scores
        )
        
        if needs_llm:
            llm_result = await self.llm_analyzer.analyze(text, language, context)
            
            # Combine results
            # If both agree on negative/frustrated, high confidence
            if (rule_result.label in [SentimentLabel.FRUSTRATED, SentimentLabel.NEGATIVE] and
                llm_result.label in [SentimentLabel.FRUSTRATED, SentimentLabel.NEGATIVE]):
                return Sentiment(
                    label=SentimentLabel.FRUSTRATED if SentimentLabel.FRUSTRATED in [rule_result.label, llm_result.label] else SentimentLabel.NEGATIVE,
                    score=min(rule_result.score, llm_result.score),
                    confidence=max(rule_result.confidence, llm_result.confidence),
                    indicators=list(set(rule_result.indicators + llm_result.indicators))[:5]
                )
            
            # Otherwise, prefer LLM result
            return llm_result
        
        return rule_result


class SentimentAnalyzerFactory:
    """Factory for creating sentiment analyzer instances."""
    
    @staticmethod
    def create(analyzer_type: str = "hybrid") -> BaseSentimentAnalyzer:
        """Create a sentiment analyzer."""
        analyzers = {
            "rule": RuleSentimentAnalyzer,
            "bedrock": BedrockSentimentAnalyzer,
            "hybrid": HybridSentimentAnalyzer,
        }
        
        analyzer_class = analyzers.get(analyzer_type)
        if not analyzer_class:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
        
        return analyzer_class()
