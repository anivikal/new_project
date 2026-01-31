"""
Entity Extraction for Battery Smart driver queries.
Extracts structured data like phone numbers, dates, locations, amounts.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import structlog

from src.config import get_settings
from src.models import Entity, EntityType, Language

logger = structlog.get_logger(__name__)


# Regex patterns for entity extraction
ENTITY_PATTERNS = {
    EntityType.PHONE_NUMBER: [
        r"(\+91[\s-]?)?([6-9]\d{9})",  # Indian mobile number
        r"(\d{4}[\s-]?\d{3}[\s-]?\d{3})",  # Formatted
    ],
    EntityType.AMOUNT: [
        r"(?:₹|rs\.?|rupees?|रुपये?)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
        r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:₹|rs\.?|rupees?|रुपये?)",
        r"(\d+)\s*(?:rupees?|रुपये?)",
    ],
    EntityType.DATE: [
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",  # DD/MM/YYYY
        r"(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
        r"(आज|today|कल|tomorrow|परसों|day after)",
        r"(पिछले?\s*(?:हफ्ते?|week|महीने?|month))",
        r"(last\s*(?:week|month|day))",
    ],
    EntityType.INVOICE_NUMBER: [
        r"(?:invoice|bill|inv)\s*(?:#|no\.?|number)?\s*([A-Z0-9-]+)",
        r"([A-Z]{2,3}[-/]\d{6,})",  # Format like INV-123456
    ],
    EntityType.STATION_NAME: [
        r"(?:station|hub|center)\s+(?:at\s+)?([A-Za-z\s]+(?:road|nagar|colony|market|chowk))",
        r"([A-Za-z]+)\s+(?:station|hub|swap\s*point)",
    ],
    EntityType.SUBSCRIPTION_PLAN: [
        r"(daily|weekly|monthly|yearly)\s*(?:plan|subscription)?",
        r"(basic|standard|premium|pro)\s*(?:plan)?",
        r"(₹?\d+)\s*(?:wala|वाला)\s*(?:plan|subscription)",
    ],
    EntityType.BATTERY_ID: [
        r"(?:battery|batt?)\s*(?:id|#|no\.?)?\s*([A-Z0-9]{6,})",
        r"([A-Z]{2}\d{6,})",  # Format like BS123456
    ],
    EntityType.DRIVER_ID: [
        r"(?:driver|rider)\s*(?:id|#|no\.?)?\s*([A-Z0-9-]+)",
        r"(?:id|ID)\s*(?:is|:)?\s*([A-Z0-9-]+)",
    ],
}

# Hindi number words to digits
HINDI_NUMBERS = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "पच्चीस": 25, "तीस": 30, "पैंतीस": 35, "चालीस": 40, "पचास": 50,
    "सौ": 100, "हज़ार": 1000,
}

# Relative date keywords
RELATIVE_DATES = {
    "आज": 0, "today": 0,
    "कल": 1, "tomorrow": 1, "kal": 1,
    "परसों": 2, "parson": 2, "day after": 2,
    "पिछला": -1, "पिछले": -1, "last": -1,
}


class BaseEntityExtractor(ABC):
    """Abstract base class for entity extraction."""
    
    @abstractmethod
    async def extract(self, text: str, language: Language) -> list[Entity]:
        """Extract entities from text."""
        pass


class PatternEntityExtractor(BaseEntityExtractor):
    """Pattern-based entity extractor using regex."""
    
    def __init__(self) -> None:
        self.patterns = {
            entity_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for entity_type, patterns in ENTITY_PATTERNS.items()
        }
    
    async def extract(self, text: str, language: Language) -> list[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get the captured group or full match
                    value = match.group(1) if match.lastindex else match.group()
                    
                    # Normalize the value
                    normalized = self._normalize_value(entity_type, value, text)
                    
                    entity = Entity(
                        type=entity_type,
                        value=value.strip(),
                        confidence=0.85,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        normalized_value=normalized
                    )
                    
                    # Avoid duplicates
                    if not any(e.type == entity.type and e.value == entity.value for e in entities):
                        entities.append(entity)
        
        # Also extract special entities
        entities.extend(await self._extract_dates(text))
        entities.extend(await self._extract_locations(text))
        
        return entities
    
    def _normalize_value(self, entity_type: EntityType, value: str, full_text: str) -> any:
        """Normalize extracted value to standard format."""
        if entity_type == EntityType.PHONE_NUMBER:
            # Remove spaces and dashes, ensure 10 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) == 10:
                return digits
            elif len(digits) == 12 and digits.startswith('91'):
                return digits[2:]
            return digits[-10:] if len(digits) >= 10 else digits
        
        elif entity_type == EntityType.AMOUNT:
            # Parse amount to float
            clean = re.sub(r'[^\d.]', '', value.replace(',', ''))
            try:
                return float(clean)
            except ValueError:
                return None
        
        elif entity_type == EntityType.DATE:
            return self._parse_date(value)
        
        return value
    
    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date string to datetime."""
        date_str_lower = date_str.lower().strip()
        
        # Check relative dates
        for keyword, offset in RELATIVE_DATES.items():
            if keyword in date_str_lower:
                return datetime.now() + timedelta(days=offset)
        
        # Check for week/month references
        if any(w in date_str_lower for w in ["week", "हफ्ते", "हफ्ता"]):
            if any(w in date_str_lower for w in ["last", "पिछले", "पिछला"]):
                return datetime.now() - timedelta(weeks=1)
        
        if any(w in date_str_lower for w in ["month", "महीने", "महीना"]):
            if any(w in date_str_lower for w in ["last", "पिछले", "पिछला"]):
                return datetime.now() - timedelta(days=30)
        
        # Try parsing standard formats
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
            "%d %b %Y", "%d %B %Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    async def _extract_dates(self, text: str) -> list[Entity]:
        """Extract date entities with enhanced Hindi support."""
        entities = []
        
        # Check for relative date keywords
        for hindi_word, offset in RELATIVE_DATES.items():
            if hindi_word in text.lower():
                date_value = datetime.now() + timedelta(days=offset)
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=hindi_word,
                    confidence=0.9,
                    normalized_value=date_value
                ))
        
        return entities
    
    async def _extract_locations(self, text: str) -> list[Entity]:
        """Extract location entities."""
        entities = []
        
        # Common location indicators
        location_patterns = [
            r"(?:near|paas|nazdeek|नज़दीक)\s+([A-Za-z\s]+?)(?:\s|$|,|\.)",
            r"(?:in|at|pe|में|पर)\s+([A-Za-z\s]+?)(?:\s+(?:area|locality|station|hub))",
        ]
        
        for pattern in location_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                location = match.group(1).strip()
                if len(location) > 2:  # Avoid single letters
                    entities.append(Entity(
                        type=EntityType.LOCATION,
                        value=location,
                        confidence=0.7,
                        start_pos=match.start(1),
                        end_pos=match.end(1),
                    ))
        
        return entities


class BedrockEntityExtractor(BaseEntityExtractor):
    """AWS Bedrock-based entity extraction for complex cases."""
    
    EXTRACTION_PROMPT = """Extract entities from this Battery Smart driver support query.
The query may be in Hindi, English, or Hinglish.

Extract these entity types if present:
- phone_number: Indian mobile numbers (10 digits)
- driver_id: Driver/rider ID
- date: Any date references (including relative like "yesterday", "last week")
- date_range: Date ranges
- location: Places, station names, areas
- station_name: Battery Smart station names
- subscription_plan: Plan types (daily, weekly, monthly, premium, etc.)
- amount: Money amounts in rupees
- invoice_number: Invoice/bill numbers
- battery_id: Battery identification numbers

Query: "{query}"

Respond in JSON:
{{
    "entities": [
        {{
            "type": "<entity_type>",
            "value": "<extracted_value>",
            "normalized_value": "<normalized_value_if_applicable>",
            "confidence": <0.0-1.0>
        }}
    ]
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
    
    async def extract(self, text: str, language: Language) -> list[Entity]:
        """Extract entities using Bedrock LLM."""
        import json
        
        try:
            client = await self._get_client()
            
            response = await client.invoke_model(
                modelId=self.settings.aws.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "user", "content": self.EXTRACTION_PROMPT.format(query=text)}
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
                return []
            
            entities = []
            for ent in result.get("entities", []):
                try:
                    entity_type = EntityType(ent["type"])
                    entities.append(Entity(
                        type=entity_type,
                        value=str(ent["value"]),
                        confidence=float(ent.get("confidence", 0.8)),
                        normalized_value=ent.get("normalized_value")
                    ))
                except (ValueError, KeyError):
                    continue
            
            return entities
            
        except Exception as e:
            logger.error("bedrock_entity_extraction_failed", error=str(e))
            # Fall back to pattern extraction
            fallback = PatternEntityExtractor()
            return await fallback.extract(text, language)


class HybridEntityExtractor(BaseEntityExtractor):
    """
    Combines pattern and LLM extraction for best results.
    Patterns for structured data, LLM for complex/ambiguous cases.
    """
    
    def __init__(self) -> None:
        self.pattern_extractor = PatternEntityExtractor()
        self.llm_extractor = BedrockEntityExtractor()
    
    async def extract(self, text: str, language: Language) -> list[Entity]:
        """Extract entities using hybrid approach."""
        # Always run pattern extraction (fast, reliable for structured data)
        pattern_entities = await self.pattern_extractor.extract(text, language)
        
        # For complex queries, also use LLM
        # Indicators of complexity: long text, Hindi script, questions
        is_complex = (
            len(text) > 50 or
            any(ord(c) > 0x900 and ord(c) < 0x97F for c in text) or  # Devanagari
            "?" in text or
            "कहाँ" in text or "कितना" in text  # Hindi question words
        )
        
        if is_complex:
            llm_entities = await self.llm_extractor.extract(text, language)
            
            # Merge entities, preferring pattern matches for same type
            pattern_types = {e.type for e in pattern_entities}
            for llm_entity in llm_entities:
                if llm_entity.type not in pattern_types:
                    pattern_entities.append(llm_entity)
        
        return pattern_entities


class EntityExtractorFactory:
    """Factory for creating entity extractor instances."""
    
    @staticmethod
    def create(extractor_type: str = "hybrid") -> BaseEntityExtractor:
        """Create an entity extractor."""
        extractors = {
            "pattern": PatternEntityExtractor,
            "bedrock": BedrockEntityExtractor,
            "hybrid": HybridEntityExtractor,
        }
        
        extractor_class = extractors.get(extractor_type)
        if not extractor_class:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        return extractor_class()
