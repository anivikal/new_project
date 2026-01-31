"""
Application configuration settings using Pydantic Settings.
Supports environment variables and .env files for flexible deployment.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSSettings(BaseSettings):
    """AWS-specific configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AWS_")
    
    region: str = Field(default="ap-south-1", description="AWS region for services")
    access_key_id: str | None = Field(default=None, description="AWS access key (optional if using IAM roles)")
    secret_access_key: str | None = Field(default=None, description="AWS secret key (optional if using IAM roles)")
    
    # Bedrock settings
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID for LLM inference"
    )
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v1",
        description="Bedrock embedding model for semantic search"
    )
    
    # Polly TTS settings
    polly_voice_hindi: str = Field(default="Aditi", description="AWS Polly voice for Hindi")
    polly_voice_english: str = Field(default="Kajal", description="AWS Polly voice for English/Hinglish")
    
    # Transcribe settings
    transcribe_language_code: str = Field(default="hi-IN", description="Primary transcribe language")


class RedisSettings(BaseSettings):
    """Redis configuration for session and caching."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: str | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    session_ttl: int = Field(default=3600, description="Session TTL in seconds")
    
    @property
    def url(self) -> str:
        """Build Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class MongoSettings(BaseSettings):
    """MongoDB configuration for conversation history."""
    
    model_config = SettingsConfigDict(env_prefix="MONGO_")
    
    uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    database: str = Field(default="voicebot", description="Database name")
    conversations_collection: str = Field(default="conversations", description="Conversations collection")
    handoffs_collection: str = Field(default="handoffs", description="Handoffs collection")


class VoicebotSettings(BaseSettings):
    """Core voicebot configuration."""
    
    model_config = SettingsConfigDict(env_prefix="VOICEBOT_")
    
    # Language settings
    primary_language: str = Field(default="hi-IN", description="Primary language code")
    secondary_language: str = Field(default="bn-IN", description="Secondary language (Bengali)")
    supported_languages: list[str] = Field(
        default=["hi-IN", "en-IN", "bn-IN", "ta-IN"],
        description="All supported language codes"
    )
    
    # Confidence thresholds
    intent_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="Minimum confidence for intent classification"
    )
    handoff_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Confidence below this triggers handoff consideration"
    )
    
    # Sentiment thresholds
    sentiment_negative_threshold: float = Field(
        default=0.35,
        ge=0.0, le=1.0,
        description="Sentiment score below this is considered negative"
    )
    sentiment_drop_threshold: float = Field(
        default=0.2,
        ge=0.0, le=1.0,
        description="Sentiment drop between turns to trigger handoff"
    )
    
    # Repetition settings
    max_repetitions_before_handoff: int = Field(
        default=3,
        ge=1,
        description="Max times to repeat clarification before handoff"
    )
    
    # Audio settings
    audio_sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    audio_channels: int = Field(default=1, description="Audio channels (mono)")
    vad_aggressiveness: int = Field(default=2, ge=0, le=3, description="VAD aggressiveness level")
    
    # Timeouts
    silence_timeout_ms: int = Field(default=2000, description="Silence timeout in milliseconds")
    max_call_duration_seconds: int = Field(default=600, description="Maximum call duration")
    
    @field_validator("supported_languages", mode="before")
    @classmethod
    def parse_languages(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v


class APISettings(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of workers")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: list[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    api_key: str | None = Field(default=None, description="API key for authentication")
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class CRMSettings(BaseSettings):
    """CRM/Jarvis integration settings."""
    
    model_config = SettingsConfigDict(env_prefix="CRM_")
    
    base_url: str = Field(
        default="https://api.batterysmart.com",
        description="CRM API base URL"
    )
    api_key: str | None = Field(default=None, description="CRM API key")
    timeout_seconds: int = Field(default=10, description="API timeout")


class GroqSettings(BaseSettings):
    """Groq LLM API settings."""
    
    model_config = SettingsConfigDict(env_prefix="GROQ_")
    
    api_key: str = Field(
        default="",
        description="Groq API key for LLM inference"
    )
    model_id: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq model to use"
    )
    timeout_seconds: int = Field(default=30, description="API timeout")
    max_tokens: int = Field(default=500, description="Max tokens for response")


class IndicASRSettings(BaseSettings):
    """AI4Bharat IndicConformer ASR settings."""
    
    model_config = SettingsConfigDict(env_prefix="INDIC_ASR_")
    
    hf_token: str = Field(
        default="",
        description="HuggingFace token for model access"
    )
    model_id: str = Field(
        default="ai4bharat/indic-conformer-600m-multilingual",
        description="IndicConformer model ID"
    )
    decoding_method: str = Field(
        default="ctc",
        description="Decoding method: ctc or rnnt"
    )
    device: str = Field(
        default="cpu",
        description="Device to run model on: cpu or cuda"
    )


class TTSSettings(BaseSettings):
    """TTS provider settings."""
    
    model_config = SettingsConfigDict(env_prefix="TTS_")
    
    provider: str = Field(
        default="gtts",
        description="TTS provider: gtts, pyttsx3, polly"
    )
    cache_size: int = Field(default=500, description="TTS cache size")


class Settings(BaseSettings):
    """Main application settings aggregating all sub-settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Sub-settings
    aws: AWSSettings = Field(default_factory=AWSSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    voicebot: VoicebotSettings = Field(default_factory=VoicebotSettings)
    api: APISettings = Field(default_factory=APISettings)
    crm: CRMSettings = Field(default_factory=CRMSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    indic_asr: IndicASRSettings = Field(default_factory=IndicASRSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
