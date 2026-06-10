"""Application settings — all environment-driven, with local-dev defaults
matching docker-compose. Phase 3 services (FastAPI, Bedrock, Redis) read
from this single object."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://compliance:compliance@localhost:5432/compliance"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    triage_fast_track_below: float = 0.40
    senior_band_at_or_above: float = 0.70
    escalation_probability_at_or_above: float = 0.50
    max_resubmit_attempts: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
