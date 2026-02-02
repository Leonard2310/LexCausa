"""
LexCausa Configuration Module.

Centralized configuration management using Pydantic Settings.
Loads environment variables from .env file and provides typed access.

Usage:
    from config import settings

    # Access configuration
    neo4j_uri = settings.neo4j_uri
    groq_api_key = settings.groq_api_key
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find project root and load .env
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
_env_file = _project_root / ".env"

# Load .env file
load_dotenv(_env_file)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    The .env file in project root is automatically loaded.
    """

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Neo4j Configuration
    # =========================================================================
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="neo4jpassword", alias="NEO4J_PASSWORD")

    # =========================================================================
    # Groq Cloud Configuration
    # =========================================================================
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct", alias="GROQ_MODEL"
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # =========================================================================
    # Embedding Model Configuration
    # =========================================================================
    embedding_model: str = Field(
        default="nlpaueb/legal-bert-base-uncased", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")
    embedding_max_length: int = Field(default=512, alias="EMBEDDING_MAX_LENGTH")

    # =========================================================================
    # API Server Configuration
    # =========================================================================
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # =========================================================================
    # Search & Retrieval Defaults
    # =========================================================================
    search_top_k_default: int = Field(
        default=20,
        alias="SEARCH_TOP_K_DEFAULT",
        description="Default number of statute results to return when not specified.",
    )
    search_use_top_n_libri: int = Field(
        default=3,
        alias="SEARCH_USE_TOP_N_LIBRI",
        description="How many top classified libri to query during statute search.",
    )
    precedents_limit_default: int = Field(
        default=10,
        alias="PRECEDENTS_LIMIT_DEFAULT",
        description="Default number of precedents to retrieve when not specified.",
    )

    # =========================================================================
    # Paths
    # =========================================================================
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return _project_root

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return _project_root / "src" / "data"

    @property
    def embeddings_dir(self) -> Path:
        """Get embeddings directory."""
        return self.data_dir / "embeddings"

    @property
    def statutes_dir(self) -> Path:
        """Get statutes directory."""
        return self.data_dir / "statuti"

    @property
    def taxonomy_path(self) -> Path:
        """Get causality taxonomy file path."""
        return self.project_root / "src" / "agents" / "tools" / "config_taxonomy.json"

    def validate_config(self) -> dict:
        """Validate configuration and return status."""
        issues = []

        if not self.groq_api_key:
            issues.append("GROQ_API_KEY not set")

        if not self.taxonomy_path.exists():
            issues.append(f"Taxonomy file not found: {self.taxonomy_path}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "neo4j_uri": self.neo4j_uri,
            "groq_model": self.groq_model,
            "embedding_model": self.embedding_model,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure only one Settings instance is created.
    """
    return Settings()


# Global settings instance for easy import
settings = get_settings()


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("LexCausa Configuration")
    print("=" * 60)
    print(f"Project Root: {settings.project_root}")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"Groq Model: {settings.groq_model}")
    print(f"Embedding Model: {settings.embedding_model}")
    print()
    validation = settings.validate_config()
    print(f"Valid: {validation['valid']}")
    if validation["issues"]:
        print(f"Issues: {validation['issues']}")
