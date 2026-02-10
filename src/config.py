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
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY_V1")
    groq_api_key_v2: str = Field(default="", alias="GROQ_API_KEY_V2")
    groq_api_key_v3: str = Field(default="", alias="GROQ_API_KEY_V3")
    groq_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct", alias="GROQ_MODEL"
    )
    groq_fallback_model: str = Field(
        default="meta-llama/llama-4-maverick-17b-128e-instruct",
        alias="GROQ_FALLBACK_MODEL",
    )

    # =========================================================================
    # Retry / Resilience Configuration
    # =========================================================================
    groq_max_retries: int = Field(
        default=3,
        alias="GROQ_MAX_RETRIES",
        description="Maximum number of retries per API call (key rotation + model fallback).",
    )
    groq_retry_base_delay: float = Field(
        default=1.0,
        alias="GROQ_RETRY_BASE_DELAY",
        description="Base delay in seconds for exponential backoff between retries.",
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=8192, alias="LLM_MAX_TOKENS")

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
        default=100,
        alias="SEARCH_TOP_K_DEFAULT",
        description="Default number of statute results to return when not specified.",
    )
    search_min_kept_statutes: int = Field(
        default=10,
        alias="SEARCH_MIN_KEPT_STATUTES",
        description="Minimum number of statutes to keep after relevance filtering. "
        "If fewer are kept, the search expands progressively by +10 until this threshold is met.",
    )
    search_expansion_step: int = Field(
        default=10,
        alias="SEARCH_EXPANSION_STEP",
        description="Number of additional statutes to fetch per expansion round.",
    )
    search_max_expansions: int = Field(
        default=5,
        alias="SEARCH_MAX_EXPANSIONS",
        description="Maximum number of progressive expansion rounds to avoid infinite loops.",
    )
    search_use_top_n_libri: int = Field(
        default=3,
        alias="SEARCH_USE_TOP_N_LIBRI",
        description="How many top classified libri to query during statute search.",
    )
    precedents_limit_default: int = Field(
        default=5,
        alias="PRECEDENTS_LIMIT_DEFAULT",
        description="Default number of precedents to retrieve when not specified.",
    )

    # =========================================================================
    # AQA / Polisher Evaluator Defaults
    # =========================================================================
    aqa_enabled: bool = Field(
        default=True,
        alias="AQA_ENABLED",
        description="Enable the AQA scoring phase in the Polisher-Evaluator.",
    )
    aqa_alpha: float = Field(
        default=0.3,
        alias="AQA_ALPHA",
        description="AQA weight for Cogency.",
    )
    aqa_beta: float = Field(
        default=0.4,
        alias="AQA_BETA",
        description="AQA weight for NormSupport.",
    )
    aqa_gamma: float = Field(
        default=0.3,
        alias="AQA_GAMMA",
        description="AQA weight for Semantics.",
    )
    aqa_attack_top_k: int = Field(
        default=3,
        alias="AQA_ATTACK_TOP_K",
        description="Top-K cross-attacks to keep per link for explainability.",
    )
    aqa_min_semantic_overlap: float = Field(
        default=0.5,
        alias="AQA_MIN_SEMANTIC_OVERLAP",
        description="Minimum semantic similarity between two links for an attack to be valid. "
        "Attacks below this threshold are filtered out.",
    )
    aqa_min_strength_ratio: float = Field(
        default=1.2,
        alias="AQA_MIN_STRENGTH_RATIO",
        description="Attacker base_score must be >= this ratio × target base_score. "
        "Ensures only meaningfully stronger arguments can inflict damage.",
    )
    aqa_damage_factor: float = Field(
        default=0.5,
        alias="AQA_DAMAGE_FACTOR",
        description="Scaling factor applied to the excess damage (attacker_base - target_base). "
        "Lower values make attacks less destructive.",
    )
    aqa_allow_factual_attacks: bool = Field(
        default=True,
        alias="AQA_ALLOW_FACTUAL_ATTACKS",
        description="Allow factual (non-normative) arguments to attack normative ones.",
    )
    aqa_allow_cross_codice: bool = Field(
        default=True,
        alias="AQA_ALLOW_CROSS_CODICE",
        description="Allow cross-codice attacks via double relevance and bridge norms.",
    )
    aqa_double_relevance_crimes: list[str] = Field(
        default_factory=lambda: [
            "595",
            "590",
            "640",
            "624",
            "635",
            "572",
            "582",
        ],
        alias="AQA_DOUBLE_RELEVANCE_CRIMES",
        description="Article numbers of crimes with automatic civil relevance.",
    )
    aqa_bridge_norms: list[str] = Field(
        default_factory=lambda: ["185", "198"],
        alias="AQA_BRIDGE_NORMS",
        description="Article numbers that bridge penale→civile liability.",
    )
    aqa_attack_type_multipliers: dict = Field(
        default_factory=lambda: {
            "contradiction": 1.5,
            "exception": 1.3,
            "derogation": 1.4,
            "extinction": 1.6,
            "factual_impediment": 1.2,
            "general_opposition": 1.0,
        },
        alias="AQA_ATTACK_TYPE_MULTIPLIERS",
        description="Damage multipliers by attack type.",
    )
    aqa_nli_min_contradiction_score: float = Field(
        default=0.65,
        alias="AQA_NLI_MIN_CONTRADICTION_SCORE",
        description="Minimum NLI contradiction probability to bypass strength ratio. "
        "If the NLI model returns contradiction with score >= this threshold, "
        "the attack proceeds regardless of strength ratio.",
    )
    aqa_strength_ratio_by_type: dict = Field(
        default_factory=lambda: {
            "contradiction": 0.0,
            "exception": 0.75,
            "derogation": 0.8,
            "extinction": 0.65,
            "factual_impediment": 0.9,
            "general_opposition": 0.95,
        },
        alias="AQA_STRENGTH_RATIO_BY_TYPE",
        description="Per-attack-type strength ratio thresholds. "
        "An attack of a given type needs raw_attack >= ratio * target_base. "
        "Set to 0.0 to always allow that type (e.g. NLI-confirmed contradictions).",
    )

    aqa_nli_min_entailment_score: float = Field(
        default=0.55,
        alias="AQA_NLI_MIN_ENTAILMENT_SCORE",
        description="Minimum NLI entailment probability to bypass strength ratio. "
        "If the NLI model returns entailment with score >= this threshold, "
        "the attack proceeds regardless of strength ratio.",
    )
    
    aqa_severity_book_map: dict = Field(
        default_factory=lambda: {
            "persone_famiglia": "I_civile",
            "successioni": "II_civile",
            "proprieta": "III_civile",
            "diritti_reali": "III_civile",
            "obbligazioni": "IV_civile",
            "contratti_generali": "IV_civile",
            "contratti_speciali": "IV_civile",
            "responsabilita": "IV_civile",
            "lavoro": "V_civile",
            "tutela_diritti": "VI_civile",
            "prescrizione": "VI_civile",
            "decadenza": "VI_civile",
            "generale": "I_penale",
            "delitti": "II_penale",
            "delitti_persona": "II_penale",
            "delitti_patrimonio": "II_penale",
            "delitti_stato": "II_penale",
            "contravvenzioni": "III_penale",
        },
        alias="AQA_SEVERITY_BOOK_MAP",
        description="Map severity_category → libro identifier for same-book checks.",
    )
    aqa_verdict_pos_threshold: float = Field(
        default=0.2,
        alias="AQA_VERDICT_POS_THRESHOLD",
        description="Final plausibility threshold for 'plausible'.",
    )
    aqa_verdict_neg_threshold: float = Field(
        default=-0.2,
        alias="AQA_VERDICT_NEG_THRESHOLD",
        description="Final plausibility threshold for 'implausible'.",
    )
    aqa_embedding_model: str = Field(
        default="all-mpnet-base-v2",
        alias="AQA_EMBEDDING_MODEL",
        description="Sentence-transformers model for overlap embeddings.",
    )
    aqa_nli_model: str = Field(
        default="MoritzLaurer/DeBERTa-v3-base-mnli",
        alias="AQA_NLI_MODEL",
        description="NLI model for semantics entailment.",
    )
    aqa_argument_quality_model: str = Field(
        default="",
        alias="AQA_ARGUMENT_QUALITY_MODEL",
        description="Optional argument-quality classifier model name.",
    )
    aqa_argument_quality_use_model: bool = Field(
        default=False,
        alias="AQA_ARGUMENT_QUALITY_USE_MODEL",
        description="Use argument-quality model when available.",
    )
    aqa_tfidf_max_features: int = Field(
        default=5000,
        alias="AQA_TFIDF_MAX_FEATURES",
        description="Max features for TF-IDF overlap vectors.",
    )
    aqa_normsupport_max_citations: int = Field(
        default=3,
        alias="AQA_NORMSUPPORT_MAX_CITATIONS",
        description="Cap for normalized citation count in NormSupport.",
    )
    aqa_normsupport_citation_weight: float = Field(
        default=0.7,
        alias="AQA_NORMSUPPORT_CITATION_WEIGHT",
        description="Weight for citation count in NormSupport.",
    )
    aqa_normsupport_retrieved_weight: float = Field(
        default=0.3,
        alias="AQA_NORMSUPPORT_RETRIEVED_WEIGHT",
        description="Weight for retrieved_norms similarity in NormSupport.",
    )
    aqa_normsupport_retrieved_agg: str = Field(
        default="avg",
        alias="AQA_NORMSUPPORT_RETRIEVED_AGG",
        description="Aggregation for retrieved_norms similarity: avg or max.",
    )
    aqa_severity_map_penale: dict = Field(
        default_factory=lambda: {
            "i": "generale",
            "ii": "delitti",
            "iii": "contravvenzioni",
            "primo": "generale",
            "secondo": "delitti",
            "terzo": "contravvenzioni",
        },
        alias="AQA_SEVERITY_MAP_PENALE",
        description="Mapping from libro token to severity category (Codice Penale).",
    )
    aqa_severity_map_civile: dict = Field(
        default_factory=lambda: {
            "i": "persone_famiglia",
            "ii": "successioni",
            "iii": "proprieta",
            "iv": "obbligazioni",
            "v": "lavoro",
            "vi": "tutela_diritti",
            "primo": "persone_famiglia",
            "secondo": "successioni",
            "terzo": "proprieta",
            "quarto": "obbligazioni",
            "quinto": "lavoro",
            "sesto": "tutela_diritti",
        },
        alias="AQA_SEVERITY_MAP_CIVILE",
        description="Mapping from libro token to severity category (Codice Civile).",
    )

    # =========================================================================
    # Paths
    # =========================================================================
    @property
    def groq_api_keys(self) -> list[str]:
        """Get all available Groq API keys (non-empty)."""
        return [
            k
            for k in [self.groq_api_key, self.groq_api_key_v2, self.groq_api_key_v3]
            if k
        ]

    @property
    def groq_models(self) -> list[str]:
        """Get ordered list of models: primary first, then fallback."""
        return [self.groq_model, self.groq_fallback_model]

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
