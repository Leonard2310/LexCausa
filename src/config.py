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

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, model_validator
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
    _groq_api_keys: list[str] = []  # populated dynamically by validator
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
    aqa_strength_ratio_by_type: dict = Field(
        default_factory=lambda: {
            "contradiction": 0.0,
            "exception": 0.0,
            "derogation": 0.0,
            "extinction": 0.0,
            "factual_impediment": 0.5,
            "general_opposition": 0.95,
        },
        alias="AQA_STRENGTH_RATIO_BY_TYPE",
        description="Per-attack-type strength ratio thresholds. "
        "An attack of a given type needs attacker_base >= ratio * target_base. "
        "Types classified by the LLM as specific (contradiction, exception, "
        "derogation, extinction) have ratio=0.0, so they always pass. "
        "Only general_opposition and factual_impediment are gated.",
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
    # Classifier LLM Defaults (task-specific overrides)
    # =========================================================================
    classifier_temperature: float = Field(
        default=0.0,
        alias="CLASSIFIER_TEMPERATURE",
        description="Temperature for classification tasks (stance, causality, attack-type). "
        "Deterministic by default.",
    )
    classifier_max_tokens: int = Field(
        default=50,
        alias="CLASSIFIER_MAX_TOKENS",
        description="Max tokens for short classification responses.",
    )
    nli_max_tokens: int = Field(
        default=64,
        alias="NLI_MAX_TOKENS",
        description="Max tokens for NLI / contradiction detection LLM calls.",
    )
    repair_max_tokens: int = Field(
        default=2048,
        alias="REPAIR_MAX_TOKENS",
        description="Max tokens for chain repair / rewrite LLM calls.",
    )
    attack_type_max_tokens: int = Field(
        default=256,
        alias="ATTACK_TYPE_MAX_TOKENS",
        description="Max tokens for attack-type classification LLM calls.",
    )
    taxonomy_max_tokens: int = Field(
        default=20,
        alias="TAXONOMY_MAX_TOKENS",
        description="Max tokens for causality taxonomy classification LLM calls.",
    )
    taxonomy_filter_max_tokens: int = Field(
        default=10,
        alias="TAXONOMY_FILTER_MAX_TOKENS",
        description="Max tokens for taxonomy norm relevance filter LLM calls.",
    )

    # =========================================================================
    # Resilience / Retry
    # =========================================================================
    model_down_ttl: float = Field(
        default=300.0,
        alias="MODEL_DOWN_TTL",
        description="Seconds to wait before retrying a model marked as down.",
    )
    chain_max_retries: int = Field(
        default=5,
        alias="CHAIN_MAX_RETRIES",
        description="Maximum generation attempts for reasoning / counter-reasoning chains.",
    )
    chain_max_steps: int = Field(
        default=10,
        alias="CHAIN_MAX_STEPS",
        description="Safety cap: maximum reasoning steps per iterative chain (LLM decides when to stop).",
    )
    chain_min_steps: int = Field(
        default=3,
        alias="CHAIN_MIN_STEPS",
        description="Minimum reasoning steps before the LLM is allowed to conclude.",
    )

    # =========================================================================
    # Text Truncation (prompt context limits)
    # =========================================================================
    truncation_statute_text: int = Field(
        default=800,
        alias="TRUNCATION_STATUTE_TEXT",
        description="Max chars of statute text sent to classifier prompts.",
    )
    truncation_summary: int = Field(
        default=600,
        alias="TRUNCATION_SUMMARY",
        description="Max chars of summary / precedent text for prompts.",
    )
    truncation_nli_text: int = Field(
        default=600,
        alias="TRUNCATION_NLI_TEXT",
        description="Max chars per passage for NLI / attack-type prompts.",
    )
    truncation_chain_text: int = Field(
        default=3000,
        alias="TRUNCATION_CHAIN_TEXT",
        description="Max chars of full chain text for repair prompts.",
    )
    truncation_context: int = Field(
        default=500,
        alias="TRUNCATION_CONTEXT",
        description="Max chars of contextual snippets in prompts.",
    )
    truncation_tool_testo: int = Field(
        default=500,
        alias="TRUNCATION_TOOL_TESTO",
        description="Max chars of statute text returned by search tools (neo4j_tools).",
    )
    truncation_tool_summary: int = Field(
        default=500,
        alias="TRUNCATION_TOOL_SUMMARY",
        description="Max chars of precedent summary returned by search tools.",
    )
    truncation_tool_excerpt: int = Field(
        default=300,
        alias="TRUNCATION_TOOL_EXCERPT",
        description="Max chars of precedent chunk_text excerpt returned by search tools.",
    )
    truncation_prompt_testo: int = Field(
        default=500,
        alias="TRUNCATION_PROMPT_TESTO",
        description="Max chars of statute text in base agent prompt formatting.",
    )
    truncation_prompt_summary: int = Field(
        default=300,
        alias="TRUNCATION_PROMPT_SUMMARY",
        description="Max chars of precedent summary in base agent prompt formatting.",
    )
    claim_classifier_max_tokens: int = Field(
        default=64,
        alias="CLAIM_CLASSIFIER_MAX_TOKENS",
        description="Max completion tokens for claim classifier LLM calls.",
    )

    # =========================================================================
    # AQA Scoring Weights & Thresholds (extended)
    # =========================================================================
    aqa_position_weight_foundational: float = Field(
        default=1.5,
        alias="AQA_POSITION_WEIGHT_FOUNDATIONAL",
        description="Position weight for foundational premises (premessa, presupposto, principio).",
    )
    aqa_position_weight_severe: float = Field(
        default=1.3,
        alias="AQA_POSITION_WEIGHT_SEVERE",
        description="Position weight for severe norm categories (delitti, contravvenzioni, tutela_diritti).",
    )
    aqa_position_weight_default: float = Field(
        default=1.0,
        alias="AQA_POSITION_WEIGHT_DEFAULT",
        description="Default position weight for other links.",
    )
    aqa_max_age: float = Field(
        default=50.0,
        alias="AQA_MAX_AGE",
        description="Maximum precedent age (years) for recency scoring. Older = 0 recency.",
    )
    aqa_default_confidence: float = Field(
        default=0.7,
        alias="AQA_DEFAULT_CONFIDENCE",
        description="Default confidence when stance confidence is missing from precedent metadata.",
    )
    aqa_precedent_sim_threshold: float = Field(
        default=0.5,
        alias="AQA_PRECEDENT_SIM_THRESHOLD",
        description="Minimum similarity score for matching a precedent to a link.",
    )
    aqa_dominant_attacks_limit: int = Field(
        default=10,
        alias="AQA_DOMINANT_ATTACKS_LIMIT",
        description="Max number of dominant attacks to include in AQA report.",
    )
    aqa_bindingness_map: dict = Field(
        default_factory=lambda: {
            "cassazione": 1.0,
            "appello": 0.7,
            "tribunale": 0.4,
            "other": 0.0,
        },
        alias="AQA_BINDINGNESS_MAP",
        description="Bindingness score by court level (substring match).",
    )

    # =========================================================================
    # Readability & Coherence Weights
    # =========================================================================
    readability_structure_weight: float = Field(
        default=0.5,
        alias="READABILITY_STRUCTURE_WEIGHT",
        description="Weight of structure score in argument quality.",
    )
    readability_quality_weight: float = Field(
        default=0.5,
        alias="READABILITY_QUALITY_WEIGHT",
        description="Weight of quality score in argument quality.",
    )
    coherence_base_weight: float = Field(
        default=0.7,
        alias="COHERENCE_BASE_WEIGHT",
        description="Weight for base similarity in coherence scoring.",
    )
    coherence_chain_weight: float = Field(
        default=0.3,
        alias="COHERENCE_CHAIN_WEIGHT",
        description="Weight for inter-sentence similarity chain in coherence scoring.",
    )

    # =========================================================================
    # Consistency Checker Thresholds
    # =========================================================================
    cc_text_match_threshold: float = Field(
        default=0.8,
        alias="CC_TEXT_MATCH_THRESHOLD",
        description="Minimum cosine similarity for a cited text to be considered matching the DB text.",
    )
    cc_consistency_existence_weight: float = Field(
        default=0.7,
        alias="CC_CONSISTENCY_EXISTENCE_WEIGHT",
        description="Weight for citation existence in the consistency score.",
    )
    cc_consistency_text_weight: float = Field(
        default=0.3,
        alias="CC_CONSISTENCY_TEXT_WEIGHT",
        description="Weight for text match in the consistency score.",
    )
    cc_core_threshold: int = Field(
        default=2,
        alias="CC_CORE_THRESHOLD",
        description="Minimum number of core indicators for an article to be classified as CORE.",
    )
    cc_conclusion_bonus: int = Field(
        default=2,
        alias="CC_CONCLUSION_BONUS",
        description="Extra core-indicator points when article is cited in the conclusion.",
    )
    cc_occurrence_threshold: int = Field(
        default=3,
        alias="CC_OCCURRENCE_THRESHOLD",
        description="Number of text occurrences that adds a core indicator point.",
    )

    # =========================================================================
    # API Metadata
    # =========================================================================
    api_version: str = Field(
        default="0.2.0",
        alias="API_VERSION",
        description="API version string returned by /health.",
    )

    # =========================================================================
    # Paths
    # =========================================================================
    @model_validator(mode="after")
    def _discover_groq_api_keys(self) -> "Settings":
        """Scan environment for all GROQ_API_KEY_V* variables.

        Discovers keys dynamically (V1, V2, …, V99) so adding more
        keys to .env is all that's needed — no code changes required.
        """
        keys: list[str] = []
        # Start with the primary key (V1) that Pydantic already loaded
        if self.groq_api_key:
            keys.append(self.groq_api_key)
        # Discover V2, V3, … VN
        idx = 2
        while True:
            val = os.environ.get(f"GROQ_API_KEY_V{idx}", "")
            if not val:
                break
            if val not in keys:  # avoid duplicates
                keys.append(val)
            idx += 1
        object.__setattr__(self, "_groq_api_keys", keys)
        return self

    @property
    def groq_api_keys(self) -> list[str]:
        """Get all available Groq API keys (non-empty, dynamically discovered)."""
        return list(self._groq_api_keys)

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

        if not self.groq_api_keys:
            issues.append("No GROQ_API_KEY_V* keys found in environment")

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
