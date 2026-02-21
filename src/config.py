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
from typing import ClassVar

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
    # Model catalog is code-owned (not env-driven).
    MODEL_ALIAS_MAP: ClassVar[dict[str, str]] = {
        "groq_llama_scout_17b": "meta-llama/llama-4-scout-17b-16e-instruct",
        "groq_llama_maverick_17b": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "gpt_oss_120b": "openai/gpt-oss-120b",
    }
    PIPELINE_MODEL_ORDER_ALIASES: ClassVar[list[str]] = [
        "groq_llama_maverick_17b",
        "groq_llama_scout_17b",
    ]
    REASONER_DEFAULT_MODEL_ALIAS: ClassVar[str] = "gpt_oss_120b"
    COUNTER_DEFAULT_MODEL_ALIAS: ClassVar[str] = "gpt_oss_120b"
    reasoner_model_fallback_aliases: list[str] = Field(
        default_factory=lambda: [
            "gpt_oss_120b",
            "groq_llama_maverick_17b",
            "groq_llama_scout_17b",
        ],
        alias="REASONER_MODEL_FALLBACK_ALIASES",
        description="Ordered model aliases used as resilient fallback chain for Reasoner.",
    )
    counter_model_fallback_aliases: list[str] = Field(
        default_factory=lambda: [
            "gpt_oss_120b",
            "groq_llama_maverick_17b",
            "groq_llama_scout_17b",
        ],
        alias="COUNTER_MODEL_FALLBACK_ALIASES",
        description="Ordered model aliases used as resilient fallback chain for Counter-Reasoner.",
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
    search_expansion_max_zero_gain_rounds: int = Field(
        default=2,
        alias="SEARCH_EXPANSION_MAX_ZERO_GAIN_ROUNDS",
        description="Early-stop retrieval expansion after this many consecutive rounds with zero newly kept statutes.",
    )
    search_use_top_n_libri: int = Field(
        default=3,
        alias="SEARCH_USE_TOP_N_LIBRI",
        description="How many top classified libri to query during statute search.",
    )
    search_hybrid_vector_weight: float = Field(
        default=0.30,
        alias="SEARCH_HYBRID_VECTOR_WEIGHT",
        description="Weight of vector-ranked candidates in hybrid statute retrieval score fusion.",
    )
    search_hybrid_fulltext_weight: float = Field(
        default=0.70,
        alias="SEARCH_HYBRID_FULLTEXT_WEIGHT",
        description="Weight of fulltext-ranked candidates in hybrid statute retrieval score fusion.",
    )
    search_hybrid_admin_vector_weight: float = Field(
        default=0.20,
        alias="SEARCH_HYBRID_ADMIN_VECTOR_WEIGHT",
        description="Vector weight used for administrative-code hybrid retrieval.",
    )
    search_hybrid_admin_fulltext_weight: float = Field(
        default=0.80,
        alias="SEARCH_HYBRID_ADMIN_FULLTEXT_WEIGHT",
        description="Fulltext weight used for administrative-code hybrid retrieval.",
    )
    search_hybrid_civile_vector_weight: float = Field(
        default=0.20,
        alias="SEARCH_HYBRID_CIVILE_VECTOR_WEIGHT",
        description="Vector weight used for civil-code hybrid retrieval.",
    )
    search_hybrid_civile_fulltext_weight: float = Field(
        default=0.80,
        alias="SEARCH_HYBRID_CIVILE_FULLTEXT_WEIGHT",
        description="Fulltext weight used for civil-code hybrid retrieval.",
    )
    search_hybrid_penale_vector_weight: float = Field(
        default=0.45,
        alias="SEARCH_HYBRID_PENALE_VECTOR_WEIGHT",
        description="Vector weight used for criminal-code hybrid retrieval.",
    )
    search_hybrid_penale_fulltext_weight: float = Field(
        default=0.55,
        alias="SEARCH_HYBRID_PENALE_FULLTEXT_WEIGHT",
        description="Fulltext weight used for criminal-code hybrid retrieval.",
    )
    search_hybrid_candidate_multiplier: int = Field(
        default=6,
        alias="SEARCH_HYBRID_CANDIDATE_MULTIPLIER",
        description="Candidate expansion factor: preliminary candidates = top_k * multiplier before fusion.",
    )
    search_hybrid_candidate_min: int = Field(
        default=40,
        alias="SEARCH_HYBRID_CANDIDATE_MIN",
        description="Minimum number of preliminary candidates per source/libro before fusion.",
    )
    search_hybrid_fused_pool_multiplier: int = Field(
        default=2,
        alias="SEARCH_HYBRID_FUSED_POOL_MULTIPLIER",
        description="Multiplier for intermediate fused pool size before final top_k truncation.",
    )
    search_hybrid_filter_priority_decay: float = Field(
        default=0.4,
        alias="SEARCH_HYBRID_FILTER_PRIORITY_DECAY",
        description="Priority decay applied to secondary/tertiary classifier filters in fused ranking.",
    )
    search_hybrid_filter_priority_floor: float = Field(
        default=0.35,
        alias="SEARCH_HYBRID_FILTER_PRIORITY_FLOOR",
        description="Lower bound for classifier-filter priority scaling in fused ranking.",
    )
    search_hybrid_keyword_bonus_max: float = Field(
        default=0.20,
        alias="SEARCH_HYBRID_KEYWORD_BONUS_MAX",
        description="Maximum lexical bonus added to fused score when query terms overlap article title/articolo.",
    )
    search_hybrid_keyword_bonus_scale: float = Field(
        default=0.25,
        alias="SEARCH_HYBRID_KEYWORD_BONUS_SCALE",
        description="Linear scale factor for lexical overlap bonus before clamping to max.",
    )
    search_hybrid_keyword_min_overlap_count: int = Field(
        default=1,
        alias="SEARCH_HYBRID_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus (general).",
    )
    search_hybrid_penale_keyword_min_overlap_count: int = Field(
        default=2,
        alias="SEARCH_HYBRID_PENALE_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus for criminal code.",
    )
    search_hybrid_civile_keyword_min_overlap_count: int = Field(
        default=3,
        alias="SEARCH_HYBRID_CIVILE_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus for civil code.",
    )
    search_hybrid_admin_keyword_min_overlap_count: int = Field(
        default=3,
        alias="SEARCH_HYBRID_ADMIN_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus for administrative code.",
    )
    search_hybrid_penale_zero_overlap_multiplier: float = Field(
        default=0.75,
        alias="SEARCH_HYBRID_PENALE_ZERO_OVERLAP_MULTIPLIER",
        description="Score multiplier for criminal-code results with zero lexical overlap against claim terms.",
    )
    search_hybrid_penale_low_overlap_multiplier: float = Field(
        default=0.94,
        alias="SEARCH_HYBRID_PENALE_LOW_OVERLAP_MULTIPLIER",
        description="Score multiplier for criminal-code results with overlap below bonus threshold.",
    )
    search_hybrid_civile_zero_overlap_multiplier: float = Field(
        default=0.8,
        alias="SEARCH_HYBRID_CIVILE_ZERO_OVERLAP_MULTIPLIER",
        description="Score multiplier for civil-code results with zero lexical overlap against claim terms.",
    )
    search_hybrid_civile_low_overlap_multiplier: float = Field(
        default=1.0,
        alias="SEARCH_HYBRID_CIVILE_LOW_OVERLAP_MULTIPLIER",
        description="Score multiplier for civil-code results with overlap below bonus threshold.",
    )
    search_hybrid_admin_zero_overlap_multiplier: float = Field(
        default=0.95,
        alias="SEARCH_HYBRID_ADMIN_ZERO_OVERLAP_MULTIPLIER",
        description="Score multiplier for administrative-code results with zero lexical overlap against claim terms.",
    )
    search_hybrid_admin_low_overlap_multiplier: float = Field(
        default=1.0,
        alias="SEARCH_HYBRID_ADMIN_LOW_OVERLAP_MULTIPLIER",
        description="Score multiplier for administrative-code results with overlap below bonus threshold.",
    )
    search_hybrid_min_keyword_length: int = Field(
        default=4,
        alias="SEARCH_HYBRID_MIN_KEYWORD_LENGTH",
        description="Minimum token length considered for normalized LLM query keywords used in lexical bonus.",
    )
    search_query_terms_mode: str = Field(
        default="llm",
        alias="SEARCH_QUERY_TERMS_MODE",
        description="Query-term extraction mode for fulltext branch. Only 'llm' is supported.",
    )
    search_query_terms_llm_max_terms: int = Field(
        default=12,
        alias="SEARCH_QUERY_TERMS_LLM_MAX_TERMS",
        description="Maximum number of keywords requested when using LLM query-term extraction.",
    )
    search_query_terms_llm_max_tokens: int = Field(
        default=128,
        alias="SEARCH_QUERY_TERMS_LLM_MAX_TOKENS",
        description="Max completion tokens for LLM-based query-term extraction.",
    )
    search_cites_enabled: bool = Field(
        default=True,
        alias="SEARCH_CITES_ENABLED",
        description="Enable citation-based expansion via Neo4j CITES edges before retrieval filters.",
    )
    search_cites_per_article_limit: int = Field(
        default=10,
        alias="SEARCH_CITES_PER_ARTICLE_LIMIT",
        description="Maximum number of cited neighbors to fetch per initially retrieved statute.",
    )
    search_cites_max_additional: int = Field(
        default=60,
        alias="SEARCH_CITES_MAX_ADDITIONAL",
        description="Maximum number of citation-expanded statutes added to a retrieval round.",
    )
    search_cites_score_decay: float = Field(
        default=0.85,
        alias="SEARCH_CITES_SCORE_DECAY",
        description="Score decay applied to cited statutes relative to their strongest parent score.",
    )
    search_cites_multi_seed_bonus: float = Field(
        default=0.03,
        alias="SEARCH_CITES_MULTI_SEED_BONUS",
        description="Bonus added per additional parent statute citing the same target.",
    )
    search_retrieval_debug_top_n: int = Field(
        default=0,
        alias="SEARCH_RETRIEVAL_DEBUG_TOP_N",
        description="How many top retrieved statutes are printed in retrieval debug logs per stage.",
    )
    search_filter_log_top_n: int = Field(
        default=100,
        alias="SEARCH_FILTER_LOG_TOP_N",
        description="Maximum number of per-item keep/discard filter logs for statutes/precedents; remaining items are summarized.",
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
        description="Allow cross-codice attacks (penale vs civile).",
    )
    aqa_procedural_severity_categories: list[str] = Field(
        default_factory=lambda: [
            "prescrizione",
            "decadenza",
            "tutela_diritti",
            "processo",
        ],
        alias="AQA_PROCEDURAL_SEVERITY_CATEGORIES",
        description="Severity categories treated as procedural in ASPIC+ domain-rules checks.",
    )
    aqa_valid_attack_types: list[str] = Field(
        default_factory=lambda: [
            "contradiction",
            "exception",
            "derogation",
            "extinction",
            "factual_impediment",
            "general_opposition",
        ],
        alias="AQA_VALID_ATTACK_TYPES",
        description="Allowed attack-type labels for AQA attack classification.",
    )
    aqa_default_attack_type: str = Field(
        default="general_opposition",
        alias="AQA_DEFAULT_ATTACK_TYPE",
        description="Fallback attack type when classifier output is missing/invalid.",
    )

    aqa_attack_type_multipliers: dict = Field(
        default_factory=lambda: {
            "contradiction": 2.0,  # Contraddizione logica totale
            "exception": 1.7,  # Limita fortemente applicabilità
            "derogation": 2.2,  # Invalida direttamente la norma citata
            "extinction": 2.5,  # Estingue completamente l'argomento
            "factual_impediment": 1.5,  # Impedimento fattuale serio
            "general_opposition": 1.3,  # Opposizione generica ma rilevante
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
            "general_opposition": 0.5,
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
            "amministrativo": "AMM_L241",
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
    counter_second_pass_enabled: bool = Field(
        default=True,
        alias="COUNTER_SECOND_PASS_ENABLED",
        description="Enable targeted second-pass retrieval for Counter-Reasoner when against statutes are scarce.",
    )
    counter_second_pass_min_against_statutes: int = Field(
        default=8,
        alias="COUNTER_SECOND_PASS_MIN_AGAINST_STATUTES",
        description="If initial counter statutes are below this threshold, run targeted second-pass retrieval.",
    )
    counter_second_pass_top_k: int = Field(
        default=40,
        alias="COUNTER_SECOND_PASS_TOP_K",
        description="Top-K statutes requested in each targeted second-pass retrieval query.",
    )
    counter_second_pass_max_queries: int = Field(
        default=2,
        alias="COUNTER_SECOND_PASS_MAX_QUERIES",
        description="Maximum number of targeted retrieval queries generated from selected counter attacks.",
    )
    counter_second_pass_max_additional: int = Field(
        default=25,
        alias="COUNTER_SECOND_PASS_MAX_ADDITIONAL",
        description="Maximum additional statutes retained from the targeted second-pass retrieval.",
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
            "other": 0.3,
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
        mode = str(self.search_query_terms_mode or "").strip().lower()
        if mode != "llm":
            object.__setattr__(self, "search_query_terms_mode", "llm")

        reasoner_aliases = [
            str(a).strip()
            for a in self.reasoner_model_fallback_aliases
            if str(a).strip()
        ]
        if not reasoner_aliases:
            reasoner_aliases = [self.REASONER_DEFAULT_MODEL_ALIAS]
        object.__setattr__(self, "reasoner_model_fallback_aliases", reasoner_aliases)

        counter_aliases = [
            str(a).strip()
            for a in self.counter_model_fallback_aliases
            if str(a).strip()
        ]
        if not counter_aliases:
            counter_aliases = [self.COUNTER_DEFAULT_MODEL_ALIAS]
        object.__setattr__(self, "counter_model_fallback_aliases", counter_aliases)

        procedural_categories = [
            str(c).strip().lower()
            for c in self.aqa_procedural_severity_categories
            if str(c).strip()
        ]
        object.__setattr__(
            self,
            "aqa_procedural_severity_categories",
            procedural_categories,
        )

        valid_attack_types = [
            str(t).strip().lower()
            for t in self.aqa_valid_attack_types
            if str(t).strip()
        ]
        if not valid_attack_types:
            valid_attack_types = ["general_opposition"]
        object.__setattr__(self, "aqa_valid_attack_types", valid_attack_types)

        default_attack = str(self.aqa_default_attack_type or "").strip().lower()
        if default_attack not in valid_attack_types:
            default_attack = "general_opposition"
        object.__setattr__(self, "aqa_default_attack_type", default_attack)

        object.__setattr__(self, "_groq_api_keys", keys)
        return self

    @property
    def groq_api_keys(self) -> list[str]:
        """Get all available Groq API keys (non-empty, dynamically discovered)."""
        return list(self._groq_api_keys)

    @property
    def model_alias_map(self) -> dict[str, str]:
        """Centralized alias -> provider model mapping."""
        return dict(self.MODEL_ALIAS_MAP)

    @property
    def available_model_aliases(self) -> list[str]:
        """Aliases exposed to frontend model selectors."""
        return list(self.model_alias_map.keys())

    @property
    def pipeline_model_order_aliases(self) -> list[str]:
        """Ordered model aliases used in resilient runtime calls."""
        deduped: list[str] = []
        for alias in self.PIPELINE_MODEL_ORDER_ALIASES:
            if alias not in deduped:
                deduped.append(alias)
        return deduped

    def _resolve_model_alias_order(self, aliases: list[str]) -> list[str]:
        """Resolve alias list to provider model ids, preserving order and uniqueness."""
        models: list[str] = []
        for alias in aliases:
            model = self.resolve_model_name(alias)
            if model and model not in models:
                models.append(model)
        return models

    @property
    def reasoner_model_fallback_order(self) -> list[str]:
        """Provider model ids used by Reasoner resilient fallback."""
        return self._resolve_model_alias_order(self.reasoner_model_fallback_aliases)

    @property
    def counter_model_fallback_order(self) -> list[str]:
        """Provider model ids used by Counter-Reasoner resilient fallback."""
        return self._resolve_model_alias_order(self.counter_model_fallback_aliases)

    @property
    def reasoner_default_model(self) -> str:
        """Default frontend alias for Reasoner model selection."""
        return self.REASONER_DEFAULT_MODEL_ALIAS

    @property
    def counter_default_model(self) -> str:
        """Default frontend alias for Counter-Reasoner model selection."""
        return self.COUNTER_DEFAULT_MODEL_ALIAS

    def resolve_model_name(self, alias_or_model: str | None) -> str:
        """Resolve alias to provider model id; pass-through raw ids unchanged."""
        alias_map = self.model_alias_map
        if not alias_or_model:
            first_alias = self.pipeline_model_order_aliases[0]
            return alias_map.get(first_alias, self.MODEL_ALIAS_MAP[first_alias])
        return alias_map.get(alias_or_model, alias_or_model)

    @property
    def groq_models(self) -> list[str]:
        """Ordered provider model ids for runtime resilience loop."""
        models: list[str] = []
        for alias in self.pipeline_model_order_aliases:
            model = self.resolve_model_name(alias)
            if model and model not in models:
                models.append(model)
        return models

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
        return self.data_dir / "statutes"

    @property
    def precedents_dir(self) -> Path:
        """Get precedents directory."""
        return self.data_dir / "precedents"

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
            "pipeline_models": self.groq_models,
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
    print(f"Pipeline Models: {settings.groq_models}")
    print(f"Embedding Model: {settings.embedding_model}")
    print()
    validation = settings.validate_config()
    print(f"Valid: {validation['valid']}")
    if validation["issues"]:
        print(f"Issues: {validation['issues']}")
