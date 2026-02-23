"""Services module for LexCausa."""

from .claim_classifier import ClaimClassifier
from .legal_search import LegalSearchPipeline, SearchResult

__all__ = ["ClaimClassifier", "LegalSearchPipeline", "SearchResult"]
