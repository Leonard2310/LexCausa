"""Services module for LexCausa."""

from .claim_classifier import ClaimClassifier
from .legal_search import LegalSearchPipeline, SearchResult
from .stance_classifier import StanceClassifier

__all__ = ["ClaimClassifier", "LegalSearchPipeline", "SearchResult", "StanceClassifier"]
