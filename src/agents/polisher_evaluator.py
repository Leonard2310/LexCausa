"""
LexCausa Polisher-Evaluator Agent.

The Polisher-Evaluator is responsible for:
1. Receiving arguments from Reasoner and Counter-Reasoner
2. Evaluating the dialectical exchange
3. Checking consistency of reasoning chains against the knowledge base
4. Determining which arguments prevail
5. Polishing the final output for presentation

This agent acts as a judge/evaluator of the argumentation.
"""

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import textstat
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

from config import settings

from .aspic_formatter import AspicFormatter
from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import get_driver

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402


class MismatchAction(Enum):
    """Action taken when a normative mismatch is detected."""

    NONE = "none"  # No mismatch or not processed
    MATCH = "match"  # Text matches DB
    REPAIRED = "repaired"  # Mismatch repaired with DB text
    DROPPED = "dropped"  # Argument dropped (peripheral)
    REPAIR_FAILED = "repair_failed"  # Repair attempted but failed


class ArgumentStatus(Enum):
    """Status of an argument after evaluation."""

    ACCEPTED = "accepted"  # Argument stands, no successful attacks
    DEFEATED = "defeated"  # Argument was successfully attacked
    DEFENDED = "defended"  # Argument was attacked but the attack was countered
    UNDECIDED = "undecided"  # Cannot determine status


@dataclass
class CitationCheck:
    """Result of checking a single citation."""

    citation: str
    found_in_kb: bool
    source_type: str  # "statute" or "precedent"
    details: str = ""
    # Text verification fields
    text_verified: bool = False  # True if text comparison was performed
    text_match: bool = False  # True if cited text matches DB text
    text_similarity: float = 0.0  # Similarity score (0-1)
    cited_text: str = ""  # Text extracted from reasoning chain
    db_text_preview: str = ""  # Preview of text from DB
    # Mismatch handling fields
    mismatch_action: str = "none"  # MismatchAction value
    is_core: bool = False  # True if article is core to the argument
    llm_mismatch_confirmed: bool = False  # True if LLM confirmed logical mismatch
    llm_validated: bool = (
        False  # True if initial similarity mismatch was rescued by LLM equivalence check
    )
    repaired_text: str = ""  # Repaired text using DB constraint
    repair_success: bool = False  # True if repair was successful

    def to_dict(self) -> dict:
        return {
            "citation": self.citation,
            "found_in_kb": self.found_in_kb,
            "source_type": self.source_type,
            "details": self.details,
            "text_verified": self.text_verified,
            "text_match": self.text_match,
            "text_similarity": self.text_similarity,
            "cited_text": self.cited_text,
            "db_text_preview": self.db_text_preview,
            "mismatch_action": self.mismatch_action,
            "is_core": self.is_core,
            "llm_mismatch_confirmed": self.llm_mismatch_confirmed,
            "llm_validated": self.llm_validated,
            "repaired_text": self.repaired_text,
            "repair_success": self.repair_success,
        }


@dataclass
class ConsistencyReport:
    """Report on the consistency of a reasoning chain with the knowledge base."""

    agent: str  # "reasoner" or "counter_reasoner"
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    text_matches: int = 0  # Citations where text matches DB
    text_mismatches: int = 0  # Citations where text doesn't match DB
    repaired_citations: int = 0  # Citations that were repaired
    dropped_citations: int = 0  # Citations that were dropped
    citation_checks: list[CitationCheck] = field(default_factory=list)
    consistency_score: float = 0.0  # 0-1 score
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "text_matches": self.text_matches,
            "text_mismatches": self.text_mismatches,
            "repaired_citations": self.repaired_citations,
            "dropped_citations": self.dropped_citations,
            "citation_checks": [c.to_dict() for c in self.citation_checks],
            "consistency_score": self.consistency_score,
            "issues": self.issues,
        }


@dataclass
class EvaluatedArgument:
    """An argument with its evaluation status."""

    argument_id: str
    content: dict
    status: ArgumentStatus
    attacks_received: list[str] = field(default_factory=list)
    defenses: list[str] = field(default_factory=list)
    score: float = 0.0  # Confidence in the status

    def to_dict(self) -> dict:
        return {
            "id": self.argument_id,
            "content": self.content,
            "status": self.status.value,
            "attacks_received": self.attacks_received,
            "defenses": self.defenses,
            "score": self.score,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    claim: str
    winning_side: str  # "support", "counter", or "undecided"
    confidence: float  # 0-1 confidence in the evaluation
    evaluated_arguments: list[EvaluatedArgument] = field(default_factory=list)
    consistency_report: dict = field(default_factory=dict)  # reasoner + counter reports
    aqa_report: dict = field(default_factory=dict)  # AQA scoring + verdict
    summary: str = ""
    polished_response: str = ""
    dialectical_tree: dict = field(default_factory=dict)
    # Repaired reasoning chains (reworked by LLM after mismatch handling)
    repaired_reasoner_chain: str = ""
    repaired_counter_chain: str = ""
    repaired_reasoner_aspic_ir: dict = field(default_factory=dict)
    repaired_counter_aspic_ir: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "winning_side": self.winning_side,
            "confidence": self.confidence,
            "evaluated_arguments": [ea.to_dict() for ea in self.evaluated_arguments],
            "consistency_report": self.consistency_report,
            "aqa_report": self.aqa_report,
            "summary": self.summary,
            "polished_response": self.polished_response,
            "dialectical_tree": self.dialectical_tree,
            "repaired_reasoner_chain": self.repaired_reasoner_chain,
            "repaired_counter_chain": self.repaired_counter_chain,
            "repaired_reasoner_aspic_ir": self.repaired_reasoner_aspic_ir,
            "repaired_counter_aspic_ir": self.repaired_counter_aspic_ir,
        }


class PolisherEvaluator(BaseAgent):
    """
    Legal Polisher-Evaluator Agent.

    Evaluates the dialectical exchange between Reasoner and Counter-Reasoner,
    determines which arguments prevail, and produces a polished final output.

    Uses grounded semantics to compute argument acceptability:
    - An argument is acceptable if all its attackers are defeated
    - An argument is defeated if at least one of its attackers is acceptable
    """

    # Regex patterns for citation extraction
    STATUTE_PATTERN = re.compile(
        r"(?:art(?:icolo)?\.?\s*)(\d{1,4})\s*(c\.?[cp]\.?|cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?))?",
        re.IGNORECASE,
    )
    PRECEDENT_PATTERN = re.compile(
        r"(?:Cass(?:azione)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?)?\.?\s*(?:n\.?\s*)?(\d+)(?:/(\d{4}))?)|"
        r"(?:sentenza\s+n\.?\s*(\d+)(?:/(\d{4}))?)",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Polisher-Evaluator agent."""
        super().__init__(config)
        self._log("Polisher-Evaluator initialized")
        self._embedder = None
        self._nli = None
        self._arg_quality = None
        self._tfidf_vectorizer = None
        self._tfidf_cache: dict[str, Any] = {}
        self._embed_cache: dict[str, Any] = {}
        self._statute_meta_cache: dict[tuple[str, str], dict] = {}
        self._aqa_enabled = settings.aqa_enabled
        self._aqa_alpha = settings.aqa_alpha
        self._aqa_beta = settings.aqa_beta
        self._aqa_gamma = settings.aqa_gamma
        self._aqa_attack_top_k = settings.aqa_attack_top_k
        self._aqa_verdict_pos = settings.aqa_verdict_pos_threshold
        self._aqa_verdict_neg = settings.aqa_verdict_neg_threshold
        self._aqa_embedding_model = settings.aqa_embedding_model
        self._aqa_nli_model = settings.aqa_nli_model
        self._aqa_arg_quality_model = settings.aqa_argument_quality_model
        self._aqa_arg_quality_use_model = settings.aqa_argument_quality_use_model
        self._aqa_tfidf_max_features = settings.aqa_tfidf_max_features
        self._aqa_normsupport_max_citations = settings.aqa_normsupport_max_citations
        self._aqa_normsupport_citation_weight = settings.aqa_normsupport_citation_weight
        self._aqa_normsupport_retrieved_weight = (
            settings.aqa_normsupport_retrieved_weight
        )
        self._aqa_normsupport_retrieved_agg = settings.aqa_normsupport_retrieved_agg
        self._aqa_severity_map_penale = settings.aqa_severity_map_penale
        self._aqa_severity_map_civile = settings.aqa_severity_map_civile

    def run(
        self,
        claim: str,
        domain: str = "CIVILE",
        reasoner_output: dict | None = None,
        counter_reasoner_output: dict | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate the dialectical exchange and produce final output.

        Args:
            claim: The original legal claim.
            domain: Legal domain from router ("CIVILE", "PENALE", or "ENTRAMBI").
            reasoner_output: Output from the Reasoner agent.
            counter_reasoner_output: Output from the Counter-Reasoner agent.
            **kwargs: Additional arguments.

        Returns:
            EvaluationResult with final assessment and polished response.
        """
        self._log("Starting consistency evaluation...")
        self._log(f"🔍 Domain: {domain}")

        reasoner_output = reasoner_output or {}
        counter_reasoner_output = counter_reasoner_output or {}

        # Check consistency of Reasoner chain
        reasoner_chain = reasoner_output.get("reasoning_chain", [])
        reasoner_raw = reasoner_output.get("raw_response", "")
        reasoner_aspic_ir = reasoner_output.get("aspic_ir", {})
        reasoner_report = self._check_consistency(
            agent="reasoner",
            reasoning_chain=reasoner_chain,
            raw_response=reasoner_raw,
            domain=domain,
            aspic_ir=reasoner_aspic_ir,
        )

        # Check consistency of Counter-Reasoner chain
        counter_chain = counter_reasoner_output.get("reasoning_chain", [])
        counter_raw = counter_reasoner_output.get("raw_response", "")
        counter_aspic_ir = counter_reasoner_output.get("aspic_ir", {})
        counter_report = self._check_consistency(
            agent="counter_reasoner",
            reasoning_chain=counter_chain,
            raw_response=counter_raw,
            domain=domain,
            aspic_ir=counter_aspic_ir,
        )

        self._log(
            f"📊 Reasoner consistency: {reasoner_report.consistency_score:.2f} "
            f"({reasoner_report.valid_citations}/{reasoner_report.total_citations} valid)"
        )
        self._log(
            f"📊 Counter consistency: {counter_report.consistency_score:.2f} "
            f"({counter_report.valid_citations}/{counter_report.total_citations} valid)"
        )

        # ------------------------------------------------------------------
        # Repair Reasoning Chains (if needed)
        # ------------------------------------------------------------------
        self._log("🔧 Checking if reasoning chains need repair...")

        # Regenerate Reasoner chain if needed
        repaired_reasoner_chain = ""
        if (
            reasoner_report.repaired_citations > 0
            or reasoner_report.dropped_citations > 0
        ):
            repaired_reasoner_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=reasoner_raw,
                citation_checks=reasoner_report.citation_checks,
                agent_name="reasoner",
            )
        else:
            repaired_reasoner_chain = reasoner_raw  # No changes needed

        # Regenerate Counter-Reasoner chain if needed
        repaired_counter_chain = ""
        if (
            counter_report.repaired_citations > 0
            or counter_report.dropped_citations > 0
        ):
            repaired_counter_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=counter_raw,
                citation_checks=counter_report.citation_checks,
                agent_name="counter_reasoner",
            )
        else:
            repaired_counter_chain = counter_raw  # No changes needed

        # Repair ASPIC IR structures by rebuilding from repaired chain
        repaired_reasoner_aspic = self._repair_aspic_ir(
            aspic_ir=reasoner_aspic_ir,
            citation_checks=reasoner_report.citation_checks,
            repaired_chain_text=repaired_reasoner_chain,
            claim=claim,
            role="support",
            statutes=reasoner_output.get("relevant_statutes", []),
            precedents=reasoner_output.get("relevant_precedents", []),
            metadata=reasoner_aspic_ir.get("metadata", {}),
        )
        repaired_counter_aspic = self._repair_aspic_ir(
            aspic_ir=counter_aspic_ir,
            citation_checks=counter_report.citation_checks,
            repaired_chain_text=repaired_counter_chain,
            claim=claim,
            role="counter",
            statutes=counter_reasoner_output.get("relevant_statutes", []),
            precedents=counter_reasoner_output.get("relevant_precedents", []),
            metadata=counter_aspic_ir.get("metadata", {}),
        )

        # Build dialectical tree (with repaired IRs)
        reasoner_ir = reasoner_output.get("aspic_ir")
        counter_ir = counter_reasoner_output.get("aspic_ir")
        dialectical_tree = {}
        if reasoner_ir or counter_ir:
            dialectical_tree = {
                "schema": "aspic_ir_bundle_v1",
                "reasoner": reasoner_ir,
                "counter": counter_ir,
                "repaired_reasoner": repaired_reasoner_aspic,
                "repaired_counter": repaired_counter_aspic,
            }

        # Generate summary
        summary = self._generate_consistency_summary(reasoner_report, counter_report)

        # ------------------------------------------------------------------
        # AQA Phase: ALWAYS use repaired ASPIC IR when available
        # ------------------------------------------------------------------
        # Determine which IR to use: prefer repaired, fallback to original
        aqa_reasoner_ir = (
            repaired_reasoner_aspic
            if repaired_reasoner_aspic
            else (reasoner_aspic_ir or {})
        )
        aqa_counter_ir = (
            repaired_counter_aspic
            if repaired_counter_aspic
            else (counter_aspic_ir or {})
        )

        # Log which IR version is being used
        if repaired_reasoner_aspic:
            r_meta = repaired_reasoner_aspic.get("_repair_metadata", {})
            self._log(
                f"🔧 AQA using REPAIRED Reasoner IR "
                f"(repaired={r_meta.get('total_repaired', 0)}, dropped={r_meta.get('total_dropped', 0)})"
            )
        else:
            self._log("📋 AQA using ORIGINAL Reasoner IR (no repairs needed)")

        if repaired_counter_aspic:
            c_meta = repaired_counter_aspic.get("_repair_metadata", {})
            self._log(
                f"🔧 AQA using REPAIRED Counter IR "
                f"(repaired={c_meta.get('total_repaired', 0)}, dropped={c_meta.get('total_dropped', 0)})"
            )
        else:
            self._log("📋 AQA using ORIGINAL Counter IR (no repairs needed)")

        aqa_report = self._run_aqa_phase(
            reasoner_ir=aqa_reasoner_ir,
            counter_ir=aqa_counter_ir,
            domain=domain,
        )

        self._log("✅ Evaluation complete")

        return EvaluationResult(
            claim=claim,
            winning_side="",
            confidence=0.0,
            consistency_report={
                "reasoner": reasoner_report.to_dict(),
                "counter_reasoner": counter_report.to_dict(),
            },
            aqa_report=aqa_report,
            summary=summary,
            polished_response="",
            dialectical_tree=dialectical_tree,
            repaired_reasoner_chain=repaired_reasoner_chain,
            repaired_counter_chain=repaired_counter_chain,
            repaired_reasoner_aspic_ir=repaired_reasoner_aspic,
            repaired_counter_aspic_ir=repaired_counter_aspic,
        )

    # ------------------------------------------------------------------
    # Neo4j Verification
    # ------------------------------------------------------------------
    def _verify_statute_in_neo4j(
        self, article_num: str, domain: str
    ) -> tuple[bool, str]:
        """
        Verify if an article exists in Neo4j and return its text.

        Args:
            article_num: The article number (e.g., "1223")
            domain: Legal domain ("CIVILE", "PENALE", or "ENTRAMBI")

        Returns:
            Tuple of (exists: bool, text: str). Text is empty if not found.
        """
        driver = get_driver()

        # Determine codice based on domain
        if domain == "CIVILE":
            # Per CIVILE, l'articolo è salvato come "art1223"
            articolo_normalized = f"art{article_num}"
            codice = "codice_civile"
        elif domain == "PENALE":
            # Per PENALE, l'articolo è salvato come "1223" (senza prefisso)
            articolo_normalized = article_num
            codice = "codice_penale"
        else:
            # ENTRAMBI: cerca in entrambi i codici
            found, text = self._verify_statute_in_neo4j(article_num, "CIVILE")
            if found:
                return found, text
            return self._verify_statute_in_neo4j(article_num, "PENALE")

        query = """
            MATCH (s:Statute)
            WHERE s.articolo = $articolo AND s.source = $codice
            RETURN s.articolo AS articolo, s.testo AS testo, s.titolo AS titolo
            LIMIT 1
        """

        try:
            with driver.session() as session:
                result = session.run(
                    query,
                    parameters={"articolo": articolo_normalized, "codice": codice},
                )
                record = result.single()
                if record:
                    testo = record.get("testo", "") or ""
                    titolo = record.get("titolo", "") or ""
                    if testo:
                        self._log(
                            f"      🗄️ Neo4j: Art. {article_num} found - '{titolo[:50]}...'"
                        )
                        self._log(f"      📄 DB text preview: '{testo[:100]}...'")
                    else:
                        self._log(
                            f"      🗄️ Neo4j: Art. {article_num} found but NO TEXT in DB"
                        )
                    return True, testo
                return False, ""
        except Exception as e:
            self._log(f"⚠️ Neo4j query failed: {e}", "warning")
            return False, ""

    def _extract_cited_text_for_article(
        self, full_text: str, article_num: str, aspic_ir: dict | None = None
    ) -> str:
        """
        Extract the text cited in the reasoning chain for a specific article.

        Searches for patterns like:
        - "L'Art. 1223 c.c. stabilisce che [testo fino al punto]"
        - "Art. 1223 c.c. - [testo fino al punto]"
        - "Secondo l'Art. 1223 c.c., [testo]"

        Args:
            full_text: The full reasoning chain text (raw_response)
            article_num: The article number to find
            aspic_ir: Optional ASPIC IR (not used in this version)

        Returns:
            Extracted text associated with the article, or empty string if not found.
        """
        # Pattern 0: block with Norma + Testo lines
        block_pattern = (
            rf"Norma.*?Art(?:icolo)?\.?\s*{article_num}.*?\n"
            rf".*?Testo\s*:\s*\"([^\"]{{20,}})\""
        )
        match = re.search(block_pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            if len(extracted) >= 20:
                self._log("      🎯 Found 'Norma + Testo' block pattern")
                return extracted

        # Pattern 1: "L'Art. 1223 c.c. stabilisce/prevede/limita che..."
        # Captures the verb + "che" + text until period
        pattern1 = rf"[Ll][''']?Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?\s+(?:stabilisce|prevede|limita|dispone|sancisce|afferma)\s+(?:che\s+)?(.+?)(?:\.|$)"
        match = re.search(pattern1, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            if len(extracted) >= 20:
                self._log(
                    "      🎯 Found 'Art. "
                    + str(article_num)
                    + " stabilisce/prevede...' pattern"
                )
                return extracted

        # Pattern 2: "Art. 1223 c.c. - [testo]"
        pattern2 = rf"Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?\s*[\-:]\s*(.+?)(?:\.|$)"
        match = re.search(pattern2, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            # Avoid if it contains other article references (means it's a list)
            if not re.search(r"Art(?:icolo)?\.?\s*\d{3,4}", extracted, re.IGNORECASE):
                if len(extracted) >= 20:
                    self._log(
                        "      🎯 Found 'Art. " + str(article_num) + " - ...' pattern"
                    )
                    return extracted

        # Pattern 3: "Secondo l'Art. 1223 c.c., [testo]"
        pattern3 = rf"[Ss]econdo\s+[Ll][''']?Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?,?\s+(.+?)(?:\.|$)"
        match = re.search(pattern3, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            if len(extracted) >= 20:
                self._log(
                    "      🎯 Found 'Secondo l'Art. "
                    + str(article_num)
                    + "...' pattern"
                )
                return extracted

        # Pattern 4: "(Art. 1223 c.c.)" in parentheses after text - capture text before
        pattern4 = rf"([^.]+?)\s*\(Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?\)"
        match = re.search(pattern4, full_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Take last sentence before the article reference
            sentences = extracted.split(".")
            if sentences:
                last_sentence = sentences[-1].strip()
                if len(last_sentence) >= 20:
                    self._log(
                        "      🎯 Found text before '(Art. "
                        + str(article_num)
                        + ")' pattern"
                    )
                    return last_sentence

        self._log("      ⚠️ No text found for Art. " + str(article_num))
        return ""

    def _extract_text_from_aspic_ir(self, aspic_ir: dict, article_num: str) -> str:
        """
        Extract the text associated with an article from ASPIC IR structure.

        Searches in:
        1. arguments[].premises[] where type="norm" and citations include the article
        2. reasoning_chain[] steps that cite the article
        3. sources.statutes for the article title

        Args:
            aspic_ir: ASPIC IR structured output
            article_num: The article number to find

        Returns:
            The text associated with the article from the IR structure.
        """
        if not aspic_ir:
            self._log("      ⚠️ ASPIC IR is empty or None")
            return ""

        self._log("      🔍 Searching ASPIC IR for Art. {article_num}")

        # Helper to check if an article is in citations
        def article_in_citations(citations: dict, art_num: str) -> bool:
            for statute in citations.get("statutes", []):
                if str(statute.get("articolo", "")).strip() == art_num:
                    return True
            return False

        # Helper to extract text specific to an article from a larger text block
        def extract_article_specific_text(full_text: str, art_num: str) -> str:
            """Extract the portion of text that refers specifically to this article."""
            clean_text = full_text.replace("**", "")
            # Pattern to find text associated with specific article
            patterns = [
                # Testo: "..."
                r"testo\s*:\s*\"([^\"]{20,})\"",
                # Testo: ... (no quotes)
                r"testo\s*:\s*([^.]+(?:\.[^.]+)?)",
                # Art. 1223: text or Art. 1223 - text
                rf"art(?:icolo)?\.?\s*{art_num}[^:.\n]*[:.-]\s*([^.]+(?:\.[^.]+)?)",
                # Art. 1223 c.c. che prevede/stabilisce che...
                rf"art(?:icolo)?\.?\s*{art_num}[^,]*(?:che\s+)?(?:prevede|stabilisce|dispone|sancisce)\s+(?:che\s+)?([^.]+(?:\.[^.]+)?)",
                # art. 1223 (testo tra parentesi)
                rf"art(?:icolo)?\.?\s*{art_num}[^(]*\(([^)]+)\)",
            ]
            for pattern in patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) >= 20:
                        return extracted
            return ""

        # Search in arguments -> premises with type="norm"
        for arg in aspic_ir.get("arguments", []):
            for premise in arg.get("premises", []):
                if premise.get("type") != "norm":
                    continue
                citations = premise.get("citations", {})
                if article_in_citations(citations, article_num):
                    full_text = premise.get("text", "").strip()
                    if full_text:
                        # Count how many articles are cited in this block
                        num_statutes = len(citations.get("statutes", []))
                        self._log(
                            "      📋 Found norm block with {} statute(s)".format(
                                num_statutes
                            )
                        )

                        if num_statutes == 1:
                            specific_text = extract_article_specific_text(
                                full_text, article_num
                            )
                            if specific_text:
                                self._log(
                                    "      ✅ Extracted testo from single-article block"
                                )
                                return specific_text
                            # Only this article, return full text
                            self._log("      ✅ Single article block - using full text")
                            return full_text
                        else:
                            # Multiple articles, try to extract specific portion
                            specific_text = extract_article_specific_text(
                                full_text, article_num
                            )
                            if specific_text:
                                self._log(
                                    "      ✅ Extracted specific text for Art. {}".format(
                                        article_num
                                    )
                                )
                                return specific_text
                            # If can't extract specific, still return full text with warning
                            self._log(
                                "      ⚠️ Multi-article block, returning full text"
                            )
                            return full_text

        # Search in reasoning_chain steps - look for step that mentions ONLY this article
        for step in aspic_ir.get("reasoning_chain", []):
            citations = step.get("citations", {})
            statutes = citations.get("statutes", [])

            # Check if this step cites the article
            if any(str(s.get("articolo", "")).strip() == article_num for s in statutes):
                text = step.get("text", "").strip()
                if text and len(text) >= 15:
                    # If only this article is cited, return full step text
                    if len(statutes) == 1:
                        self._log("      ✅ Found in reasoning chain (single article)")
                        return text
                    else:
                        # Try to extract specific portion
                        specific_text = extract_article_specific_text(text, article_num)
                        if specific_text:
                            self._log("      ✅ Extracted from reasoning chain step")
                            return specific_text

        # Fallback: get title from sources
        for statute in aspic_ir.get("sources", {}).get("statutes", []):
            if str(statute.get("articolo", "")).strip() == article_num:
                title = statute.get("title", "").strip()
                if title:
                    self._log(
                        "      📚 Using title from sources: " + title[:50] + "..."
                    )
                    return f"[Titolo] {title}"

        self._log("      ❌ No text found in ASPIC IR for Art. " + str(article_num))
        return ""

    def _compute_text_similarity(self, cited_text: str, db_text: str) -> float:
        """
        Compute similarity between cited text and database text.

        Uses a simple approach: check if key phrases from cited text appear in db_text.

        Args:
            cited_text: Text extracted from reasoning chain
            db_text: Text from database

        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not cited_text or not db_text:
            return 0.0

        # Normalize both texts
        cited_lower = cited_text.lower().strip()
        db_lower = db_text.lower().strip()

        # Check direct substring match
        if cited_lower in db_lower:
            return 1.0

        # Extract meaningful words (>3 chars)
        cited_words = set(w for w in re.findall(r"\b\w{4,}\b", cited_lower))
        db_words = set(w for w in re.findall(r"\b\w{4,}\b", db_lower))

        if not cited_words:
            return 0.0

        # Calculate Jaccard-like similarity
        common_words = cited_words & db_words
        similarity = len(common_words) / len(cited_words)

        return min(similarity, 1.0)

    # ------------------------------------------------------------------
    # Mismatch Handling (LLM Verification, Repair, Drop)
    # ------------------------------------------------------------------
    def _verify_mismatch_with_llm(
        self,
        article_num: str,
        cited_text: str,
        db_text: str,
    ) -> bool:
        """
        Use LLM to verify if the cited text is logically different from DB text.

        Args:
            article_num: Article number for context
            cited_text: Text extracted from reasoning chain
            db_text: Official text from database

        Returns:
            True if LLM confirms the texts are logically different (mismatch).
        """
        try:
            llm = get_chat_groq(temperature=0, max_tokens=2048)

            system_prompt = """You are an expert in Italian law. Your task is to determine whether two normative texts are LOGICALLY EQUIVALENT or DIFFERENT.

Two texts are EQUIVALENT if:
- They express the same legal concept, even with different wording
- One is a faithful paraphrase of the other
- They don't add or omit substantial normative elements

Two texts are DIFFERENT if:
- They add requirements not present in the original
- They omit essential elements
- They change the legal meaning
- They introduce concepts not contained in the original

Respond ONLY with one of these words: EQUIVALENTI or DIVERSI (in Italian)"""

            user_prompt = f"""Article {article_num}

CITED TEXT (from reasoning):
"{cited_text}"

OFFICIAL TEXT (from database):
"{db_text}"

Are the two texts EQUIVALENTI or DIVERSI? (Answer in Italian)"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            answer = response.content.strip().upper()

            is_different = "DIVERSI" in answer
            self._log(
                f"      🤖 LLM mismatch verification: {'DIVERSI ⚠️' if is_different else 'EQUIVALENTI ✅'}"
            )
            return is_different

        except Exception as e:
            self._log(f"      ⚠️ LLM verification failed: {e}", "warning")
            # In case of error, assume mismatch to be conservative
            return True

    def _check_pertinence_with_llm(
        self,
        article_num: str,
        cited_text: str,
        full_text: str,
    ) -> bool:
        """
        Use LLM to check if a peripheral norm is still logically pertinent
        to the reasoning chain, even if not core.

        A norm is PERTINENT if it contributes meaningfully to the legal
        reasoning — e.g., it provides context, reinforces a conclusion,
        or integrates the normative framework. It should be kept and repaired.

        A norm is NOT PERTINENT if it is tangential, redundant, or does not
        add value to the argument. It can safely be dropped.

        Args:
            article_num: Article number
            cited_text: Text cited in the reasoning chain for this article
            full_text: Full reasoning chain text

        Returns:
            True if the norm is pertinent (should be repaired),
            False if not pertinent (can be dropped).
        """
        try:
            llm = get_chat_groq(temperature=0, max_tokens=2048)

            system_prompt = """You are an expert in Italian law. Your task is to assess whether a legal norm cited in a reasoning chain is PERTINENT or NOT_PERTINENT to the argumentative logic.

A norm is PERTINENT if:
- It contributes meaningfully to the legal reasoning (e.g., establishes liability, defines scope, sets conditions)
- It reinforces or integrates the legal conclusion
- It provides necessary normative context for the argument's thesis
- It completes the regulatory framework by filling a logical gap
- It is causally or logically connected to other steps in the chain

A norm is NOT_PERTINENT if:
- It is tangential to the main line of reasoning
- It is redundant with respect to other norms already cited that cover the same concept
- It adds no argumentative value to the reasoning chain
- It has no logical or causal link to the conclusion

Respond ONLY with one of these words: PERTINENT or NOT_PERTINENT"""

            user_prompt = f"""COMPLETE REASONING CHAIN:
\"\"\"
{full_text[:3000]}
\"\"\"

NORM TO EVALUATE: Art. {article_num}
CITED TEXT: "{cited_text}"

Is this norm PERTINENT or NOT_PERTINENT to the argumentative logic of the reasoning chain?"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            answer = response.content.strip().upper()

            is_pertinent = (
                "NOT_PERTINENT" not in answer and "NOT PERTINENT" not in answer
            )
            self._log(
                f"      🤖 LLM pertinence check Art. {article_num}: "
                f"{'PERTINENTE ✅' if is_pertinent else 'NON PERTINENTE 🗑️'}"
            )
            return is_pertinent

        except Exception as e:
            self._log(f"      ⚠️ LLM pertinence check failed: {e}", "warning")
            # In case of error, assume pertinent to be conservative (repair > drop)
            return True

    def _is_article_core(
        self,
        article_num: str,
        aspic_ir: dict | None,
        full_text: str,
    ) -> bool:
        """
        Determine if an article is CORE (essential) or PERIPHERAL to the argument.

        An article is CORE if:
        - It's the only normative support for a conclusion
        - It appears in the main norm/rule of an argument
        - It's cited multiple times in the reasoning chain
        - It's in the conclusion's citations

        Args:
            article_num: Article number to check
            aspic_ir: ASPIC IR structure
            full_text: Full text of reasoning chain

        Returns:
            True if the article is core, False if peripheral.
        """
        core_indicators = 0

        # Check ASPIC IR for core indicators
        if aspic_ir:
            for arg in aspic_ir.get("arguments", []):
                # Check premises with type="norm"
                norm_premises = [
                    p for p in arg.get("premises", []) if p.get("type") == "norm"
                ]

                for premise in norm_premises:
                    citations = premise.get("citations", {})
                    statutes = citations.get("statutes", [])

                    # Article is in a norm premise
                    article_cited = any(
                        str(s.get("articolo", "")).strip() == article_num
                        for s in statutes
                    )

                    if article_cited:
                        # If it's the ONLY statute in a norm premise → definitely core
                        if len(statutes) == 1:
                            self._log(
                                f"      🎯 Art. {article_num} is CORE: only statute in norm premise"
                            )
                            return True
                        core_indicators += 1

                # Check conclusion citations
                conclusion = arg.get("conclusion", {})
                if conclusion:
                    conclusion_citations = conclusion.get("citations", {})
                    for s in conclusion_citations.get("statutes", []):
                        if str(s.get("articolo", "")).strip() == article_num:
                            core_indicators += 2  # In conclusion = more important
                            self._log(
                                f"      🎯 Art. {article_num} cited in conclusion"
                            )

        # Count occurrences in full text
        pattern = rf"art(?:icolo)?\.?\s*{article_num}"
        occurrences = len(re.findall(pattern, full_text, re.IGNORECASE))
        if occurrences >= 3:
            core_indicators += 1
            self._log(
                f"      📊 Art. {article_num} appears {occurrences} times in text"
            )

        # Threshold: if 2+ core indicators, consider it core
        is_core = core_indicators >= 2
        self._log(
            f"      {'🎯 CORE' if is_core else '📎 PERIPHERAL'}: Art. {article_num} (indicators: {core_indicators})"
        )
        return is_core

    def _repair_with_db_constraint(
        self,
        article_num: str,
        db_text: str,
        original_context: str,
    ) -> tuple[bool, str]:
        """
        Attempt to repair the normative citation using the official DB text.

        The LLM must rewrite the norm/causal link using ONLY the DB text,
        and include a verbatim quote from the DB.

        Args:
            article_num: Article number
            db_text: Official text from database
            original_context: Original context where the article was used

        Returns:
            Tuple of (success: bool, repaired_text: str).
            Success is True only if the repaired text contains a verbatim DB quote.
        """
        try:
            llm = get_chat_groq(temperature=0, max_tokens=2048)

            system_prompt = """You are an expert in Italian law. You must rewrite a normative passage using EXCLUSIVELY the official text of the provided article.

MANDATORY RULES:
1. You must include a VERBATIM QUOTE (exact copy) of at least 15 consecutive words from the official text
2. The quote must be enclosed in «» (guillemets/angle quotes)
3. You cannot add concepts not present in the official text
4. The result must be legally correct and coherent

OUTPUT: Write ONLY the rewritten text in Italian, without explanations."""

            user_prompt = f"""ARTICLE: Art. {article_num}

OFFICIAL TEXT TO USE:
"{db_text}"

ORIGINAL CONTEXT (to correct):
"{original_context[:500]}"

Rewrite the normative passage in Italian using only the official text, including a verbatim quote enclosed in «»."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            repaired_text = response.content.strip()

            # Validate: check if there's a verbatim quote from DB
            # Extract text between «»
            quote_match = re.search(r"«([^»]+)»", repaired_text)
            if quote_match:
                quote = quote_match.group(1).lower().strip()
                db_lower = db_text.lower()

                # Check if quote is a substring of DB text
                if quote in db_lower and len(quote) >= 15:
                    self._log(
                        f"      ✅ Repair SUCCESS: verbatim quote found ({len(quote)} chars)"
                    )
                    return True, repaired_text
                else:
                    self._log(
                        "      ❌ Repair FAILED: quote not found in DB or too short"
                    )
                    return False, ""
            else:
                self._log("      ❌ Repair FAILED: no «» quote in response")
                return False, ""

        except Exception as e:
            self._log(f"      ⚠️ Repair failed with error: {e}", "warning")
            return False, ""

    def _handle_norm_mismatch(
        self,
        check: CitationCheck,
        article_num: str,
        cited_text: str,
        db_text: str,
        aspic_ir: dict | None,
        full_text: str,
        report: ConsistencyReport,
    ) -> None:
        """
        Handle a normative mismatch by verifying with LLM and taking appropriate action.

        Flow:
        1. Verify mismatch with LLM
        2. If confirmed, classify as core/peripheral
        3. If core: attempt repair with DB constraint
        4. If peripheral: check pertinence with LLM
           4a. If pertinent: attempt repair (keep useful norms)
           4b. If not pertinent: drop the citation
        5. If repair fails (core or peripheral): drop the citation

        Args:
            check: CitationCheck object to update
            article_num: Article number
            cited_text: Cited text from reasoning
            db_text: Official DB text
            aspic_ir: ASPIC IR structure
            full_text: Full reasoning text
            report: ConsistencyReport to update
        """
        self._log(f"      🔧 Handling mismatch for Art. {article_num}...")

        # Step 1: Verify mismatch with LLM
        llm_confirmed = self._verify_mismatch_with_llm(article_num, cited_text, db_text)
        check.llm_mismatch_confirmed = llm_confirmed

        if not llm_confirmed:
            # LLM says they're equivalent → treat as match despite low similarity
            self._log("      ✅ LLM says texts are equivalent - treating as match")
            check.mismatch_action = MismatchAction.MATCH.value
            check.text_match = True
            check.llm_validated = True  # Mark as LLM-rescued
            report.text_matches += 1
            report.text_mismatches -= 1  # Correct the earlier increment
            return

        # Step 2: Classify as core or peripheral
        is_core = self._is_article_core(article_num, aspic_ir, full_text)
        check.is_core = is_core

        if is_core:
            # Step 3: Attempt repair
            self._log("      🔄 Attempting repair for CORE article...")
            success, repaired_text = self._repair_with_db_constraint(
                article_num, db_text, cited_text
            )

            if success:
                check.mismatch_action = MismatchAction.REPAIRED.value
                check.repaired_text = repaired_text
                check.repair_success = True
                check.details += " [REPAIRED with DB text]"
                report.repaired_citations += 1
                self._log(f"      ✅ Art. {article_num} REPAIRED successfully")
            else:
                check.mismatch_action = MismatchAction.REPAIR_FAILED.value
                check.repair_success = False
                check.details += " [REPAIR FAILED - citation unreliable]"
                report.dropped_citations += 1
                report.issues.append(
                    f"Art. {article_num}: CORE citation repair failed - argument may be invalid"
                )
                self._log(
                    f"      ❌ Art. {article_num} repair FAILED - marked as unreliable"
                )
        else:
            # Peripheral: check if it's still pertinent to the reasoning
            self._log(
                f"      🔍 Art. {article_num} is PERIPHERAL - checking pertinence..."
            )
            is_pertinent = self._check_pertinence_with_llm(
                article_num, cited_text, full_text
            )

            if is_pertinent:
                # Pertinent peripheral norm → attempt repair (keep it)
                self._log(
                    f"      🔄 Art. {article_num} is PERTINENT - attempting repair..."
                )
                success, repaired_text = self._repair_with_db_constraint(
                    article_num, db_text, cited_text
                )

                if success:
                    check.mismatch_action = MismatchAction.REPAIRED.value
                    check.repaired_text = repaired_text
                    check.repair_success = True
                    check.details += " [REPAIRED - pertinent peripheral citation]"
                    report.repaired_citations += 1
                    self._log(
                        f"      ✅ Art. {article_num} REPAIRED (pertinent peripheral)"
                    )
                else:
                    # Repair failed even for pertinent norm → drop
                    check.mismatch_action = MismatchAction.REPAIR_FAILED.value
                    check.repair_success = False
                    check.details += (
                        " [REPAIR FAILED - pertinent peripheral, repair unsuccessful]"
                    )
                    report.dropped_citations += 1
                    report.issues.append(
                        f"Art. {article_num}: pertinent peripheral citation repair failed"
                    )
                    self._log(
                        f"      ❌ Art. {article_num} repair FAILED (pertinent peripheral)"
                    )
            else:
                # Not pertinent → safe to drop
                check.mismatch_action = MismatchAction.DROPPED.value
                check.details += " [DROPPED - non-pertinent peripheral citation]"
                report.dropped_citations += 1
                self._log(
                    f"      🗑️ Art. {article_num} DROPPED (non-pertinent peripheral)"
                )

    # ------------------------------------------------------------------
    # Consistency Checking
    # ------------------------------------------------------------------
    def _check_consistency(
        self,
        agent: str,
        reasoning_chain: list[str],
        raw_response: str,
        domain: str,
        aspic_ir: dict | None = None,
    ) -> ConsistencyReport:
        """
        Check the consistency of a reasoning chain by verifying citations in Neo4j.

        Performs two checks:
        1. Existence check: Does the article exist in Neo4j?
        2. Text verification: Does the cited text match the actual article text?

        Args:
            agent: Name of the agent ("reasoner" or "counter_reasoner")
            reasoning_chain: List of reasoning steps
            raw_response: Raw LLM response text
            domain: Legal domain ("CIVILE", "PENALE", or "ENTRAMBI")
            aspic_ir: ASPIC IR structured output for text extraction

        Returns:
            ConsistencyReport with verification results.
        """
        report = ConsistencyReport(agent=agent)

        # Combine chain and raw response for extraction
        full_text = "\n".join(reasoning_chain) + "\n" + raw_response

        # Extract statute citations from the text
        statute_citations = self._extract_statute_citations(full_text)
        self._log(
            f"📜 [{agent}] Found {len(statute_citations)} statute citations to verify"
        )

        # Track already verified articles to avoid duplicates
        verified_articles: set[str] = set()

        for citation in statute_citations:
            # Extract article number from citation (e.g., "Art. 1223 c.c." -> "1223")
            match = re.search(r"(\d{1,4})", citation)
            if not match:
                continue

            article_num = match.group(1)

            # Skip if already verified
            if article_num in verified_articles:
                self._log(f"   ⏭️ Art. {article_num} already verified, skipping")
                continue
            verified_articles.add(article_num)

            # Verify existence in Neo4j and get text
            found, db_text = self._verify_statute_in_neo4j(article_num, domain)

            # Initialize check object
            check = CitationCheck(
                citation=citation,
                found_in_kb=found,
                source_type="statute",
            )

            if found:
                report.valid_citations += 1
                self._log(f"   ✅ Art. {article_num} -> EXISTS in Neo4j")

                # Extract cited text from the reasoning chain / raw response
                cited_text = self._extract_cited_text_for_article(
                    full_text, article_num, aspic_ir
                )

                if cited_text:
                    self._log(f"      📖 Cited text extracted: '{cited_text[:80]}...'")

                if cited_text and db_text:
                    # Perform text verification
                    similarity = self._compute_text_similarity(cited_text, db_text)
                    text_match = similarity >= 0.8  # Threshold for considering a match

                    check.text_verified = True
                    check.text_match = text_match
                    check.text_similarity = similarity
                    check.cited_text = cited_text
                    check.db_text_preview = db_text

                    if text_match:
                        report.text_matches += 1
                        check.mismatch_action = MismatchAction.MATCH.value
                        check.details = f"Verified in Neo4j ({domain}), text match: {similarity:.0%}"
                        self._log(
                            f"      📝 Text similarity: {similarity:.0%} ✅ MATCH"
                        )
                    else:
                        report.text_mismatches += 1
                        check.details = f"Verified in Neo4j ({domain}), text mismatch: {similarity:.0%}"
                        self._log(
                            f"      📝 Text similarity: {similarity:.0%} ⚠️ MISMATCH"
                        )
                        report.issues.append(
                            f"Art. {article_num}: cited text differs from DB (similarity: {similarity:.0%})"
                        )

                        # Handle mismatch: LLM verification + repair/drop
                        self._handle_norm_mismatch(
                            check=check,
                            article_num=article_num,
                            cited_text=cited_text,
                            db_text=db_text,
                            aspic_ir=aspic_ir,
                            full_text=full_text,
                            report=report,
                        )
                elif not cited_text and db_text:
                    # Article exists in DB but no text was cited in the response.
                    # Treat as mismatch → repair by injecting DB text.
                    self._log(
                        f"      ⚠️ Art. {article_num}: no text cited but DB text available → forcing repair"
                    )
                    check.text_verified = True
                    check.text_match = False
                    check.text_similarity = 0.0
                    check.cited_text = ""
                    check.db_text_preview = db_text
                    check.is_core = (
                        True  # no text at all → treat as core (must be repaired)
                    )
                    check.mismatch_action = MismatchAction.REPAIRED.value
                    check.repaired_text = db_text
                    check.repair_success = True
                    check.details = f"Verified in Neo4j ({domain}), no text cited → repaired with DB text"
                    report.text_mismatches += 1
                    report.repaired_citations += 1
                    report.issues.append(
                        f"Art. {article_num}: no text cited in response, repaired with DB text"
                    )
                else:
                    check.details = f"Verified in Neo4j ({domain}), no text available for verification"
                    self._log("      📝 No cited text and no DB text for verification")
            else:
                report.invalid_citations += 1
                check.details = f"Not found in Neo4j ({domain})"
                report.issues.append(f"Art. {article_num} not found in {domain} codice")
                self._log(f"   ❌ Art. {article_num} -> NOT FOUND in Neo4j")

            report.citation_checks.append(check)
            report.total_citations += 1

        # Calculate consistency score
        # Weighted: 70% existence + 30% text match (among verified texts)
        if report.total_citations > 0:
            existence_score = report.valid_citations / report.total_citations
        else:
            existence_score = 1.0

        total_text_verified = report.text_matches + report.text_mismatches
        if total_text_verified > 0:
            text_score = report.text_matches / total_text_verified
        else:
            text_score = 1.0  # No text verified = no penalty

        report.consistency_score = 0.7 * existence_score + 0.3 * text_score

        return report

    def _extract_statute_citations(self, text: str) -> list[str]:
        """Extract statute citations from text."""
        citations = []
        seen = set()

        for match in self.STATUTE_PATTERN.finditer(text):
            article_num = match.group(1)
            code_part = match.group(2) or ""

            # Determine code
            if code_part:
                code = (
                    "c.c."
                    if "c" in code_part.lower() and "p" not in code_part.lower()
                    else "c.p."
                )
            else:
                code = ""

            citation = f"Art. {article_num} {code}".strip()
            if citation.lower() not in seen:
                seen.add(citation.lower())
                citations.append(citation)

        return citations

    def _extract_precedent_citations(self, text: str) -> list[str]:
        """Extract precedent/case law citations from text."""
        citations = []
        seen = set()

        for match in self.PRECEDENT_PATTERN.finditer(text):
            # Get whichever group matched
            num = match.group(1) or match.group(3) or ""
            year = match.group(2) or match.group(4) or ""

            if num:
                citation = f"n. {num}"
                if year:
                    citation += f"/{year}"
                if citation.lower() not in seen:
                    seen.add(citation.lower())
                    citations.append(citation)

        return citations

    # ------------------------------------------------------------------
    # AQA Scoring
    # ------------------------------------------------------------------
    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"[.!?]\s+", self._normalize_text(text))
        return [p.strip() for p in parts if p.strip()]

    def _get_sentence_transformer(self):
        if self._embedder is not None:
            return self._embedder
        if not self._aqa_embedding_model:
            self._embedder = None
            return None
        if SentenceTransformer is None:
            self._embedder = None
            return None
        try:
            self._embedder = SentenceTransformer(self._aqa_embedding_model)
        except Exception:
            self._embedder = None
        return self._embedder

    def _get_nli_pipeline(self):
        if self._nli is not None:
            return self._nli
        if not self._aqa_nli_model:
            self._nli = None
            return None
        if pipeline is None:
            self._nli = None
            return None
        try:
            self._nli = pipeline(
                "text-classification",
                model=self._aqa_nli_model,
                top_k=None,
            )
        except Exception:
            self._nli = None
        return self._nli

    def _get_tfidf_vectorizer(self):
        if self._tfidf_vectorizer is not None:
            return self._tfidf_vectorizer
        if TfidfVectorizer is None:
            self._tfidf_vectorizer = None
            return None
        try:
            self._tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=None,
                max_features=self._aqa_tfidf_max_features,
            )
        except Exception:
            self._tfidf_vectorizer = None
        return self._tfidf_vectorizer

    def _vector_to_list(self, vec: Any) -> list[float]:
        if hasattr(vec, "toarray"):
            vec = vec.toarray()
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if isinstance(vec, list) and vec and isinstance(vec[0], list):
            return vec[0]
        return vec or []

    def _cosine_sim(self, v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def _embed_text(self, text: str) -> list[float]:
        text = self._normalize_text(text)
        if text in self._embed_cache:
            return self._embed_cache[text]
        embedder = self._get_sentence_transformer()
        if embedder is None:
            return []
        try:
            vec = embedder.encode([text], convert_to_numpy=True)[0]
            result = vec.tolist()
            self._embed_cache[text] = result
            return result
        except Exception:
            return []

    def _tfidf_vector(self, text: str, other: str) -> tuple[list[float], list[float]]:
        vectorizer = self._get_tfidf_vectorizer()
        if vectorizer is None:
            return [], []
        key = f"{text}||{other}"
        if key in self._tfidf_cache:
            return self._tfidf_cache[key]
        try:
            matrix = vectorizer.fit_transform([text, other])
            v1 = self._vector_to_list(matrix[0])
            v2 = self._vector_to_list(matrix[1])
            self._tfidf_cache[key] = (v1, v2)
            return v1, v2
        except Exception:
            return [], []

    def _tfidf_similarity(self, text_a: str, text_b: str) -> float:
        v1, v2 = self._tfidf_vector(text_a, text_b)
        if v1 and v2:
            return self._clamp01(self._cosine_sim(v1, v2))
        return 0.0

    def _get_arg_quality_model(self):
        if self._arg_quality is not None:
            return self._arg_quality
        if not self._aqa_arg_quality_use_model or not self._aqa_arg_quality_model:
            self._arg_quality = None
            return None
        if pipeline is None:
            self._arg_quality = None
            return None
        try:
            self._arg_quality = pipeline(
                "text-classification",
                model=self._aqa_arg_quality_model,
                top_k=None,
            )
        except Exception:
            self._arg_quality = None
        return self._arg_quality

    def _similarity(self, text_a: str, text_b: str) -> float:
        text_a = self._normalize_text(text_a)
        text_b = self._normalize_text(text_b)
        if not text_a or not text_b:
            return 0.0
        vec_a = self._embed_text(text_a)
        vec_b = self._embed_text(text_b)
        if vec_a and vec_b:
            return self._clamp01(self._cosine_sim(vec_a, vec_b))
        v1, v2 = self._tfidf_vector(text_a, text_b)
        if v1 and v2:
            return self._clamp01(self._cosine_sim(v1, v2))
        # Fallback to Jaccard
        tokens_a = set(re.findall(r"\b\w{4,}\b", text_a.lower()))
        tokens_b = set(re.findall(r"\b\w{4,}\b", text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return self._clamp01(len(tokens_a & tokens_b) / len(tokens_a | tokens_b))

    def _readability_score(self, text: str) -> float:
        text = self._normalize_text(text)
        if not text:
            return 0.0
        if textstat is not None:
            try:
                flesch = textstat.flesch_reading_ease(text)
                fog = textstat.gunning_fog(text)
                smog = textstat.smog_index(text)
                flesch_score = self._clamp01(flesch / 100.0)
                fog_score = 1.0 - self._clamp01(fog / 20.0)
                smog_score = 1.0 - self._clamp01(smog / 20.0)
                scores = [flesch_score, fog_score, smog_score]
                return sum(scores) / len(scores)
            except Exception:
                pass
        # Fallback: prefer moderate length
        word_count = len(text.split())
        if word_count <= 0:
            return 0.0
        if word_count <= 30:
            return 0.6
        if word_count <= 80:
            return 0.8
        if word_count <= 150:
            return 0.6
        return 0.4

    def _argument_quality_score(
        self, premise_text: str, rule_text: str, conclusion_text: str
    ) -> float:
        structure = 0
        structure += 1 if premise_text else 0
        structure += 1 if rule_text else 0
        structure += 1 if conclusion_text else 0
        structure_score = structure / 3.0
        quality_text = f"{premise_text}\n{rule_text}\n{conclusion_text}".strip()
        model_score = None
        model = self._get_arg_quality_model()
        if model is not None and quality_text:
            try:
                result = model(quality_text)
                scores = []
                for item in result[0] if isinstance(result, list) else []:
                    if isinstance(item, dict) and "score" in item:
                        scores.append(float(item["score"]))
                if scores:
                    model_score = max(scores)
            except Exception:
                model_score = None
        if model_score is not None:
            quality_score = self._clamp01(model_score)
        else:
            similarity_score = self._tfidf_similarity(
                premise_text + " " + rule_text, conclusion_text
            )
            if similarity_score == 0.0:
                similarity_score = self._similarity(
                    premise_text + " " + rule_text, conclusion_text
                )
            quality_score = self._clamp01(similarity_score)
        return self._clamp01(0.5 * structure_score + 0.5 * quality_score)

    def _coherence_score(
        self, premise_text: str, conclusion_text: str, rule_text: str
    ) -> float:
        base = self._similarity(premise_text + " " + rule_text, conclusion_text)
        sentences = self._split_sentences(rule_text)
        if len(sentences) < 2:
            return base
        sims = []
        for idx in range(len(sentences) - 1):
            sims.append(self._similarity(sentences[idx], sentences[idx + 1]))
        if not sims:
            return base
        return self._clamp01(0.7 * base + 0.3 * (sum(sims) / len(sims)))

    def _build_chain_text(self, aspic_ir: dict) -> str:
        """
        Build the full text of a reasoning chain from ASPIC IR for norm support calculation.

        Extracts text from all arguments (premises, rules, conclusions) and reasoning chain steps.

        Args:
            aspic_ir: ASPIC IR structure

        Returns:
            Concatenated text of the entire chain.
        """
        if not aspic_ir:
            return ""

        parts = []

        # Extract from arguments
        for arg in aspic_ir.get("arguments", []):
            # Premises
            for premise in arg.get("premises", []):
                if isinstance(premise, dict):
                    parts.append(premise.get("text", ""))
                elif isinstance(premise, str):
                    parts.append(premise)
            # Rule/norm
            rule = arg.get("rule") or arg.get("norm") or {}
            if isinstance(rule, dict):
                parts.append(rule.get("text", ""))
            elif isinstance(rule, str):
                parts.append(rule)
            # Conclusion
            conclusion = arg.get("conclusion") or {}
            if isinstance(conclusion, dict):
                parts.append(conclusion.get("text", ""))
            elif isinstance(conclusion, str):
                parts.append(conclusion)

        # Extract from reasoning_chain steps
        for step in aspic_ir.get("reasoning_chain", []):
            if isinstance(step, dict):
                parts.append(step.get("text", ""))
            elif isinstance(step, str):
                parts.append(step)

        # Filter empty and join
        return " ".join(p for p in parts if p and isinstance(p, str))

    def _norm_support_score(
        self,
        text: str,
        retrieved_norms: list[dict] | None = None,
        max_citations: int | None = None,
    ) -> tuple[float, dict]:
        citations = self._extract_statute_citations(text)
        max_citations = max_citations or self._aqa_normsupport_max_citations
        citation_count = len(citations)
        citation_score = min(citation_count, max_citations) / max_citations
        retrieved_score = 0.0
        if retrieved_norms:
            scores = []
            for item in retrieved_norms:
                val = item.get("similarity") or item.get("score") or 0.0
                if isinstance(val, (int, float)):
                    scores.append(float(val))
            if scores:
                if self._aqa_normsupport_retrieved_agg == "max":
                    retrieved_score = max(scores)
                else:
                    retrieved_score = sum(scores) / len(scores)
        final_score = self._clamp01(
            self._aqa_normsupport_citation_weight * citation_score
            + self._aqa_normsupport_retrieved_weight * retrieved_score
        )
        details = {
            "citation_count": citation_count,
            "citation_score": citation_score,
            "retrieved_score": retrieved_score,
            "final": final_score,
        }
        return final_score, details

    def _semantics_score(
        self, premise_text: str, conclusion_text: str
    ) -> tuple[float, dict]:
        nli = self._get_nli_pipeline()
        if nli is not None:
            try:
                result = nli({"text": premise_text, "text_pair": conclusion_text})
                entail = 0.0
                for item in result[0]:
                    if item.get("label", "").lower().startswith("entail"):
                        entail = float(item.get("score", 0.0))
                        break
                return self._clamp01(entail), {"method": "nli", "score": entail}
            except Exception:
                pass
        score = self._similarity(premise_text, conclusion_text)
        return self._clamp01(score), {"method": "similarity", "score": score}

    def _extract_statute_refs_from_text(self, text: str) -> list[dict]:
        refs = []
        for match in self.STATUTE_PATTERN.finditer(text):
            art = match.group(1)
            code_part = match.group(2) or ""
            source = ""
            if code_part:
                source = (
                    "codice_civile"
                    if "c" in code_part.lower() and "p" not in code_part.lower()
                    else "codice_penale"
                )
            refs.append({"articolo": art, "source": source})
        return refs

    def _get_statute_meta(self, article_num: str, domain: str) -> dict:
        key = (article_num, domain)
        if key in self._statute_meta_cache:
            return self._statute_meta_cache[key]
        driver = get_driver()
        if domain == "CIVILE":
            articolo = f"art{article_num}"
            codice = "codice_civile"
        elif domain == "PENALE":
            articolo = article_num
            codice = "codice_penale"
        else:
            meta = self._get_statute_meta(article_num, "CIVILE")
            if meta:
                return meta
            meta = self._get_statute_meta(article_num, "PENALE")
            if meta:
                return meta
            return {}
        query = """
            MATCH (s:Statute)
            WHERE s.articolo = $articolo AND s.source = $codice
            RETURN s.libro AS libro, s.source AS source
            LIMIT 1
        """
        try:
            with driver.session() as session:
                record = session.run(
                    query, {"articolo": articolo, "codice": codice}
                ).single()
                if record:
                    meta = {
                        "libro": record.get("libro"),
                        "source": record.get("source"),
                    }
                    self._statute_meta_cache[key] = meta
                    return meta
        except Exception:
            pass
        self._statute_meta_cache[key] = {}
        return {}

    def _derive_severity_category(self, meta: dict) -> str:
        source = meta.get("source") or ""
        libro = (meta.get("libro") or "").lower()
        match = re.search(
            r"libro\s+(iv|vi|v|iii|ii|i|primo|secondo|terzo|quarto|quinto|sesto)",
            libro,
        )
        token = match.group(1) if match else ""
        if source == "codice_penale":
            return (self._aqa_severity_map_penale or {}).get(token, "")
        if source == "codice_civile":
            if "fuori range" in libro:
                return ""
            return (self._aqa_severity_map_civile or {}).get(token, "")
        return ""

    def _collect_links(self, aspic_ir: dict, role: str, domain: str) -> list[dict]:
        links: list[dict] = []
        reasoning_chain = aspic_ir.get("reasoning_chain", [])

        # Check if this is a repaired IR
        repair_meta = aspic_ir.get("_repair_metadata", {})
        if repair_meta:
            self._log(
                f"   🔧 [{role.upper()}] Using REPAIRED IR: "
                f"{repair_meta.get('total_repaired', 0)} repaired, "
                f"{repair_meta.get('total_dropped', 0)} dropped"
            )

        # Filter reasoning_chain to exclude noise steps (e.g., "Precedents: none found.")
        valid_chain_steps = [
            step
            for step in reasoning_chain
            if step.get("text")
            and not step.get("text", "").lower().startswith("precedent")
        ]

        if len(valid_chain_steps) < 2:
            self._log(
                f"   ⚠️ [{role.upper()}] Not enough chain steps to build links (found {len(valid_chain_steps)})"
            )
            return links

        self._log(
            f"   📋 [{role.upper()}] Building {len(valid_chain_steps) - 1} links from reasoning_chain"
        )

        for idx in range(len(valid_chain_steps) - 1):
            step_from = valid_chain_steps[idx]
            step_to = valid_chain_steps[idx + 1]

            step_from_id = step_from.get("id") or f"S{idx + 1}"
            step_to_id = step_to.get("id") or f"S{idx + 2}"
            link_id = f"{step_from_id}->{step_to_id}"

            premise_text = self._normalize_text(step_from.get("text", ""))
            conclusion_text = self._normalize_text(step_to.get("text", ""))
            # The "rule" or nesso is the logical connection between the two steps
            # We use the premise text as it contains the normative basis
            link_text = premise_text

            # Extract citations from both steps
            citations_text = f"{premise_text} {conclusion_text}"
            statute_refs = self._extract_statute_refs_from_text(citations_text)

            libri = set()
            severities = set()
            for ref in statute_refs:
                meta = self._get_statute_meta(ref.get("articolo", ""), domain)
                libro = meta.get("libro")
                if libro:
                    libri.add(libro)
                severity = self._derive_severity_category(meta)
                if severity:
                    severities.add(severity)

            link_libro = list(libri)[0] if len(libri) == 1 else ""
            severity_category = list(severities)[0] if len(severities) == 1 else ""

            # Log the link
            self._log(
                f"      🔗 Link {idx + 1}: {link_id} | "
                f"from={step_from_id} to={step_to_id} | norms={len(statute_refs)}"
            )
            if statute_refs:
                refs_str = ", ".join(
                    f"Art. {r.get('articolo', '?')}" for r in statute_refs[:3]
                )
                self._log(f"         📜 Statutes: {refs_str}")
            if link_libro or severity_category:
                self._log(
                    f"         🗂️ Libro: {link_libro or '-'} | Severity: {severity_category or '-'}"
                )

            links.append(
                {
                    "link_id": link_id,
                    "argument_id": f"chain_{idx + 1}",
                    "premise_ids": [step_from_id],
                    "conclusion_id": step_to_id,
                    "rule_id": "",
                    "role": role,
                    "text": link_text,
                    "premise_text": premise_text,
                    "conclusion_text": conclusion_text,
                    "libro": link_libro,
                    "severity_category": severity_category,
                    "retrieved_norms": None,
                }
            )

        self._log(
            f"   ✅ [{role.upper()}] Collected {len(links)} links from reasoning_chain"
        )
        return links

    def _compute_cross_attacks(
        self, pro_links: list[dict], contra_links: list[dict]
    ) -> None:
        for link in pro_links + contra_links:
            link["attacks_received"] = []
            link["attacks_sum"] = 0.0
        for target in pro_links:
            for opponent in contra_links:
                has_libro = target.get("libro") and opponent.get("libro")
                has_severity = target.get("severity_category") and opponent.get(
                    "severity_category"
                )
                allowed = (
                    has_libro
                    and has_severity
                    and target.get("libro") == opponent.get("libro")
                    and target.get("severity_category")
                    == opponent.get("severity_category")
                )
                if not allowed:
                    if not has_libro or not has_severity:
                        reason = "denied: missing severity/libro"
                    else:
                        reason = "denied: domain rules"
                else:
                    reason = "allowed"
                overlap = (
                    self._similarity(target.get("text", ""), opponent.get("text", ""))
                    if allowed
                    else 0.0
                )
                attack_value = overlap * opponent.get("total_score", 0.0)
                if overlap > 0.0:
                    target["attacks_received"].append(
                        {
                            "opponent_link_id": opponent.get("link_id"),
                            "overlap": overlap,
                            "opponent_score": opponent.get("base_score", 0.0),
                            "attack_value": attack_value,
                            "reason": reason,
                        }
                    )
                    target["attacks_sum"] += attack_value
                else:
                    target["attacks_received"].append(
                        {
                            "opponent_link_id": opponent.get("link_id"),
                            "overlap": 0.0,
                            "opponent_score": opponent.get("base_score", 0.0),
                            "attack_value": 0.0,
                            "reason": reason,
                        }
                    )
        for link in pro_links + contra_links:
            attacks = sorted(
                link.get("attacks_received", []),
                key=lambda x: x.get("attack_value", 0.0),
                reverse=True,
            )
            link["attacks_received"] = attacks[: self._aqa_attack_top_k]
            link["attacks_sum"] = sum(a.get("attack_value", 0.0) for a in attacks)

    def _extract_year(self, text: str) -> int | None:
        if not text:
            return None
        matches = re.findall(r"(?:19|20)\d{2}", text)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except ValueError:
            return None

    def _compute_recency(self, year: int | None) -> float:
        if not year:
            return 0.0
        current_year = datetime.utcnow().year
        age = max(0, current_year - year)
        max_age = 50.0
        return self._clamp01(1.0 - (age / max_age))

    def _stance_from_link_type(self, link_type: str | None) -> int:
        link_type = (link_type or "").lower()
        if link_type in {"supports", "support"}:
            return 1
        if link_type in {"attacks", "attack", "contradicts", "opposes", "against"}:
            return -1
        return 0

    def _build_precedent_text(self, meta: dict) -> str:
        parts = [
            meta.get("title") or "",
            meta.get("summary") or "",
            meta.get("excerpt") or "",
        ]
        return self._normalize_text(" ".join(p for p in parts if p))

    def _build_precedent_influence(
        self, prec_meta: dict, link: dict, edge: dict, match_reason: str
    ) -> dict:
        link_text = self._normalize_text(
            " ".join(
                [
                    link.get("premise_text", ""),
                    link.get("text", ""),
                    link.get("conclusion_text", ""),
                ]
            )
        )
        prec_text = self._build_precedent_text(prec_meta)
        similarity = prec_meta.get("similarity")
        if similarity is None:
            similarity = prec_meta.get("score")
        if similarity is None and prec_text and link_text:
            similarity = self._similarity(link_text, prec_text)
        similarity = self._clamp01(self._safe_float(similarity, 0.0))

        stance = self._stance_from_link_type(edge.get("type"))
        if stance == 0:
            stance = 1

        court = (
            prec_meta.get("court")
            or prec_meta.get("court_level")
            or prec_meta.get("organo")
            or prec_meta.get("title")
            or ""
        )
        bindingness = prec_meta.get("bindingness")
        if bindingness is None:
            bindingness = self._bindingness_from_court(court)
        bindingness = self._clamp01(self._safe_float(bindingness, 0.0))

        year = prec_meta.get("year") or self._extract_year(
            f"{prec_meta.get('title','')} {prec_meta.get('summary','')}"
        )
        recency = prec_meta.get("recency")
        if recency is None:
            recency = self._compute_recency(year)
        recency = self._clamp01(self._safe_float(recency, 0.0))

        confidence = prec_meta.get("confidence")
        if confidence is None:
            confidence = prec_meta.get("stance_confidence")
        if confidence is None:
            confidence = 0.7
        confidence = self._clamp01(self._safe_float(confidence, 0.7))

        return {
            "precedent_id": prec_meta.get("precedent_id") or prec_meta.get("title"),
            "stance": stance,
            "bindingness": bindingness,
            "similarity": similarity,
            "recency": recency,
            "confidence": confidence,
            "court": court,
            "year": year,
            "link_match": match_reason,
        }

    def _populate_precedent_influences(self, aspic_ir: dict, links: list[dict]) -> None:
        if not aspic_ir or not links:
            return
        precedent_nodes = {
            node.get("id"): node
            for node in aspic_ir.get("precedent_nodes", [])
            if node.get("id")
        }
        precedent_by_id = {
            node.get("precedent_id"): node
            for node in aspic_ir.get("precedent_nodes", [])
            if node.get("precedent_id")
        }

        target_to_link: dict[str, dict] = {}
        for link in links:
            targets = set()
            for key in (
                link.get("link_id"),
                link.get("argument_id"),
                link.get("rule_id"),
                link.get("conclusion_id"),
            ):
                if key:
                    targets.add(key)
            for pid in link.get("premise_ids", []) or []:
                if pid:
                    targets.add(pid)
            link["precedent_influences"] = link.get("precedent_influences", [])
            for target in targets:
                target_to_link[target] = link

        step_text_by_id = {
            step.get("id"): self._normalize_text(step.get("text", ""))
            for step in aspic_ir.get("reasoning_chain", [])
            if step.get("id")
        }

        used = set()
        for edge in aspic_ir.get("precedent_links", []) or []:
            prec_node_id = edge.get("from") or ""
            target_id = edge.get("to") or ""
            prec_meta = precedent_nodes.get(prec_node_id) or precedent_by_id.get(
                prec_node_id
            )
            if not prec_meta:
                continue

            matched_link: dict | None = target_to_link.get(target_id)
            match_reason = "direct"
            if matched_link is None and target_id in step_text_by_id:
                step_text = step_text_by_id.get(target_id, "")
                best_link = None
                best_score = 0.0
                for candidate in links:
                    candidate_text = self._normalize_text(
                        " ".join(
                            [
                                candidate.get("premise_text", ""),
                                candidate.get("text", ""),
                                candidate.get("conclusion_text", ""),
                            ]
                        )
                    )
                    score = self._similarity(step_text, candidate_text)
                    if score > best_score:
                        best_score = score
                        best_link = candidate
                if best_link and best_score >= 0.5:
                    matched_link = best_link
                    match_reason = "chain_step_similarity"

            if matched_link is None:
                continue

            prec_id = prec_meta.get("precedent_id") or prec_meta.get("title")
            link_id = matched_link.get("link_id")
            dedup_key = (link_id, prec_id)
            if dedup_key in used:
                continue
            used.add(dedup_key)

            influence = self._build_precedent_influence(
                prec_meta=prec_meta,
                link=matched_link,
                edge=edge,
                match_reason=match_reason,
            )
            matched_link["precedent_influences"].append(influence)

    def _compute_precedent_delta(self, link: dict) -> tuple[float, list[dict]]:
        influences = []
        total_delta = 0.0
        precedents = link.get("precedent_influences") or []
        for prec in precedents:
            stance = prec.get("stance", 0)
            if isinstance(stance, str):
                stance_norm = stance.strip().lower()
                if stance_norm in {"support", "pro", "favour"}:
                    stance = 1
                elif stance_norm in {"contradict", "contra", "against"}:
                    stance = -1
                else:
                    stance = 0
            bindingness = prec.get("bindingness", 0.0)
            if not bindingness:
                court = (
                    prec.get("court")
                    or prec.get("court_level")
                    or prec.get("organo")
                    or ""
                )
                bindingness = self._bindingness_from_court(court)
            similarity = prec.get("similarity", 0.0)
            recency = prec.get("recency", 0.0)
            confidence = prec.get("confidence", 0.0)
            try:
                delta = (
                    float(bindingness)
                    * float(similarity)
                    * float(recency)
                    * float(stance)
                    * float(confidence)
                )
            except Exception:
                delta = 0.0
            influences.append(
                {
                    "precedent_id": prec.get("precedent_id"),
                    "delta": delta,
                }
            )
            total_delta += delta
        return total_delta, influences

    def _bindingness_from_court(self, court: str) -> float:
        court_norm = (court or "").lower()
        if "cass" in court_norm:
            return 1.0
        if "appello" in court_norm:
            return 0.7
        if "tribunale" in court_norm:
            return 0.4
        return 0.0

    def _run_aqa_phase(self, reasoner_ir: dict, counter_ir: dict, domain: str) -> dict:
        if not self._aqa_enabled:
            self._log("🧪 AQA disabled - skipping scoring")
            return {"enabled": False}
        self._log("🧪 Starting AQA scoring on REPAIRED reasoning chains...")
        self._log(f"🧠 AQA domain: {domain}")

        # Check if we're using repaired IRs
        reasoner_repaired = bool(reasoner_ir.get("_repair_metadata"))
        counter_repaired = bool(counter_ir.get("_repair_metadata"))
        if reasoner_repaired or counter_repaired:
            self._log(
                f"   ✅ Using repaired chains: Reasoner={reasoner_repaired}, Counter={counter_repaired}"
            )
        else:
            self._log("   ℹ️ No repairs were needed - using original chains")

        pro_links = self._collect_links(reasoner_ir, "support", domain)
        contra_links = self._collect_links(counter_ir, "counter", domain)
        total_links = len(pro_links) + len(contra_links)
        self._log(
            f"🔗 AQA links collected: pro={len(pro_links)} contra={len(contra_links)}"
        )
        self._populate_precedent_influences(reasoner_ir, pro_links)
        self._populate_precedent_influences(counter_ir, contra_links)

        # Calculate norm_support once per chain (not per link)
        reasoner_chain_text = self._build_chain_text(reasoner_ir)
        counter_chain_text = self._build_chain_text(counter_ir)
        pro_norm_support, pro_norm_details = self._norm_support_score(
            text=reasoner_chain_text,
            retrieved_norms=None,
        )
        contra_norm_support, contra_norm_details = self._norm_support_score(
            text=counter_chain_text,
            retrieved_norms=None,
        )
        self._log(
            f"📚 Chain-level norm support: pro={pro_norm_support:.2f} "
            f"(cit={pro_norm_details.get('citation_count', 0)}), "
            f"contra={contra_norm_support:.2f} "
            f"(cit={contra_norm_details.get('citation_count', 0)})"
        )

        for idx, link in enumerate(pro_links + contra_links, start=1):
            link_id = link.get("link_id") or f"L{idx}"
            role = (link.get("role") or "link").upper()
            self._log(f"   🔹 [{idx}/{total_links}] {role} link {link_id}")
            premise_text = link.get("premise_text", "")
            rule_text = link.get("text", "")
            conclusion_text = link.get("conclusion_text", "")

            # Fallback: if premise_text is empty, use rule_text for semantics
            semantics_premise = premise_text if premise_text else rule_text
            if not premise_text and rule_text:
                self._log("      ⚠️ premise_text empty, using rule_text for semantics")

            semantics, semantics_details = self._semantics_score(
                semantics_premise, conclusion_text
            )
            # Add fallback info to semantics_details
            if not premise_text and rule_text:
                semantics_details["used_fallback"] = True
                semantics_details["fallback_source"] = "rule_text"

            coherence = self._coherence_score(premise_text, conclusion_text, rule_text)
            arg_quality = self._argument_quality_score(
                premise_text, rule_text, conclusion_text
            )
            # Cogency: average of coherence and argument_quality (readability removed)
            cogency = self._clamp01((coherence + arg_quality) / 2.0)
            cogency_details = {
                "coherence_score": coherence,
                "argument_quality_score": arg_quality,
                "explanation": "coherence/argument_quality averaged (readability removed)",
            }

            # Link base_score uses only cogency and semantics (normSupport is chain-level)
            # Normalize weights for 2 parameters: alpha/(alpha+gamma) and gamma/(alpha+gamma)
            weight_sum = self._aqa_alpha + self._aqa_gamma
            if weight_sum > 0:
                base = (self._aqa_alpha / weight_sum) * cogency + (
                    self._aqa_gamma / weight_sum
                ) * semantics
            else:
                base = (cogency + semantics) / 2.0

            self._log(
                "      🧠 Cogency "
                f"{cogency:.2f} (coh {coherence:.2f}, qual {arg_quality:.2f})"
            )
            self._log(
                "      🧩 Semantics "
                f"{semantics:.2f} ({semantics_details.get('method', 'n/a')})"
            )
            self._log(f"      ➕ Link base score {base:.2f}")
            if link.get("libro") or link.get("severity_category"):
                self._log(
                    "      🗂️ Severity "
                    f"{link.get('severity_category') or '-'} / libro "
                    f"{link.get('libro') or '-'}"
                )

            link["cogency"] = cogency
            link["semantics"] = semantics
            link["cogency_details"] = cogency_details
            link["semantics_details"] = semantics_details
            link["base_score"] = base
            link["total_score"] = base

        self._log("⚔️ Cross-attacks disabled - independent link scoring")

        for idx, link in enumerate(pro_links + contra_links, start=1):
            link_id = link.get("link_id") or f"L{idx}"
            role = (link.get("role") or "link").upper()
            delta, influences = self._compute_precedent_delta(link)
            link["precedent_delta"] = delta
            link["precedent_influences"] = influences
            nesso = self._clamp01(link.get("base_score", 0.0) + delta)
            link["nesso_plausibility"] = nesso
            self._log(
                "   🔸 "
                f"[{idx}/{total_links}] {role} link {link_id} "
                f"precedent Δ {delta:.2f} nesso {nesso:.2f}"
            )
            if influences:
                self._log(f"      📚 Precedent influences: {len(influences)}")

        # Helper to compute average of a field across links
        def avg_field(items: list[dict], field: str) -> float:
            if not items:
                return 0.0
            return sum(i.get(field, 0.0) for i in items) / len(items)

        # Calculate chain-level averages for cogency and semantics
        pro_cogency_avg = avg_field(pro_links, "cogency")
        pro_semantics_avg = avg_field(pro_links, "semantics")
        contra_cogency_avg = avg_field(contra_links, "cogency")
        contra_semantics_avg = avg_field(contra_links, "semantics")

        # base_score = α*cogency + β*normSupport + γ*semantics (at chain level)
        pro_base_score = self._clamp01(
            self._aqa_alpha * pro_cogency_avg
            + self._aqa_beta * pro_norm_support
            + self._aqa_gamma * pro_semantics_avg
        )
        contra_base_score = self._clamp01(
            self._aqa_alpha * contra_cogency_avg
            + self._aqa_beta * contra_norm_support
            + self._aqa_gamma * contra_semantics_avg
        )

        # Apply precedent delta at chain level (average of link deltas)
        pro_precedent_delta = avg_field(pro_links, "precedent_delta")
        contra_precedent_delta = avg_field(contra_links, "precedent_delta")

        pro_score = self._clamp01(pro_base_score + pro_precedent_delta)
        contra_score = self._clamp01(contra_base_score + contra_precedent_delta)
        final_plausibility = pro_score - contra_score

        self._log(
            f"📊 Chain averages: pro(cog={pro_cogency_avg:.2f}, sem={pro_semantics_avg:.2f}), "
            f"contra(cog={contra_cogency_avg:.2f}, sem={contra_semantics_avg:.2f})"
        )
        self._log(
            f"📚 base_score = α({self._aqa_alpha:.2f})*cogency + β({self._aqa_beta:.2f})*normSupport + γ({self._aqa_gamma:.2f})*semantics"
        )
        self._log(
            f"📈 Base scores: pro={pro_base_score:.2f}, contra={contra_base_score:.2f}"
        )
        self._log(
            f"⚖️ With precedent Δ: pro={pro_score:.2f} (+{pro_precedent_delta:.2f}), "
            f"contra={contra_score:.2f} (+{contra_precedent_delta:.2f})"
        )

        self._log(
            "📈 AQA scores "
            f"pro={pro_score:.2f} contra={contra_score:.2f} "
            f"final={final_plausibility:.2f}"
        )

        if final_plausibility >= self._aqa_verdict_pos:
            verdict = "plausible"
        elif final_plausibility <= self._aqa_verdict_neg:
            verdict = "implausible"
        else:
            verdict = "uncertain"

        self._log(
            "🧾 AQA verdict "
            f"{verdict} (pos≥{self._aqa_verdict_pos:.2f}, "
            f"neg≤{self._aqa_verdict_neg:.2f})"
        )

        weakest = sorted(
            pro_links + contra_links,
            key=lambda x: x.get("nesso_plausibility", 0.0),
        )[:3]
        if weakest:
            weakest_ids = ", ".join(
                str(item.get("link_id") or "link") for item in weakest
            )
            self._log(f"🔎 Weakest links: {weakest_ids}")
        precedent_swings = [
            {
                "link_id": link.get("link_id"),
                "delta": link.get("precedent_delta"),
            }
            for link in pro_links + contra_links
            if abs(link.get("precedent_delta", 0.0)) > 0.0
        ]
        self._log("⚔️ Dominant attacks: disabled")
        if precedent_swings:
            self._log(f"📚 Precedent swings: {len(precedent_swings)}")
        severity_debug = [
            {
                "link_id": link.get("link_id"),
                "role": link.get("role"),
                "libro": link.get("libro"),
                "severity_category": link.get("severity_category"),
            }
            for link in pro_links + contra_links
        ]

        return {
            "enabled": True,
            "weights": {
                "alpha": self._aqa_alpha,
                "beta": self._aqa_beta,
                "gamma": self._aqa_gamma,
            },
            "links": {
                "pro": pro_links,
                "contra": contra_links,
            },
            "chain_scores": {
                "pro": {
                    "cogency_avg": pro_cogency_avg,
                    "semantics_avg": pro_semantics_avg,
                    "norm_support": pro_norm_support,
                    "norm_support_details": pro_norm_details,
                    "base_score": pro_base_score,
                    "precedent_delta": pro_precedent_delta,
                    "final_score": pro_score,
                },
                "contra": {
                    "cogency_avg": contra_cogency_avg,
                    "semantics_avg": contra_semantics_avg,
                    "norm_support": contra_norm_support,
                    "norm_support_details": contra_norm_details,
                    "base_score": contra_base_score,
                    "precedent_delta": contra_precedent_delta,
                    "final_score": contra_score,
                },
            },
            "net_plausibility": {
                "pro": pro_score,
                "contra": contra_score,
                "final": final_plausibility,
            },
            "verdict": verdict,
            "notes": {
                "weakest_links": weakest,
                "dominant_attacks": [],
                "attacks_enabled": False,
                "precedent_swings": precedent_swings,
                "severity_debug": severity_debug,
            },
        }

    # ------------------------------------------------------------------
    # Summary Generation
    # ------------------------------------------------------------------
    def _generate_consistency_summary(
        self,
        reasoner_report: ConsistencyReport,
        counter_report: ConsistencyReport,
    ) -> str:
        """Generate a human-readable summary of the consistency analysis."""
        lines = []

        lines.append("## Analisi di Consistenza\n")

        # Reasoner summary
        lines.append("### Reasoner")
        lines.append(
            f"- Citazioni valide: {reasoner_report.valid_citations}/{reasoner_report.total_citations}"
        )
        lines.append(
            f"- Testo verificato: {reasoner_report.text_matches} match / {reasoner_report.text_matches + reasoner_report.text_mismatches} verificati"
        )
        if reasoner_report.repaired_citations > 0:
            lines.append(f"- Citazioni riparate: {reasoner_report.repaired_citations}")
        if reasoner_report.dropped_citations > 0:
            lines.append(f"- Citazioni scartate: {reasoner_report.dropped_citations}")
        lines.append(f"- Score di consistenza: {reasoner_report.consistency_score:.2%}")
        if reasoner_report.issues:
            lines.append(f"- Problemi: {len(reasoner_report.issues)}")

        lines.append("")

        # Counter-Reasoner summary
        lines.append("### Counter-Reasoner")
        lines.append(
            f"- Citazioni valide: {counter_report.valid_citations}/{counter_report.total_citations}"
        )
        lines.append(
            f"- Testo verificato: {counter_report.text_matches} match / {counter_report.text_matches + counter_report.text_mismatches} verificati"
        )
        if counter_report.repaired_citations > 0:
            lines.append(f"- Citazioni riparate: {counter_report.repaired_citations}")
        if counter_report.dropped_citations > 0:
            lines.append(f"- Citazioni scartate: {counter_report.dropped_citations}")
        lines.append(f"- Score di consistenza: {counter_report.consistency_score:.2%}")
        if counter_report.issues:
            lines.append(f"- Problemi: {len(counter_report.issues)}")

        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reasoning Chain Repair (LLM Reworking)
    # ------------------------------------------------------------------
    def _regenerate_reasoning_chain_with_llm(
        self,
        original_chain: str,
        citation_checks: list[CitationCheck],
        agent_name: str,
    ) -> str:
        """
        Regenerate the reasoning chain using LLM to integrate repairs and handle dropped citations.

        If no citations were repaired or dropped, returns the original chain unchanged.

        Args:
            original_chain: The original reasoning chain text
            citation_checks: List of CitationCheck objects with mismatch actions
            agent_name: Name of the agent ("reasoner" or "counter_reasoner")

        Returns:
            The regenerated (or original if no changes needed) reasoning chain.
        """
        # Collect repaired and dropped citations
        repaired = []
        dropped = []

        for check in citation_checks:
            if check.mismatch_action == MismatchAction.REPAIRED.value:
                repaired.append(
                    {
                        "citation": check.citation,
                        "original_text": check.cited_text,
                        "repaired_text": check.repaired_text,
                    }
                )
            elif check.mismatch_action in (
                MismatchAction.DROPPED.value,
                MismatchAction.REPAIR_FAILED.value,
            ):
                dropped.append(
                    {
                        "citation": check.citation,
                        "original_text": check.cited_text,
                        "reason": (
                            "peripheral citation"
                            if check.mismatch_action == MismatchAction.DROPPED.value
                            else "repair failed"
                        ),
                    }
                )

        # If no changes needed, return original
        if not repaired and not dropped:
            self._log(f"   ✅ [{agent_name}] No repairs needed - chain unchanged")
            return original_chain

        self._log(
            f"   🔄 [{agent_name}] Regenerating chain: {len(repaired)} repaired, {len(dropped)} dropped"
        )

        try:
            llm = get_chat_groq(temperature=0, max_tokens=2048)

            system_prompt = """You are an expert in Italian law. Your task is to rewrite a legal reasoning chain to integrate corrections.

You will receive:
1. The ORIGINAL reasoning chain
2. A list of REPAIRED citations (with the corrected normative text to use)
3. A list of DROPPED citations (citations that should be removed or reformulated without them)

RULES:
1. Integrate the repaired texts naturally into the reasoning
2. For dropped citations, reformulate the argument to work without them, or mark them as "[Citation removed - unreliable source]"
3. Maintain the logical flow and coherence of the argument
4. Keep the same structure and style
5. Write the output in Italian
6. Do not add new legal concepts not present in the original

OUTPUT: Write ONLY the revised reasoning chain in Italian, without explanations or meta-commentary."""

            # Build the user prompt
            repaired_info = ""
            if repaired:
                repaired_info = "\n\nREPAIRED CITATIONS (use these corrected texts):\n"
                for r in repaired:
                    repaired_info += f"- {r['citation']}:\n  Original: \"{r['original_text'][:200]}...\"\n  Corrected: \"{r['repaired_text']}\"\n"

            dropped_info = ""
            if dropped:
                dropped_info = (
                    "\n\nDROPPED CITATIONS (remove or reformulate without these):\n"
                )
                for d in dropped:
                    dropped_info += f"- {d['citation']}: \"{d['original_text'][:100]}...\" (Reason: {d['reason']})\n"

            user_prompt = f"""ORIGINAL REASONING CHAIN:
\"\"\"
{original_chain[:3000]}
\"\"\"
{repaired_info}{dropped_info}

Rewrite the reasoning chain in Italian, integrating the corrections and handling the dropped citations appropriately."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            regenerated = response.content.strip()

            self._log(
                f"   ✅ [{agent_name}] Chain regenerated successfully ({len(regenerated)} chars)"
            )
            return regenerated

        except Exception as e:
            self._log(f"   ⚠️ [{agent_name}] Chain regeneration failed: {e}", "warning")
            # Return original with annotations on failure
            return original_chain

    def _repair_aspic_ir(
        self,
        aspic_ir: dict,
        citation_checks: list[CitationCheck],
        repaired_chain_text: str = "",
        claim: str = "",
        role: str = "support",
        statutes: list[dict] | None = None,
        precedents: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Rebuild the ASPIC IR from the repaired reasoning chain.

        Instead of patching the original IR, this method re-parses the repaired
        chain text and rebuilds the IR from scratch using AspicFormatter,
        exactly like Reasoner/Counter-Reasoner do for their original chains.

        Args:
            aspic_ir: Original ASPIC IR structure (used only for fallback/metadata)
            citation_checks: List of CitationCheck objects with mismatch actions
            repaired_chain_text: The LLM-regenerated reasoning chain text
            claim: The original legal claim
            role: "support" or "counter"
            statutes: Relevant statutes for the formatter
            precedents: Relevant precedents for the formatter
            metadata: Additional metadata (causal_type_id, theory_id, etc.)

        Returns:
            Rebuilt ASPIC IR structure from the repaired chain, or empty dict
            if no repairs were needed.
        """
        if not aspic_ir:
            return {}

        # Count repairs and drops
        total_repaired = sum(
            1
            for c in citation_checks
            if c.mismatch_action == MismatchAction.REPAIRED.value
        )
        total_dropped = sum(
            1
            for c in citation_checks
            if c.mismatch_action
            in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value)
        )

        # If no changes were needed, return empty (signals "use original")
        if total_repaired == 0 and total_dropped == 0:
            return {}

        # If no repaired text available, fall back to original
        if not repaired_chain_text:
            self._log(
                f"   ⚠️ [{role}] No repaired chain text available, skipping IR rebuild"
            )
            return {}

        self._log(
            f"   🔄 [{role.upper()}] Rebuilding ASPIC IR from repaired chain "
            f"(repaired={total_repaired}, dropped={total_dropped})"
        )

        # Parse the repaired chain text the same way agents do
        reasoning_chain = self._extract_reasoning_chain(repaired_chain_text)
        reasoning_chain = self._sanitize_reasoning_chain(
            reasoning_chain, precedents or []
        )
        arguments = self._extract_arguments_from_text(repaired_chain_text)

        # Build new IR using AspicFormatter (same as Reasoner/Counter-Reasoner)
        formatter = AspicFormatter(
            role=role,
            statutes=statutes or [],
            precedents=precedents or [],
        )
        rebuilt_ir = formatter.format(
            claim=claim or aspic_ir.get("claim", ""),
            raw_response=repaired_chain_text,
            reasoning_chain=reasoning_chain,
            arguments=arguments,
            metadata=metadata or aspic_ir.get("metadata", {}),
        )

        # Add repair metadata
        rebuilt_ir["_repair_metadata"] = {
            "total_repaired": total_repaired,
            "total_dropped": total_dropped,
        }

        self._log(
            f"   ✅ [{role.upper()}] ASPIC IR rebuilt: "
            f"{len(rebuilt_ir.get('arguments', []))} arguments, "
            f"{len(rebuilt_ir.get('reasoning_chain', []))} chain steps"
        )

        return rebuilt_ir

    def _extract_arguments_from_text(self, response: str) -> list[dict]:
        """
        Extract argument blocks from raw response text.

        Reuses the same logic as Reasoner._extract_arguments and
        CounterReasoner._extract_arguments to parse structured blocks.
        """
        arguments = []
        current_arg: dict[str, str] = {}
        current_section = None

        section_markers = {
            "premessa": "premise",
            "premessa alternativa": "premise",
            "norma": "norm",
            "nesso causale": "link",
            "nesso causale alternativo": "link",
            "causal link": "link",
            "conclusione": "conclusion",
            "conclusione contraria": "conclusion",
        }

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Remove markdown bold markers
            clean = line.replace("**", "").strip()
            lower_clean = clean.lower()

            # Check for section headers
            matched_section = None
            for marker, section_key in section_markers.items():
                if lower_clean.startswith(marker):
                    matched_section = section_key
                    # Extract content after the marker
                    content = clean[len(marker) :].lstrip(" :.-").strip()
                    break

            if matched_section:
                if matched_section == "premise" and current_arg.get("conclusion"):
                    # New argument block
                    if current_arg:
                        arguments.append(current_arg)
                    current_arg = {}

                current_section = matched_section
                if content:
                    current_arg[current_section] = content
                continue

            # Check for chain/reasoning section → stop parsing arguments
            if any(
                kw in lower_clean
                for kw in [
                    "ragionamento",
                    "chain of reasoning",
                    "catena di ragionamento",
                ]
            ):
                break

            # Continuation of current section
            if current_section and line:
                existing = current_arg.get(current_section, "")
                current_arg[current_section] = (
                    (existing + " " + line).strip() if existing else line
                )

        if current_arg:
            arguments.append(current_arg)

        return arguments

    def compute_grounded_extension(
        self,
        arguments: list[dict],
        attacks: list[tuple[str, str]],
    ) -> set[str]:
        """
        Compute the grounded extension of the argumentation framework.

        The grounded extension is the minimal complete labeling that
        assigns arguments as accepted, rejected, or undecided.

        Args:
            arguments: List of all arguments (both support and counter).
            attacks: List of (attacker_id, target_id) tuples.

        Returns:
            Set of accepted argument IDs.

        TODO: Implement using Dung's grounded semantics.
        """
        raise NotImplementedError("Grounded extension not yet implemented")

    def evaluate_argument_status(
        self,
        argument_id: str,
        grounded_extension: set[str],
        attacks: list[tuple[str, str]],
    ) -> ArgumentStatus:
        """
        Determine the status of a specific argument.

        TODO: Implement.
        """
        raise NotImplementedError("Argument status evaluation not yet implemented")

    def generate_summary(
        self,
        evaluated_arguments: list[EvaluatedArgument],
        winning_side: str,
    ) -> str:
        """
        Generate a human-readable summary of the evaluation.

        TODO: Implement using LLM.
        """
        raise NotImplementedError("Summary generation not yet implemented")

    def polish_response(
        self,
        claim: str,
        summary: str,
        winning_arguments: list[EvaluatedArgument],
    ) -> str:
        """
        Produce a polished, well-formatted final response.

        This is the main output that will be shown to the user,
        summarizing the legal analysis and conclusions.

        TODO: Implement using LLM.
        """
        raise NotImplementedError("Response polishing not yet implemented")

    def build_dialectical_tree(
        self,
        arguments: list[dict],
        counter_arguments: list[dict],
        attacks: list[tuple[str, str]],
    ) -> dict:
        """
        Build a dialectical tree representation.

        The tree shows the back-and-forth of arguments and counter-arguments,
        with the root being the main claim.

        Returns a structure suitable for visualization.

        TODO: Implement.
        """
        raise NotImplementedError("Dialectical tree building not yet implemented")
