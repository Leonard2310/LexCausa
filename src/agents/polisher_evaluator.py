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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import get_driver


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
        if reasoner_report.repaired_citations > 0 or reasoner_report.dropped_citations > 0:
            repaired_reasoner_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=reasoner_raw,
                citation_checks=reasoner_report.citation_checks,
                agent_name="reasoner",
            )
        else:
            repaired_reasoner_chain = reasoner_raw  # No changes needed

        # Regenerate Counter-Reasoner chain if needed
        repaired_counter_chain = ""
        if counter_report.repaired_citations > 0 or counter_report.dropped_citations > 0:
            repaired_counter_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=counter_raw,
                citation_checks=counter_report.citation_checks,
                agent_name="counter_reasoner",
            )
        else:
            repaired_counter_chain = counter_raw  # No changes needed

        # Repair ASPIC IR structures
        repaired_reasoner_aspic = self._repair_aspic_ir(
            aspic_ir=reasoner_aspic_ir,
            citation_checks=reasoner_report.citation_checks,
        )
        repaired_counter_aspic = self._repair_aspic_ir(
            aspic_ir=counter_aspic_ir,
            citation_checks=counter_report.citation_checks,
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

        self._log("✅ Evaluation complete")

        return EvaluationResult(
            claim=claim,
            winning_side="",
            confidence=0.0,
            consistency_report={
                "reasoner": reasoner_report.to_dict(),
                "counter_reasoner": counter_report.to_dict(),
            },
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
            llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

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

            response = llm.invoke(messages)
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
            llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

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

            response = llm.invoke(messages)
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
                        f"      ❌ Repair FAILED: quote not found in DB or too short"
                    )
                    return False, ""
            else:
                self._log(f"      ❌ Repair FAILED: no «» quote in response")
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
        4. If peripheral or repair fails: drop the citation

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
            self._log(f"      ✅ LLM says texts are equivalent - treating as match")
            check.mismatch_action = MismatchAction.MATCH.value
            check.text_match = True
            report.text_matches += 1
            report.text_mismatches -= 1  # Correct the earlier increment
            return

        # Step 2: Classify as core or peripheral
        is_core = self._is_article_core(article_num, aspic_ir, full_text)
        check.is_core = is_core

        if is_core:
            # Step 3: Attempt repair
            self._log(f"      🔄 Attempting repair for CORE article...")
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
                self._log(f"      ❌ Art. {article_num} repair FAILED - marked as unreliable")
        else:
            # Peripheral: just drop it
            check.mismatch_action = MismatchAction.DROPPED.value
            check.details += " [DROPPED - peripheral citation]"
            report.dropped_citations += 1
            self._log(f"      🗑️ Art. {article_num} DROPPED (peripheral)")

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
                else:
                    check.details = (
                        f"Verified in Neo4j ({domain}), no text cited for verification"
                    )
                    if not cited_text:
                        self._log("      📝 No cited text found for verification")
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
                repaired.append({
                    "citation": check.citation,
                    "original_text": check.cited_text,
                    "repaired_text": check.repaired_text,
                })
            elif check.mismatch_action in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value):
                dropped.append({
                    "citation": check.citation,
                    "original_text": check.cited_text,
                    "reason": "peripheral citation" if check.mismatch_action == MismatchAction.DROPPED.value else "repair failed",
                })

        # If no changes needed, return original
        if not repaired and not dropped:
            self._log(f"   ✅ [{agent_name}] No repairs needed - chain unchanged")
            return original_chain

        self._log(f"   🔄 [{agent_name}] Regenerating chain: {len(repaired)} repaired, {len(dropped)} dropped")

        try:
            import json
            llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

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
                dropped_info = "\n\nDROPPED CITATIONS (remove or reformulate without these):\n"
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

            response = llm.invoke(messages)
            regenerated = response.content.strip()

            self._log(f"   ✅ [{agent_name}] Chain regenerated successfully ({len(regenerated)} chars)")
            return regenerated

        except Exception as e:
            self._log(f"   ⚠️ [{agent_name}] Chain regeneration failed: {e}", "warning")
            # Return original with annotations on failure
            return original_chain

    def _repair_aspic_ir(
        self,
        aspic_ir: dict,
        citation_checks: list[CitationCheck],
    ) -> dict:
        """
        Repair the ASPIC IR structure based on citation checks.

        Updates premises and conclusions with repaired texts, and marks
        dropped citations as inactive.

        Args:
            aspic_ir: Original ASPIC IR structure
            citation_checks: List of CitationCheck objects with mismatch actions

        Returns:
            Updated ASPIC IR structure (deep copy with modifications).
        """
        import copy

        if not aspic_ir:
            return {}

        # Deep copy to avoid modifying original
        repaired_ir = copy.deepcopy(aspic_ir)

        # Build lookup for citation checks by article number
        checks_by_article: dict[str, CitationCheck] = {}
        for check in citation_checks:
            # Extract article number from citation
            match = re.search(r"(\d{1,4})", check.citation)
            if match:
                article_num = match.group(1)
                checks_by_article[article_num] = check

        if not checks_by_article:
            return repaired_ir

        # Helper to check if citations dict contains an article
        def find_article_in_citations(citations: dict, art_num: str) -> bool:
            for statute in citations.get("statutes", []):
                if str(statute.get("articolo", "")).strip() == art_num:
                    return True
            return False

        # Process arguments
        for arg in repaired_ir.get("arguments", []):
            # Process premises
            new_premises = []
            for premise in arg.get("premises", []):
                citations = premise.get("citations", {})

                # Check each article in this premise
                should_drop = False
                for art_num, check in checks_by_article.items():
                    if find_article_in_citations(citations, art_num):
                        if check.mismatch_action == MismatchAction.REPAIRED.value:
                            # Update text with repaired version
                            premise["text"] = check.repaired_text
                            premise["_repaired"] = True
                            premise["_original_text"] = check.cited_text
                            self._log(f"      📝 ASPIC IR: Repaired Art. {art_num} in premise")
                        elif check.mismatch_action in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value):
                            # Mark as dropped
                            premise["_dropped"] = True
                            premise["_drop_reason"] = check.mismatch_action
                            should_drop = True
                            self._log(f"      🗑️ ASPIC IR: Marked Art. {art_num} premise as dropped")

                # Keep premise unless it's type="norm" and dropped
                if not (should_drop and premise.get("type") == "norm"):
                    new_premises.append(premise)
                else:
                    # Add a marker that this premise was removed
                    removed_marker = {
                        "type": "removed",
                        "original_type": premise.get("type"),
                        "reason": "dropped citation",
                        "citation": str(citations.get("statutes", [])),
                    }
                    new_premises.append(removed_marker)

            arg["premises"] = new_premises

            # Process conclusion
            conclusion = arg.get("conclusion", {})
            if conclusion:
                conclusion_citations = conclusion.get("citations", {})
                for art_num, check in checks_by_article.items():
                    if find_article_in_citations(conclusion_citations, art_num):
                        if check.mismatch_action == MismatchAction.REPAIRED.value:
                            conclusion["text"] = check.repaired_text
                            conclusion["_repaired"] = True
                            self._log(f"      📝 ASPIC IR: Repaired Art. {art_num} in conclusion")
                        elif check.mismatch_action in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value):
                            conclusion["_has_dropped_citation"] = True
                            self._log(f"      ⚠️ ASPIC IR: Conclusion has dropped citation Art. {art_num}")

        # Process reasoning_chain steps
        for step in repaired_ir.get("reasoning_chain", []):
            step_citations = step.get("citations", {})
            for art_num, check in checks_by_article.items():
                if find_article_in_citations(step_citations, art_num):
                    if check.mismatch_action == MismatchAction.REPAIRED.value:
                        step["text"] = check.repaired_text
                        step["_repaired"] = True
                    elif check.mismatch_action in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value):
                        step["_dropped"] = True

        # Add repair metadata
        repaired_ir["_repair_metadata"] = {
            "total_repaired": sum(1 for c in citation_checks if c.mismatch_action == MismatchAction.REPAIRED.value),
            "total_dropped": sum(1 for c in citation_checks if c.mismatch_action in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value)),
        }

        return repaired_ir

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
