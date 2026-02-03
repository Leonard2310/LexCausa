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

from .base import AgentConfig, BaseAgent
from .tools.neo4j_tools import get_driver


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

        # Build dialectical tree
        reasoner_ir = reasoner_output.get("aspic_ir")
        counter_ir = counter_reasoner_output.get("aspic_ir")
        dialectical_tree = {}
        if reasoner_ir or counter_ir:
            dialectical_tree = {
                "schema": "aspic_ir_bundle_v1",
                "reasoner": reasoner_ir,
                "counter": counter_ir,
            }

        # Generate summary
        summary = self._generate_consistency_summary(
            reasoner_report, counter_report
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
            summary=summary,
            polished_response="",
            dialectical_tree=dialectical_tree,
        )

    # ------------------------------------------------------------------
    # Neo4j Verification
    # ------------------------------------------------------------------
    def _verify_statute_in_neo4j(self, article_num: str, domain: str) -> tuple[bool, str]:
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
                        self._log(f"      🗄️ Neo4j: Art. {article_num} found - '{titolo[:50]}...'")
                        self._log(f"      📄 DB text preview: '{testo[:100]}...'")
                    else:
                        self._log(f"      🗄️ Neo4j: Art. {article_num} found but NO TEXT in DB")
                    return True, testo
                return False, ""
        except Exception as e:
            self._log(f"⚠️ Neo4j query failed: {e}", "warning")
            return False, ""

    def _extract_cited_text_for_article(self, full_text: str, article_num: str, aspic_ir: dict | None = None) -> str:
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
        # Pattern 1: "L'Art. 1223 c.c. stabilisce/prevede/limita che..."
        # Captures the verb + "che" + text until period
        pattern1 = rf"[Ll][''']?Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?\s+(?:stabilisce|prevede|limita|dispone|sancisce|afferma)\s+(?:che\s+)?(.+?)(?:\.|$)"
        match = re.search(pattern1, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'\s+', ' ', extracted)
            if len(extracted) >= 20:
                self._log(f"      🎯 Found 'Art. {article_num} stabilisce/prevede...' pattern")
                return extracted

        # Pattern 2: "Art. 1223 c.c. - [testo]"
        pattern2 = rf"Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?\s*[\-:]\s*(.+?)(?:\.|$)"
        match = re.search(pattern2, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'\s+', ' ', extracted)
            # Avoid if it contains other article references (means it's a list)
            if not re.search(r'Art(?:icolo)?\.?\s*\d{3,4}', extracted, re.IGNORECASE):
                if len(extracted) >= 20:
                    self._log(f"      🎯 Found 'Art. {article_num} - ...' pattern")
                    return extracted

        # Pattern 3: "Secondo l'Art. 1223 c.c., [testo]"
        pattern3 = rf"[Ss]econdo\s+[Ll][''']?Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?,?\s+(.+?)(?:\.|$)"
        match = re.search(pattern3, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'\s+', ' ', extracted)
            if len(extracted) >= 20:
                self._log(f"      🎯 Found 'Secondo l'Art. {article_num}...' pattern")
                return extracted

        # Pattern 4: "(Art. 1223 c.c.)" in parentheses after text - capture text before
        pattern4 = rf"([^.]+?)\s*\(Art(?:icolo)?\.?\s*{article_num}\s*(?:c\.?\s*c\.?|c\.?\s*p\.?)?\)"
        match = re.search(pattern4, full_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Take last sentence before the article reference
            sentences = extracted.split('.')
            if sentences:
                last_sentence = sentences[-1].strip()
                if len(last_sentence) >= 20:
                    self._log(f"      🎯 Found text before '(Art. {article_num})' pattern")
                    return last_sentence

        self._log(f"      ⚠️ No text found for Art. {article_num}")
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
            self._log(f"      ⚠️ ASPIC IR is empty or None")
            return ""

        self._log(f"      🔍 Searching ASPIC IR for Art. {article_num}")

        # Helper to check if an article is in citations
        def article_in_citations(citations: dict, art_num: str) -> bool:
            for statute in citations.get("statutes", []):
                if str(statute.get("articolo", "")).strip() == art_num:
                    return True
            return False

        # Helper to extract text specific to an article from a larger text block
        def extract_article_specific_text(full_text: str, art_num: str) -> str:
            """Extract the portion of text that refers specifically to this article."""
            # Pattern to find text associated with specific article
            patterns = [
                # Art. 1223: text or Art. 1223 - text
                rf"art(?:icolo)?\.?\s*{art_num}[^:.\n]*[:.-]\s*([^.]+(?:\.[^.]+)?)",
                # Art. 1223 c.c. che prevede/stabilisce che...
                rf"art(?:icolo)?\.?\s*{art_num}[^,]*(?:che\s+)?(?:prevede|stabilisce|dispone|sancisce)\s+(?:che\s+)?([^.]+(?:\.[^.]+)?)",
                # art. 1223 (testo tra parentesi)
                rf"art(?:icolo)?\.?\s*{art_num}[^(]*\(([^)]+)\)",
            ]
            for pattern in patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
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
                        self._log(f"      📋 Found norm block with {num_statutes} statute(s)")
                        
                        if num_statutes == 1:
                            # Only this article, return full text
                            self._log(f"      ✅ Single article block - using full text")
                            return full_text
                        else:
                            # Multiple articles, try to extract specific portion
                            specific_text = extract_article_specific_text(full_text, article_num)
                            if specific_text:
                                self._log(f"      ✅ Extracted specific text for Art. {article_num}")
                                return specific_text
                            # If can't extract specific, still return full text with warning
                            self._log(f"      ⚠️ Multi-article block, returning full text")
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
                        self._log(f"      ✅ Found in reasoning chain (single article)")
                        return text
                    else:
                        # Try to extract specific portion
                        specific_text = extract_article_specific_text(text, article_num)
                        if specific_text:
                            self._log(f"      ✅ Extracted from reasoning chain step")
                            return specific_text

        # Fallback: get title from sources
        for statute in aspic_ir.get("sources", {}).get("statutes", []):
            if str(statute.get("articolo", "")).strip() == article_num:
                title = statute.get("title", "").strip()
                if title:
                    self._log(f"      📚 Using title from sources: {title[:50]}...")
                    return f"[Titolo] {title}"

        self._log(f"      ❌ No text found in ASPIC IR for Art. {article_num}")
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
        cited_words = set(w for w in re.findall(r'\b\w{4,}\b', cited_lower))
        db_words = set(w for w in re.findall(r'\b\w{4,}\b', db_lower))

        if not cited_words:
            return 0.0

        # Calculate Jaccard-like similarity
        common_words = cited_words & db_words
        similarity = len(common_words) / len(cited_words)

        return min(similarity, 1.0)

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
        self._log(f"📜 [{agent}] Found {len(statute_citations)} statute citations to verify")

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

                # Extract cited text from ASPIC IR (preferred) or reasoning chain (fallback)
                cited_text = self._extract_cited_text_for_article(full_text, article_num, aspic_ir)

                if cited_text:
                    self._log(f"      📖 Cited text extracted: '{cited_text[:80]}...'")

                if cited_text and db_text:
                    # Perform text verification
                    similarity = self._compute_text_similarity(cited_text, db_text)
                    text_match = similarity >= 0.5  # Threshold for considering a match

                    check.text_verified = True
                    check.text_match = text_match
                    check.text_similarity = similarity
                    check.cited_text = cited_text[:200]  # Truncate for storage
                    check.db_text_preview = db_text[:200]  # Preview of DB text

                    if text_match:
                        report.text_matches += 1
                        check.details = f"Verified in Neo4j ({domain}), text match: {similarity:.0%}"
                        self._log(f"      📝 Text similarity: {similarity:.0%} ✅ MATCH")
                    else:
                        report.text_mismatches += 1
                        check.details = f"Verified in Neo4j ({domain}), text mismatch: {similarity:.0%}"
                        self._log(f"      📝 Text similarity: {similarity:.0%} ⚠️ MISMATCH")
                        report.issues.append(
                            f"Art. {article_num}: cited text differs from DB (similarity: {similarity:.0%})"
                        )
                else:
                    check.details = f"Verified in Neo4j ({domain}), no text cited for verification"
                    if not cited_text:
                        self._log(f"      📝 No cited text found for verification")
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
                code = "c.c." if "c" in code_part.lower() and "p" not in code_part.lower() else "c.p."
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
        lines.append(f"- Score di consistenza: {counter_report.consistency_score:.2%}")
        if counter_report.issues:
            lines.append(f"- Problemi: {len(counter_report.issues)}")

        lines.append("")

        return "\n".join(lines)

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
