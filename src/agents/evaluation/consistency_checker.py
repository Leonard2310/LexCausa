"""
Consistency checking and repair for reasoning chains.

Verifies that citations in the reasoning chain match the knowledge base,
handles mismatches (repair or drop), and regenerates chains via LLM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402

from ..aspic_formatter import AspicFormatter  # noqa: E402
from ..tools.neo4j_tools import get_driver  # noqa: E402
from .models import CitationCheck, ConsistencyReport, MismatchAction  # noqa: E402


class ConsistencyMixin:
    """Mixin providing consistency-checking methods to the evaluator."""

    def _verify_statute_in_neo4j(
        self, article_num: str, domain: str, citation_str: str = ""
    ) -> tuple[bool, str]:
        """
        Verify if an article exists in Neo4j and return its text.

        Args:
            article_num: The article number (e.g., "1223")
            domain: Legal domain ("CIVILE", "PENALE", or "ENTRAMBI")
            citation_str: Original citation string (e.g., "Art. 52 c.p.") used
                          to disambiguate codice when domain is ENTRAMBI.

        Returns:
            Tuple of (exists: bool, text: str). Text is empty if not found.
        """
        driver = get_driver()

        # When domain is ENTRAMBI, try to determine the correct codice
        # from the citation string (e.g. "c.p." → penale, "c.c." → civile)
        if domain == "ENTRAMBI" and citation_str:
            citation_lower = citation_str.lower()
            if "c.p" in citation_lower or (
                "cod" in citation_lower and "pen" in citation_lower
            ):
                self._log(
                    f"      🔍 ENTRAMBI → detected c.p. in '{citation_str}', searching PENALE"
                )
                return self._verify_statute_in_neo4j(
                    article_num, "PENALE", citation_str
                )
            elif "c.c" in citation_lower or (
                "cod" in citation_lower and "civ" in citation_lower
            ):
                self._log(
                    f"      🔍 ENTRAMBI → detected c.c. in '{citation_str}', searching CIVILE"
                )
                return self._verify_statute_in_neo4j(
                    article_num, "CIVILE", citation_str
                )

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
            # ENTRAMBI without citation hint: try both codici
            found, text = self._verify_statute_in_neo4j(
                article_num, "PENALE", citation_str
            )
            if found:
                return found, text
            return self._verify_statute_in_neo4j(article_num, "CIVILE", citation_str)

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
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

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
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

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
    {full_text[:settings.truncation_chain_text]}
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
                            core_indicators += (
                                settings.cc_conclusion_bonus
                            )  # In conclusion = more important
                            self._log(
                                f"      🎯 Art. {article_num} cited in conclusion"
                            )

        # Count occurrences in full text
        pattern = rf"art(?:icolo)?\.?\s*{article_num}"
        occurrences = len(re.findall(pattern, full_text, re.IGNORECASE))
        if occurrences >= settings.cc_occurrence_threshold:
            core_indicators += 1
            self._log(
                f"      📊 Art. {article_num} appears {occurrences} times in text"
            )

        # Threshold: configurable via settings.cc_core_threshold
        is_core = core_indicators >= settings.cc_core_threshold
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
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

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
    "{original_context[:settings.truncation_context]}"

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
    # Precedent verification helpers
    # ------------------------------------------------------------------

    def _verify_precedent_in_neo4j(
        self, title: str, precedent_id: str | None = None
    ) -> tuple[bool, str]:
        """
        Verify if a precedent exists in Neo4j and return its summary.

        Prefers lookup by ``precedent_id`` (exact, no whitespace issues).
        Falls back to a case-insensitive, trim-safe title comparison so
        that trailing whitespace or casing differences do not cause false
        negatives.

        Returns:
            Tuple of (exists: bool, summary: str). Summary is empty if
            not found.
        """
        driver = get_driver()

        # --- Strategy 1: lookup by precedent_id (most reliable) ---
        if precedent_id:
            query_by_id = """
                MATCH (p:Precedent)
                WHERE p.precedent_id = $id
                RETURN p.precedent_id AS id,
                       p.title AS title,
                       p.summary AS summary
                LIMIT 1
            """
            try:
                with driver.session() as session:
                    result = session.run(query_by_id, parameters={"id": precedent_id})
                    record = result.single()
                    if record:
                        summary = record.get("summary", "") or ""
                        self._log(
                            f"      🗄️ Neo4j: precedent found by ID ({precedent_id}) - "
                            f"'{(record.get('title') or '')[:60]}...'"
                        )
                        return True, summary
            except Exception as e:
                self._log(f"⚠️ Neo4j precedent query by ID failed: {e}", "warning")

        # --- Strategy 2: fallback to title with trim() for whitespace safety ---
        query_by_title = """
            MATCH (p:Precedent)
            WHERE toLower(trim(p.title)) = toLower(trim($title))
            RETURN p.precedent_id AS id,
                   p.title AS title,
                   p.summary AS summary
            LIMIT 1
        """
        try:
            with driver.session() as session:
                result = session.run(query_by_title, parameters={"title": title})
                record = result.single()
                if record:
                    summary = record.get("summary", "") or ""
                    self._log(
                        f"      🗄️ Neo4j: precedent found by title - "
                        f"'{(record.get('title') or '')[:60]}...'"
                    )
                    return True, summary
                return False, ""
        except Exception as e:
            self._log(f"⚠️ Neo4j precedent query failed: {e}", "warning")
            return False, ""

    def _extract_cited_text_for_precedent(
        self, full_text: str, precedent_title: str
    ) -> str:
        """
        Extract the text the LLM cited around a precedent title.

        Looks for patterns like:
        - «Titolo del precedente», il quale stabilisce che [testo]
        - Come confermato in «Titolo del precedente», [testo]
        - ... «Titolo del precedente» ... (sentence containing the title)

        Returns the surrounding sentence/context or empty string.
        """
        title_lower = precedent_title.lower()
        text_lower = full_text.lower()

        idx = text_lower.find(title_lower)
        if idx == -1:
            self._log(
                f"      ⚠️ Precedent title not found in text: "
                f"'{precedent_title[:60]}...'"
            )
            return ""

        # Extract a window around the title mention (the full sentence)
        # Go backwards to find sentence start
        start = max(0, idx - 200)
        end = min(len(full_text), idx + len(precedent_title) + 300)

        # Try to find sentence boundaries
        window = full_text[start:end]

        # Find the sentence containing the title
        sentences = re.split(r"(?<=[.!?])\s+", window)
        for sentence in sentences:
            if title_lower in sentence.lower():
                cleaned = sentence.strip()
                if len(cleaned) >= 20:
                    self._log(
                        f"      🎯 Extracted precedent context: " f"'{cleaned[:80]}...'"
                    )
                    return cleaned

        # Fallback: return the raw window
        raw = window.strip()
        if len(raw) >= 20:
            return raw
        return ""

    def _handle_precedent_mismatch(
        self,
        check: CitationCheck,
        precedent_title: str,
        cited_text: str,
        db_summary: str,
        full_text: str,
        report: ConsistencyReport,
    ) -> None:
        """
        Handle a precedent citation mismatch.

        Simpler than the statute flow because precedents are always
        treated as *pertinent* (they were explicitly cited by the LLM)
        and repair always uses the DB summary.

        Flow:
        1. Verify mismatch with LLM
        2. If confirmed → repair by rewriting with correct DB summary
        3. If repair fails → drop the citation
        """
        self._log(
            f"      🔧 Handling precedent mismatch for "
            f"'{precedent_title[:50]}...'..."
        )

        # Step 1: Verify mismatch with LLM (reuse the existing method
        # but with "Precedent" context instead of article number)
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )
            system_prompt = (
                "You are an expert in Italian law. Determine whether two "
                "descriptions of a court precedent are LOGICALLY EQUIVALENT "
                "or DIFFERENT.\n\n"
                "EQUIVALENT: same legal holding, even with different wording.\n"
                "DIFFERENT: different holding, added/omitted elements, "
                "changed meaning.\n\n"
                "Respond ONLY with: EQUIVALENTI or DIVERSI"
            )
            user_prompt = (
                f'CITED TEXT (from reasoning):\n"{cited_text}"\n\n'
                f'OFFICIAL SUMMARY (from database):\n"{db_summary[:1500]}"\n\n'
                f"Are the two texts EQUIVALENTI or DIVERSI?"
            )
            response = resilient_chat_call(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            answer = response.content.strip().upper()
            llm_confirmed = "DIVERSI" in answer
        except Exception as e:
            self._log(
                f"      ⚠️ LLM precedent mismatch check failed: {e}",
                "warning",
            )
            llm_confirmed = True

        check.llm_mismatch_confirmed = llm_confirmed

        if not llm_confirmed:
            self._log(
                "      ✅ LLM says precedent texts are equivalent "
                "- treating as match"
            )
            check.mismatch_action = MismatchAction.MATCH.value
            check.text_match = True
            check.llm_validated = True
            report.text_matches += 1
            report.text_mismatches -= 1
            return

        # Step 2: Attempt repair — rewrite the passage with the DB summary
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )
            system_prompt = (
                "You are an expert in Italian law. Rewrite a passage that "
                "cites a court precedent using EXCLUSIVELY the official "
                "summary provided.\n\n"
                "RULES:\n"
                "1. Include a VERBATIM QUOTE of at least 15 words from the "
                "official summary enclosed in «»\n"
                "2. Do not add concepts not in the official summary\n"
                "3. Write in Italian\n"
                "4. Output ONLY the rewritten text"
            )
            user_prompt = (
                f"PRECEDENT: {precedent_title}\n\n"
                f'OFFICIAL SUMMARY:\n"{db_summary[:2000]}"\n\n'
                f"ORIGINAL CONTEXT (to correct):\n"
                f'"{cited_text[:settings.truncation_context]}"\n\n'
                f"Rewrite using only the official summary."
            )
            response = resilient_chat_call(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            repaired_text = response.content.strip()

            # Validate: check verbatim quote
            quote_match = re.search(r"«([^»]+)»", repaired_text)
            if quote_match:
                quote = quote_match.group(1).lower().strip()
                if quote in db_summary.lower() and len(quote) >= 15:
                    check.mismatch_action = MismatchAction.REPAIRED.value
                    check.repaired_text = repaired_text
                    check.repair_success = True
                    check.details += " [REPAIRED with DB summary]"
                    report.repaired_citations += 1
                    self._log(
                        f"      ✅ Precedent REPAIRED successfully "
                        f"(quote {len(quote)} chars)"
                    )
                    return

            self._log("      ❌ Precedent repair FAILED: no valid quote")
        except Exception as e:
            self._log(f"      ⚠️ Precedent repair failed: {e}", "warning")

        # Repair failed → drop
        check.mismatch_action = MismatchAction.REPAIR_FAILED.value
        check.repair_success = False
        check.details += " [REPAIR FAILED - precedent citation unreliable]"
        report.dropped_citations += 1
        report.issues.append(
            f"Precedent '{precedent_title[:40]}': citation repair failed"
        )
        self._log(f"      ❌ Precedent '{precedent_title[:40]}' repair FAILED")

    def _check_consistency(
        self,
        agent: str,
        reasoning_chain: list[str],
        raw_response: str,
        domain: str,
        aspic_ir: dict | None = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
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

        def _emit_citation_progress(check_obj: CitationCheck) -> None:
            if not progress_callback:
                return
            try:
                progress_callback(
                    {
                        "agent": agent,
                        "check": check_obj.to_dict(),
                        "totals": {
                            "processed": report.total_citations,
                            "valid": report.valid_citations,
                            "invalid": report.invalid_citations,
                            "text_matches": report.text_matches,
                            "text_mismatches": report.text_mismatches,
                            "repaired": report.repaired_citations,
                            "dropped": report.dropped_citations,
                            "expected_total": expected_total_checks,
                        },
                    }
                )
            except Exception:
                # Progress callback failures must not affect consistency logic.
                pass

        # Combine chain and raw response for extraction
        full_text = "\n".join(reasoning_chain) + "\n" + raw_response

        # Extract statute citations from the text
        statute_citations = self._extract_statute_citations(full_text)
        self._log(
            f"📜 [{agent}] Found {len(statute_citations)} statute citations to verify"
        )

        # Prepare precedent candidates early so progress events can expose
        # an accurate total expected checks count for live progress bars.
        precedent_titles_in_ir: list[dict] = []
        if aspic_ir:
            for p in aspic_ir.get("sources", {}).get("precedents", []):
                title = (p.get("title") or "").strip()
                if title:
                    precedent_titles_in_ir.append(p)

        # Also collect titles from precedent_nodes (richer data)
        prec_node_map: dict[str, dict] = {}
        if aspic_ir:
            for pn in aspic_ir.get("precedent_nodes", []):
                title = (pn.get("title") or "").strip()
                if title:
                    prec_node_map[title.lower()] = pn

        # Build a list of all known precedent titles (from sources)
        # and a title → precedent_id map for reliable Neo4j lookup
        known_prec_titles: list[str] = []
        known_prec_titles_seen: set[str] = set()
        prec_id_by_title: dict[str, str] = {}
        for p in precedent_titles_in_ir:
            title = (p.get("title") or "").strip()
            title_key = title.lower()
            if title and title_key not in known_prec_titles_seen:
                known_prec_titles_seen.add(title_key)
                known_prec_titles.append(title)
                prec_id = (p.get("precedent_id") or "").strip()
                if prec_id:
                    prec_id_by_title[title_key] = prec_id

        # Enrich map from precedent_nodes (they carry precedent_id too)
        for pn in prec_node_map.values():
            title = (pn.get("title") or "").strip().lower()
            prec_id = (pn.get("precedent_id") or "").strip()
            if title and prec_id and title not in prec_id_by_title:
                prec_id_by_title[title] = prec_id

        # Find which precedents are actually cited in the text
        full_text_lower = full_text.lower()
        cited_prec_titles = [
            t for t in known_prec_titles if t.lower() in full_text_lower
        ]
        unique_statute_articles = {
            m.group(1) for c in statute_citations if (m := re.search(r"(\d{1,4})", c))
        }
        expected_total_checks = len(unique_statute_articles) + len(cited_prec_titles)

        # Track already verified articles to avoid duplicates
        verified_articles: set[str] = set()
        verified_precedent_titles: set[str] = set()

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
            found, db_text = self._verify_statute_in_neo4j(
                article_num, domain, citation
            )

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
                    text_match = similarity >= settings.cc_text_match_threshold

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
            _emit_citation_progress(check)

        # ----- Precedent citation verification -----
        # `cited_prec_titles` and `prec_id_by_title` are precomputed above.

        if cited_prec_titles:
            self._log(
                f"📜 [{agent}] Found {len(cited_prec_titles)} precedent "
                f"citations to verify"
            )
        else:
            self._log(f"📜 [{agent}] No precedent citations detected in text")

        for prec_title in cited_prec_titles:
            if prec_title.lower() in verified_precedent_titles:
                continue
            verified_precedent_titles.add(prec_title.lower())

            # Verify existence in Neo4j and get summary
            # Use precedent_id when available for reliable lookup
            prec_id = prec_id_by_title.get(prec_title.lower())
            found, db_summary = self._verify_precedent_in_neo4j(
                prec_title, precedent_id=prec_id
            )

            check = CitationCheck(
                citation=f"Prec: {prec_title[:80]}",
                found_in_kb=found,
                source_type="precedent",
            )

            if found:
                report.valid_citations += 1
                self._log(f"   ✅ Precedent '{prec_title[:50]}...' -> EXISTS in Neo4j")

                # Extract cited text around the precedent mention
                cited_text = self._extract_cited_text_for_precedent(
                    full_text, prec_title
                )

                if cited_text and db_summary:
                    similarity = self._compute_text_similarity(cited_text, db_summary)
                    text_match = similarity >= settings.cc_text_match_threshold

                    check.text_verified = True
                    check.text_match = text_match
                    check.text_similarity = similarity
                    check.cited_text = cited_text
                    check.db_text_preview = db_summary[:500]

                    if text_match:
                        report.text_matches += 1
                        check.mismatch_action = MismatchAction.MATCH.value
                        check.details = (
                            f"Precedent verified in Neo4j, "
                            f"text match: {similarity:.0%}"
                        )
                        self._log(
                            f"      📝 Text similarity: " f"{similarity:.0%} ✅ MATCH"
                        )
                    else:
                        report.text_mismatches += 1
                        check.details = (
                            f"Precedent verified in Neo4j, "
                            f"text mismatch: {similarity:.0%}"
                        )
                        self._log(
                            f"      📝 Text similarity: " f"{similarity:.0%} ⚠️ MISMATCH"
                        )
                        report.issues.append(
                            f"Precedent '{prec_title[:40]}': "
                            f"cited text differs from DB "
                            f"(similarity: {similarity:.0%})"
                        )
                        self._handle_precedent_mismatch(
                            check=check,
                            precedent_title=prec_title,
                            cited_text=cited_text,
                            db_summary=db_summary,
                            full_text=full_text,
                            report=report,
                        )
                elif not cited_text and db_summary:
                    # Title is present but no surrounding text extracted.
                    # Treat as match (the LLM cited it correctly by title).
                    check.text_verified = True
                    check.text_match = True
                    check.text_similarity = 1.0
                    check.cited_text = prec_title
                    check.db_text_preview = db_summary[:500]
                    check.mismatch_action = MismatchAction.MATCH.value
                    check.details = (
                        "Precedent verified in Neo4j, "
                        "cited by exact title (no surrounding text)"
                    )
                    report.text_matches += 1
                    self._log("      📝 Precedent cited by exact title ✅ MATCH")
                else:
                    check.details = (
                        "Precedent verified in Neo4j, "
                        "no text available for verification"
                    )
            else:
                report.invalid_citations += 1
                check.details = "Precedent not found in Neo4j"
                report.issues.append(
                    f"Precedent '{prec_title[:40]}' not found in Neo4j"
                )
                self._log(
                    f"   ❌ Precedent '{prec_title[:50]}...' " f"-> NOT FOUND in Neo4j"
                )

            report.citation_checks.append(check)
            report.total_citations += 1
            _emit_citation_progress(check)

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

        report.consistency_score = (
            settings.cc_consistency_existence_weight * existence_score
            + settings.cc_consistency_text_weight * text_score
        )

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
    # Threshold: below this text_similarity the step's reasoning is
    # likely built on a completely wrong article and a simple text
    # replacement won't fix the logical incoherence → the step must
    # be regenerated by an LLM.
    # ------------------------------------------------------------------
    _SEVERE_MISMATCH_THRESHOLD = 0.40

    def _regenerate_reasoning_chain_with_llm(
        self,
        original_chain: str,
        citation_checks: list[CitationCheck],
        agent_name: str,
    ) -> str:
        """Apply **hybrid** repairs to the reasoning chain text.

        The strategy adapts to the severity of each mismatch:

        **Surgical (no LLM)**
        * *No text cited*  →  inject ``«DB text»`` after the article
          mention.
        * *Text mismatch ≥ 40 %*  →  in-place replacement of the
          incorrect passage with the corrected ``repaired_text``.
        * *Dropped / repair-failed*  →  mark with
          ``[Citation removed - unreliable source]``.

        **Per-step LLM regeneration**
        * *Text mismatch < 40 %*  →  the article was probably confused
          with another one (e.g. Art. 23 cited with Art. 43's text).
          The entire reasoning step is extracted, sent to an LLM with
          the correct DB text, and rewritten **individually**.

        This preserves the original structure (headers, numbered chain)
        while fixing steps whose reasoning was built on a wrong article.

        Args:
            original_chain: The original reasoning chain text
            citation_checks: List of CitationCheck objects with mismatch actions
            agent_name: Name of the agent ("reasoner" or "counter_reasoner")

        Returns:
            The repaired chain text.
        """
        surgical_count = 0
        llm_regen_count = 0
        dropped_count = 0
        chain = original_chain

        for check in citation_checks:
            # --- REPAIRED citations ---
            if check.mismatch_action == MismatchAction.REPAIRED.value:
                # Precedent citations: surgical replace of cited context
                if check.source_type == "precedent":
                    if check.cited_text and check.repaired_text:
                        if check.cited_text in chain:
                            chain = chain.replace(
                                check.cited_text, check.repaired_text, 1
                            )
                            surgical_count += 1
                        else:
                            idx = chain.lower().find(check.cited_text.lower())
                            if idx >= 0:
                                chain = (
                                    chain[:idx]
                                    + check.repaired_text
                                    + chain[idx + len(check.cited_text) :]
                                )
                                surgical_count += 1
                    continue

                # Statute citations (original logic)
                if not check.cited_text and check.repaired_text:
                    # Case A: no text was cited → inject DB text (surgical)
                    chain = self._inject_text_after_article(
                        chain, check.citation, check.repaired_text
                    )
                    surgical_count += 1

                elif check.cited_text and check.repaired_text:
                    # Decide: surgical or per-step LLM?
                    if check.text_similarity >= self._SEVERE_MISMATCH_THRESHOLD:
                        # Case B: text was reasonably close → surgical replace
                        if check.cited_text in chain:
                            chain = chain.replace(
                                check.cited_text, check.repaired_text, 1
                            )
                            surgical_count += 1
                        else:
                            idx = chain.lower().find(check.cited_text.lower())
                            if idx >= 0:
                                chain = (
                                    chain[:idx]
                                    + check.repaired_text
                                    + chain[idx + len(check.cited_text) :]
                                )
                                surgical_count += 1
                            else:
                                chain = self._inject_text_after_article(
                                    chain, check.citation, check.repaired_text
                                )
                                surgical_count += 1
                    else:
                        # Case C: severe mismatch → regenerate the step
                        chain = self._regenerate_single_step(chain, check, agent_name)
                        llm_regen_count += 1

            # --- DROPPED / REPAIR_FAILED citations ---
            elif check.mismatch_action in (
                MismatchAction.DROPPED.value,
                MismatchAction.REPAIR_FAILED.value,
            ):
                if check.cited_text and check.cited_text in chain:
                    chain = chain.replace(
                        check.cited_text,
                        "[Citation removed - unreliable source]",
                        1,
                    )
                dropped_count += 1

        total = surgical_count + llm_regen_count + dropped_count
        if total == 0:
            self._log(f"   ✅ [{agent_name}] No repairs needed - chain unchanged")
            return original_chain

        self._log(
            f"   🔄 [{agent_name}] Repair complete: "
            f"{surgical_count} surgical, {llm_regen_count} LLM-regenerated, "
            f"{dropped_count} dropped"
        )
        return chain

    # ------------------------------------------------------------------
    # Per-step LLM regeneration (only for severe mismatches)
    # ------------------------------------------------------------------

    def _regenerate_single_step(
        self,
        chain: str,
        check: CitationCheck,
        agent_name: str,
    ) -> str:
        """Find and regenerate the specific step that cites *check.citation*.

        Locates the numbered chain step (e.g. ``3. L'Art. 23 c.p. …``)
        that contains the problematic article, sends **only that step**
        to the LLM with the correct DB text, and replaces it in-place.

        If the step cannot be isolated or the LLM call fails, falls back
        to a surgical text injection.
        """
        # Extract article number
        m = re.search(r"(\d{1,4})", check.citation)
        if not m:
            return self._inject_text_after_article(
                chain, check.citation, check.repaired_text
            )
        article_num = m.group(1)

        # Find the numbered step containing this article in the chain.
        # Steps are formatted as "N. <text>\n" where N is 1, 2, 3…
        # We look for a line starting with a number, containing our article.
        step_pattern = re.compile(
            rf"^(\d+)\.\s+(.*?Art(?:icolo)?\.?\s*{article_num}\b.*?)$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = step_pattern.search(chain)
        if not match:
            self._log(
                f"      ⚠️ [{agent_name}] Could not isolate step for "
                f"{check.citation}; falling back to injection"
            )
            return self._inject_text_after_article(
                chain, check.citation, check.repaired_text
            )

        step_number = match.group(1)
        original_step_text = match.group(2)
        full_match_text = match.group(0)  # "N. <text>"

        self._log(
            f"      🔬 [{agent_name}] Regenerating step {step_number} "
            f"for {check.citation} (similarity was {check.text_similarity:.0%})"
        )

        # Ask LLM to rewrite only this step
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

            system_prompt = (
                "You are an expert Italian jurist. You must rewrite a SINGLE "
                "reasoning step from a legal argument.\n\n"
                "You will receive:\n"
                "1. The ORIGINAL step text (which contains an incorrect "
                "normative citation)\n"
                "2. The CORRECT official text of the article from the database\n\n"
                "RULES:\n"
                "1. Rewrite the step so that it correctly uses the OFFICIAL "
                "text of the article\n"
                "2. Adjust the legal reasoning to be coherent with the "
                "CORRECT article text\n"
                "3. Include a verbatim quote from the official text in «»\n"
                "4. Keep approximately the same length and depth\n"
                "5. Write in Italian\n"
                "6. Output ONLY the rewritten step text, nothing else\n"
                "7. Do NOT include the step number (e.g. do NOT start "
                'with "3.")'
            )

            user_prompt = (
                f"ARTICLE: {check.citation}\n\n"
                f"CORRECT OFFICIAL TEXT:\n"
                f'"{check.db_text_preview}"\n\n'
                f"ORIGINAL STEP (to rewrite):\n"
                f'"{original_step_text[:800]}"\n\n'
                f"Rewrite this step in Italian using the correct article text."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            new_step_text = response.content.strip()

            # Remove any leading "N." the LLM might have added
            new_step_text = re.sub(r"^\d+\.\s*", "", new_step_text)

            if len(new_step_text) < 30:
                self._log(
                    f"      ⚠️ [{agent_name}] LLM step regen too short; "
                    f"falling back to injection"
                )
                return self._inject_text_after_article(
                    chain, check.citation, check.repaired_text
                )

            # Replace only this step in the chain
            new_full_step = f"{step_number}. {new_step_text}"
            chain = chain.replace(full_match_text, new_full_step, 1)

            self._log(
                f"      ✅ [{agent_name}] Step {step_number} regenerated "
                f"({len(new_step_text)} chars)"
            )
            return chain

        except Exception as e:
            self._log(
                f"      ⚠️ [{agent_name}] Step regen failed: {e}; "
                f"falling back to injection",
                "warning",
            )
            return self._inject_text_after_article(
                chain, check.citation, check.repaired_text
            )

    @staticmethod
    def _inject_text_after_article(
        chain: str, citation: str, repaired_text: str
    ) -> str:
        """Insert a normative-text annotation after the first mention of *citation*.

        For example, if ``citation`` is ``"Art. 43 c.p."`` and the chain
        contains ``"… Art. 43 c.p. …"``, this method turns it into
        ``"… Art. 43 c.p. [Testo: «…»] …"``.

        The ``repaired_text`` is truncated to a reasonable length to
        avoid bloating the chain.
        """
        # Extract article number from citation (e.g. "Art. 43 c.p." → "43")
        m = re.search(r"(\d{1,4})", citation)
        if not m:
            return chain
        article_num = m.group(1)

        # Build a regex that matches "Art. 43 c.p." or "Art. 43 c.c."
        pattern = re.compile(
            rf"(Art(?:icolo)?\.?\s*{article_num}\s*c\.?\s*[cp]\.?)",
            re.IGNORECASE,
        )

        match = pattern.search(chain)
        if not match:
            return chain

        # Truncate repaired text to first 200 chars if very long
        snippet = repaired_text[:200].rstrip()
        if len(repaired_text) > 200:
            snippet += "…"

        insert_pos = match.end()
        annotation = f", il quale dispone che «{snippet}»,"
        chain = chain[:insert_pos] + annotation + chain[insert_pos:]
        return chain

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

        Preserves the original step order by using indexed extraction
        rather than relying on AspicFormatter's re-parsing which may
        reorder steps.

        Args:
            aspic_ir: Original ASPIC IR structure (used for fallback/metadata)
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

        # Parse the repaired chain text - extract steps in order.
        # NOTE: we intentionally do NOT call _sanitize_reasoning_chain here.
        # That method uses keyword filtering ("precedent"/"precedente") and
        # removes valid reasoning steps that happen to cite precedents when
        # the precedents list is empty.  _build_chain_steps inside
        # AspicFormatter already handles noise/dedup without destroying
        # legitimate steps.
        reasoning_chain = self._extract_reasoning_chain_ordered(repaired_chain_text)

        # Fallback: if the repair LLM dropped the numbered chain format,
        # the parser may extract < 2 steps. In that case, reuse the
        # original IR's reasoning_chain so AQA can still build links.
        original_chain_steps = aspic_ir.get("reasoning_chain", [])
        if len(reasoning_chain) < 2 and len(original_chain_steps) >= 2:
            self._log(
                f"   ⚠️ [{role.upper()}] Repaired chain produced only "
                f"{len(reasoning_chain)} parsed step(s); falling back to "
                f"original chain ({len(original_chain_steps)} steps)"
            )
            reasoning_chain = [
                s.get("text", s) if isinstance(s, dict) else s
                for s in original_chain_steps
            ]

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

    def _extract_reasoning_chain_ordered(self, text: str) -> list[str]:
        """Extract reasoning chain steps preserving their original numbered order.

        Parses numbered steps (1. 2. 3. ...) from the chain section and
        returns them in their original order, not re-sorted.
        """
        steps: list[str] = []

        # Find the chain section
        chain_start = text.lower().find("catena di ragionamento")
        if chain_start == -1:
            chain_start = text.lower().find("reasoning chain")
        if chain_start == -1:
            chain_start = 0

        chain_text = text[chain_start:]

        # Pattern for numbered steps: "1. text" or "1) text"
        step_pattern = re.compile(
            r"^\s*(\d+)[.)\s]+(.+?)(?=^\s*\d+[.)\s]|\Z)", re.MULTILINE | re.DOTALL
        )

        # Extract all numbered steps
        numbered_steps: list[tuple[int, str]] = []
        for match in step_pattern.finditer(chain_text):
            step_num = int(match.group(1))
            step_text = match.group(2).strip()
            # Clean up the step text
            step_text = re.sub(r"\s+", " ", step_text)
            if step_text and len(step_text) > 10:
                numbered_steps.append((step_num, step_text))

        # Sort by step number to preserve original order
        numbered_steps.sort(key=lambda x: x[0])

        # Extract just the text
        steps = [text for _, text in numbered_steps]

        return steps

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
