"""Prompt registry for LexCausa.

This module contains:
- prompt identifiers (PromptKey)
- prompt templates (PROMPTS)
- helper functions (get_prompt / render_prompt)
"""

from __future__ import annotations

import re
from enum import StrEnum


class PromptKey(StrEnum):
    """Prompt identifiers used by the registry."""

    CLAIM_CLASSIFIER_SYSTEM = "claim_classifier.system"
    CLAIM_CLASSIFIER_TAXONOMY_USER = "claim_classifier.taxonomy_user"
    ROUTER_SYSTEM = "router.system"
    ROUTER_USER = "router.user"
    BASE_FILTER_RELEVANT_STATUTES = "base.filter_relevant_statutes"
    BASE_EXTRACT_LEGAL_CONTEXT = "base.extract_legal_context"
    BASE_FILTER_APPLICABLE_STATUTES = "base.filter_applicable_statutes"
    BASE_FILTER_PRECEDENTS = "base.filter_precedents"
    BASE_FACT_LOCK_CHECK = "base.fact_lock_check"
    LEGAL_SEARCH_QUERY_TERMS_SYSTEM = "legal_search.query_terms_system"
    LEGAL_SEARCH_QUERY_TERMS_USER = "legal_search.query_terms_user"
    NEO4J_TOOLS_EXTRACT_KEYWORDS = "neo4j_tools.extract_keywords"
    TAXONOMY_TOOLS_CLASSIFICATION = "taxonomy_tools.classification"
    TAXONOMY_TOOLS_CAUSALITY_CLAIM_PROMPT = "taxonomy_tools.causality_claim_prompt"
    TAXONOMY_TOOLS_FILTER_NORM = "taxonomy_tools.filter_norm"
    REASONER_SYSTEM = "reasoner.system"
    REASONER_CLASSIFY_CAUSALITY = "reasoner.classify_causality"
    REASONER_GENERATE_PLAN = "reasoner.generate_plan"
    REASONER_SUPPORT_STEP = "reasoner.support_step"
    REASONER_SUPPORT_PLAN_REWRITE = "reasoner.support_plan_rewrite"
    REASONER_SEMANTIC_REDUNDANCY = "reasoner.semantic_redundancy"
    REASONER_SUPPORT_CONCLUSION_REWRITE = "reasoner.support_conclusion_rewrite"
    REASONER_GENERATE_CONCLUSION = "reasoner.generate_conclusion"
    REASONER_REASONING_WITH_CONTEXT = "reasoner.reasoning_with_context"
    COUNTER_REASONER_SYSTEM = "counter_reasoner.system"
    COUNTER_REASONER_PICK_ATTACKS = "counter_reasoner.pick_attacks"
    COUNTER_REASONER_GENERATE_PLAN = "counter_reasoner.generate_plan"
    COUNTER_REASONER_STEP_PROMPT = "counter_reasoner.step_prompt"
    COUNTER_REASONER_SEMANTIC_REDUNDANCY = "counter_reasoner.semantic_redundancy"
    COUNTER_REASONER_ATTACK_ALIGNMENT = "counter_reasoner.attack_alignment"
    COUNTER_REASONER_STEP_OPPOSITION_CHECK = "counter_reasoner.step_opposition_check"
    COUNTER_REASONER_STANCE_REWRITE = "counter_reasoner.stance_rewrite"
    AQA_ENGINE_ATTACK_TYPE_SYSTEM = "aqa_engine.attack_type_system"
    AQA_ENGINE_ATTACK_TYPE_USER = "aqa_engine.attack_type_user"
    NLP_UTILS_NLI_SYSTEM = "nlp_utils.nli_system"
    NLP_UTILS_NLI_USER = "nlp_utils.nli_user"
    POLISHER_COUNTER_GATE = "polisher.counter_gate"
    CONSISTENCY_VERIFY_MISMATCH_SYSTEM = "consistency.verify_mismatch_system"
    CONSISTENCY_VERIFY_MISMATCH_USER = "consistency.verify_mismatch_user"
    CONSISTENCY_PERTINENCE_SYSTEM = "consistency.pertinence_system"
    CONSISTENCY_PERTINENCE_USER = "consistency.pertinence_user"
    CONSISTENCY_REPAIR_DB_SYSTEM = "consistency.repair_db_system"
    CONSISTENCY_REPAIR_DB_USER = "consistency.repair_db_user"
    CONSISTENCY_PRECEDENT_MISMATCH_SYSTEM = "consistency.precedent_mismatch_system"
    CONSISTENCY_PRECEDENT_MISMATCH_USER = "consistency.precedent_mismatch_user"
    CONSISTENCY_PRECEDENT_REPAIR_SYSTEM = "consistency.precedent_repair_system"
    CONSISTENCY_PRECEDENT_REPAIR_USER = "consistency.precedent_repair_user"
    CONSISTENCY_REGENERATE_STEP_SYSTEM = "consistency.regenerate_step_system"
    CONSISTENCY_REGENERATE_STEP_USER = "consistency.regenerate_step_user"


PROMPTS: dict[PromptKey, str] = {
    # ---------------------------------------------------------------------
    # Claim Classification & Routing
    # ---------------------------------------------------------------------
    PromptKey.CLAIM_CLASSIFIER_SYSTEM: """You are a legal-domain routing classifier for Italian law.

Your task is to assign a legal claim to the most relevant category
chosen ONLY from the provided taxonomy.

Rules:
- Output ONE category ID by default.
- Output MORE THAN ONE category ONLY if multiple categories are clearly and independently relevant.
- Output AT MOST 2 category IDs.
- If only one category applies, output ONLY one.
- Do NOT explain the decision.
- Do NOT add any text, symbols, or formatting.
- Do NOT cite articles or laws.
- Do NOT invent new categories.

You must follow these rules strictly.
If uncertain, prefer fewer categories.
The response language must be Italian.
""",
    PromptKey.CLAIM_CLASSIFIER_TAXONOMY_USER: """TAXONOMY

[[taxonomy_block]]

CLAIM
<<<
[[claim]]
>>>

Respond in Italian.""",
    PromptKey.ROUTER_SYSTEM: """You are a preliminary router for a legal causal reasoning system.
Your only task is to classify the DOMAIN of the claim:
- "CIVILE" if the claim pertains to civil law (contracts, torts, damages, compensation)
- "PENALE" if the claim pertains to criminal law (crimes, criminal liability, punishment)
- "AMMINISTRATIVO" if the claim pertains to administrative procedure/public administration acts
- "ENTRAMBI" if the claim involves multiple domains (civil/criminal/administrative)

Rules:
- Respond ONLY with compact JSON: {"domain": "CIVILE" | "PENALE" | "AMMINISTRATIVO" | "ENTRAMBI"}
- Do not add text, comments, or explanations.
- If uncertain, prefer "ENTRAMBI".
""",
    PromptKey.ROUTER_USER: """[[router_system]]

Claim:
\"\"\"[[claim]]\"\"\"

Domain options:
- CIVILE: civil liability, damages, breach of contract, tort liability
- PENALE: crimes, criminal liability, causal nexus between conduct and harmful event
- AMMINISTRATIVO: administrative procedure, access to documents, deadlines, motivation, defects of administrative acts (L. 241/1990)
- ENTRAMBI: cases involving more than one domain among civil, criminal, administrative

Respond with compact JSON.""",
    # ---------------------------------------------------------------------
    # Base Agent (shared retrieval filters)
    # ---------------------------------------------------------------------
    PromptKey.BASE_FILTER_RELEVANT_STATUTES: """Legal Claim:
"[[claim]]"

Article:
"[[article_number]] - [[article_title]] - [[article_desc]]"

Instruction:
Determine whether the main topic of the article is directly mentioned or implied in the claim.

Rules:
- Do NOT evaluate whether the article fully resolves the issue.
- Do NOT suggest any additional articles.
- Do NOT use external knowledge; only consider the claim and this article.
- Do NOT add explanations or comments.
- Answer YES in all cases with even indirect connection.
- Use NO only when the article is clearly about a different domain.
- If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.
No punctuation. No new lines. No extra spaces.
""",
    PromptKey.BASE_EXTRACT_LEGAL_CONTEXT: """You are a legal triage assistant.

Given this claim:
"[[claim]]"

Extract a compact legal context string (max 20 words) including:
- legal domain (criminal/civil/administrative/labour/commercial/etc.)
- party relationship (private-private, citizen-state, company-shareholders, etc.)
- procedural posture (investigation/trial/enforcement/contract dispute/etc.)

If uncertain, provide the most plausible generic context.

Respond with EXACTLY one short line and no extra text.""",
    PromptKey.BASE_FILTER_APPLICABLE_STATUTES: """Legal Situation:
"[[claim]]"

Legal Context: [[legal_context]]

Statute:
"[[article_number]] - [[article_title]]"
"[[article_text]]"

Question:
Does this statute APPLY to the legal situation described?

Evaluation Criteria:
1. Subject Scope: Does the statute apply to the TYPE of parties involved?
2. Substantive Scope: Does the statute regulate the LEGAL ISSUE at stake?
3. Temporal Scope: Is the statute relevant to the PROCEDURAL PHASE?

Rules:
- Answer YES only if the statute directly regulates THIS situation.
- Answer NO if it applies to a different:
  * relationship type
  * legal domain
  * offense class
  * procedural phase
- If uncertain but potentially on-point, answer YES.

Respond with EXACTLY one token: YES or NO.""",
    PromptKey.BASE_FILTER_PRECEDENTS: """You are a senior Italian legal expert.

CLAIM (the legal case under evaluation):
"[[claim]]"

PRECEDENT:
Title: "[[title]]"[[materia_line]]
Summary: "[[summary]]"

TASK — Decide whether a competent lawyer would cite this precedent
when arguing the above claim (either to support or to counter it).

Answer YES when ANY of the following is true:
1. The precedent addresses the SAME or a closely analogous legal
   question (e.g. same offence, same cause of action, same defence).
2. The precedent establishes a legal PRINCIPLE (causation test,
   evidentiary standard, constitutional interpretation, procedural
   rule) that directly applies to the claim.
3. The factual scenario of the precedent is substantially similar to
   the claim, making the ruling transferable.

Answer NO when:
- The precedent concerns a completely unrelated area of law with no
  transferable principle (e.g. tax evasion vs. divorce).
- The connection is merely superficial (shared keywords but different
  legal substance).

If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.""",
    PromptKey.BASE_FACT_LOCK_CHECK: """You are a factual consistency checker for legal reasoning steps.

CLAIM (all explicit facts are fixed and true):
\"\"\"[[claim]]\"\"\"

CANDIDATE STEP:
\"\"\"[[candidate_step]]\"\"\"

Task:
Decide whether the candidate step CONTRADICTS any explicit fact stated in the claim.

Important rules:
- The step MAY discuss legal qualification, applicability of norms, causal inference, or legal conclusions.
- The step MAY NOT deny, reverse, or weaken an explicit factual statement in the claim.
- If the claim reports a factual finding/perizia as given, the step cannot claim that the finding is absent or says the opposite.
- If doubtful, choose CONSISTENT unless there is a clear factual contradiction.

Answer with EXACTLY one word: CONSISTENT or CONTRADICTS.
""",
    # ---------------------------------------------------------------------
    # Legal Search & Neo4j keyword extraction
    # ---------------------------------------------------------------------
    PromptKey.LEGAL_SEARCH_QUERY_TERMS_SYSTEM: """You are a legal information-retrieval assistant.
Given a claim, extract ONLY legal keywords useful for search (offenses, legal institutes, qualifications, decisive factual elements).
Prioritize legal concepts over narrative details.
Avoid generic words that create lexical noise (e.g., "problem", "situation", "event") unless legally meaningful.

Output format: only a comma-separated list, with no explanations.""",
    PromptKey.LEGAL_SEARCH_QUERY_TERMS_USER: """Extract up to [[max_terms]] keywords.
CLAIM:
[[query_text]]""",
    PromptKey.NEO4J_TOOLS_EXTRACT_KEYWORDS: """Extract the most important legal keywords from this claim.
Focus on: legal concepts, legal domains, types of offenses/violations,
key factual elements, and relevant legal categories.

CLAIM:
"[[claim]]"

RULES:
- Extract 5 to 10 keywords or short phrases (max 3 words each).
- Use Italian legal terminology.
- One keyword per line.
- Do NOT add numbering, bullets, or explanations.
- Do NOT repeat the claim.
- Output ONLY the keywords, nothing else.
""",
    # ---------------------------------------------------------------------
    # Taxonomy Tools
    # ---------------------------------------------------------------------
    PromptKey.TAXONOMY_TOOLS_CLASSIFICATION: """You are an expert legal classifier. Choose the best causal_type_id from the list.

Allowed causal_type_id values:
[[options_text]]

Rules:
- Respond ONLY with the id.
- No explanations, no extra text.

CLAIM:
[[claim]]""",
    PromptKey.TAXONOMY_TOOLS_CAUSALITY_CLAIM_PROMPT: """CLAIM
<<<
[[claim]]
>>>

CONTEXT (if available)
<<<
[[context]]
>>>""",
    PromptKey.TAXONOMY_TOOLS_FILTER_NORM: """Legal Claim:
"[[claim]]"

Norm from taxonomy:
"[[ref]]" - "[[role]]"

Instruction:
Assess whether this norm is relevant to the claim. Answer YES unless the norm is clearly outside the domain of the facts and legal institutes in the claim. If uncertain, YES.

Respond with a single token: YES or NO.""",
    # ---------------------------------------------------------------------
    # Reasoner
    # ---------------------------------------------------------------------
    PromptKey.REASONER_SYSTEM: """IMPORTANT: You MUST respond ENTIRELY in Italian. Every word of your response must be in Italian.

You are the Reasoner. The router already set causal_type_id and theory_id.
Do NOT re-classify. Use these as structural constraints:
- anchor_norms (core + accessory) from config
- principle_tests for the causal type

You receive a pre-retrieved KNOWLEDGE BASE (statutes/precedents) filtered for relevance.
Build the primary legal reasoning on the claim using the provided sources.

Critical rules:
- Cite ONLY statutes and precedents present in the KNOWLEDGE BASE.
- If a needed statute is missing, state "articolo non disponibile nella knowledge base".
- Keep reasoning independent: do not reference the Counter-Reasoner.
- Use ONLY facts explicitly stated in the claim; do NOT invent, assume, or complete missing facts.
- Your response MUST end with a **Catena di ragionamento**: section containing a numbered list.
- Numbered lists (1. 2. 3. ...) are ONLY allowed inside **Catena di ragionamento**. Use prose or bullet points ("-") everywhere else.
- MANDATORY: Your ENTIRE response must be written in Italian. Do NOT write in English.""",
    PromptKey.REASONER_CLASSIFY_CAUSALITY: """You are a classifier. Based PRIMARILY on the CITED ARTICLES from the reasoning chain, choose the most appropriate causal_type_id.

Allowed causal_type_id values (domain=[[domain]]):
[[type_descriptions]]

Classification criteria (based on cited articles):
- If articles are from codice civile (c.c.) like Art. 2043, 2056, 1223, 1226, 1227 → civil causality types
- If articles are from codice penale (c.p.) like Art. 40, 41 → criminal causality types
- If articles are from L. 241/1990 / procedimento amministrativo (e.g. Art. 1, 2, 3, 10-bis, 21-octies) → administrative causality/procedural types
- Consider the combination of articles to determine the most specific causal type

If uncertain, choose the closest from the allowed list.
Respond with ONLY the causal_type_id (no JSON, no explanation, just the id).

ORIGINAL CLAIM (for context only):
[[claim]]

CITED ARTICLES FROM REASONING (primary classification basis):
[[articles_text]]

REASONING CHAIN (for context):
[[chain_text]]
""",
    PromptKey.REASONER_GENERATE_PLAN: """You are a legal planning engine for Italian law.

Create a step-by-step plan to analyze the claim and build a primary legal thesis.
The plan must be executable in sequence and each step must be materially different.
Return ONLY valid JSON (no markdown, no prose) with this schema:
{
  "steps": [
    {
      "id": "P1",
      "goal": "specific legal objective for this step",
      "focus": "single legal/factual focus",
      "expected_norm": "article expected to be cited or 'N/A'",
      "citation_requirement": "required | optional | none"
    }
  ]
}

CLAIM:
"[[claim]]"

DOMAIN: [[routing_domain]]
ANCHOR NORMS:
[[anchor_text]]

PRINCIPLE TESTS:
[[principle_text]]

ALLOWED STATUTES:
[[statutes_list]]

ALLOWED PRECEDENTS:
[[precedents_list]]

KNOWLEDGE BASE:
[[knowledge_base]]

RULES:
- Number of steps must be between [[min_steps]] and [[max_steps]].
- Each step must address a DIFFERENT objective (no overlap/rephrasing).
- Steps must be ordered logically (premise -> legal qualification -> applicability -> consequence -> resulting legal assessment).
- Use only facts explicitly present in claim (no assumptions, no hypothetical factual completions).
- Treat explicit claim facts (including explicit factual findings/perizie stated in the claim) as fixed and true; attack legal implications, not the given facts.
- Prefer using different statutes across steps when possible.
- Set "citation_requirement" to:
  * "required" for norm-application / legal-qualification steps
  * "optional" for inferential bridge or factual-application steps
  * "none" only for pure synthesis/transition steps
- If "expected_norm" is not "N/A", "citation_requirement" should normally be "required".
- Keep each 'goal' and 'focus' concise (max 25 words each).
""",
    PromptKey.REASONER_SUPPORT_STEP: """You are an expert Italian jurist.
You must execute ONLY one planned reasoning step.

CLAIM:
"[[claim]]"

DOMAIN: [[routing_domain]]
ANCHOR NORMS:
[[anchor_text]]
PRINCIPLE TESTS:
[[principle_text]]

KNOWLEDGE BASE (use only these sources):
[[knowledge_base]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]

GLOBAL PLAN:
[[plan_lines]]

CURRENT STEP TO EXECUTE: [[plan_index]]
- Goal: [[plan_goal]]
- Focus: [[plan_focus]]
- Expected norm: [[plan_expected_norm]]
- Citation requirement: [[plan_citation_requirement]]

ALREADY GENERATED STEP SUMMARIES:
[[summary_lines]]

NORMS ALREADY USED: [[used_norms_text]]

HARD RULES:
- Generate EXACTLY ONE atomic step in Italian (2-4 sentences).
- It must advance the plan and add NEW information, not paraphrase prior steps.
- It must materially advance the legal analysis of the claim.
- Use only facts explicitly in claim.
- Do not infer unprovided facts (no assumptions, no hypothetical completions of the factual scenario).
- If citation requirement is "required", cite at least one statute.
- If citation requirement is "optional", citation is recommended but not mandatory.
- If citation requirement is "none", do not force a citation.
- If citing a precedent, include its full exact title from allowed list.

RESPONSE FORMAT:
STEP: [italian atomic step]
""",
    PromptKey.REASONER_SUPPORT_PLAN_REWRITE: """[[previous_prompt]]

YOUR PREVIOUS STEP WAS REJECTED.
REASON: [[invalid_reason]]
INVALID STEP:
[[invalid_step]]

Rewrite only this step. Keep the same planned objective, but produce NEW,
non-redundant and logically consistent content.
RESPONSE FORMAT:
STEP: [italian atomic step]""",
    PromptKey.REASONER_SEMANTIC_REDUNDANCY: """Assess whether the NEW step truly adds new legal information.

CLAIM:
[[claim]]

ARGUMENTATIVE ROLE: [[role]]

PREVIOUS STEPS:
[[context_prev]]

NEW STEP:
[[candidate_step]]

Rule:
- Answer REPEAT if the new step is substantially a paraphrase/duplication of previous steps.
- Answer NEW if it introduces a genuinely different legal/factual point.

Answer with EXACTLY one word: NEW or REPEAT.
""",
    PromptKey.REASONER_SUPPORT_CONCLUSION_REWRITE: """You are an expert Italian jurist. Rewrite the conclusion below so it is strictly consistent with the reasoning chain.

CLAIM:
"[[claim]]"

REASONING CHAIN:
[[chain_text]]

CITED NORMS: [[norms_text]]

INVALID CONCLUSION TO REWRITE:
"[[invalid_conclusion]]"

RULES:
- Keep it in Italian.
- 2-4 sentences max.
- Must clearly synthesize the legal assessment emerging from the chain.
- It may conclude in favor of or against liability/right depending on the chain, but it must not contradict the chain.
- Do not add facts or norms outside the chain.

CONCLUSION:""",
    PromptKey.REASONER_GENERATE_CONCLUSION: """You are an expert Italian jurist. Based on the legal reasoning chain below, generate a concise and precise CONCLUSION.

ORIGINAL CLAIM:
"[[claim]]"

REASONING CHAIN:
[[chain_text]]

CITED NORMS: [[norms_text]]

INSTRUCTIONS:
- Write a conclusion of 2-4 sentences in Italian.
- The conclusion must SYNTHESIZE the result of the legal analysis, not repeat the individual steps.
- Clearly state the resulting legal assessment/qualification and WHY, based on the norms analyzed.
- Do NOT introduce norms or facts not mentioned in the reasoning chain.
- Be direct and assertive in the final verdict.
- Your ENTIRE response must be written in Italian.

        CONCLUSION:""",
    PromptKey.REASONER_REASONING_WITH_CONTEXT: """Analyze the following claim and build a primary legal reasoning.

CLAIM:
"[[claim]]"

DOMAIN (from router):
[[routing_domain]]

ANCHOR NORMS (structural constraints):
[[anchor_text]]

PRINCIPLE TESTS (evaluation criteria):
[[principle_text]]

=== KNOWLEDGE BASE (USE ONLY THESE SOURCES) ===
[[knowledge_base]]
=== END KNOWLEDGE BASE ===

ALLOWED STATUTE REFERENCES (do not cite others):
[[statutes_list]]

ALLOWED PRECEDENT REFERENCES (do not cite others):
[[precedents_list]]

INSTRUCTIONS:
1) Build arguments appropriate for the [[routing_domain]] domain.
2) Use anchor norms and principle tests as structural constraints, but DO NOT limit yourself to them.
   Your reasoning MUST cite multiple statutes from the KNOWLEDGE BASE — not only anchor norms.
   Anchor norms provide the framework, but you MUST integrate additional non-anchor statutes
   from the ALLOWED STATUTES list that are relevant to the specific facts of the claim.
   A good legal argument combines the general principle (anchor) with specific rules that apply
   to the concrete case (e.g., warranty, defects, remedies, damages, obligations).
3) If the knowledge base lacks a statute's text, still cite the article but do NOT invent quotes.
4) Build arguments using ONLY knowledge base sources, with EXACTLY these Italian headers:
   **Premessa**: (premise — write in prose, NO numbered lists)
   **Norma**: (statute with precise citation from ALLOWED STATUTES; if absent, write "articolo non disponibile nella knowledge base" — use bullet points with "-" if listing multiple norms, NEVER numbered lists)
   **Precedente**: (only if present in ALLOWED PRECEDENTS; otherwise omit — NO numbered lists)
   **Nesso Causale**: (causal link — write in prose, NO numbered lists)
   **Conclusione**: (conclusion — write in prose, NO numbered lists)
5) After the arguments, you MUST add the following header and numbered chain.
   This section is MANDATORY and must NEVER be omitted:

   **Catena di ragionamento**:
   1. [First reasoning step — cite the specific article(s) it relies on, e.g. Art. XX c.p.]
   2. [Second reasoning step — cite the specific article(s)]
   3. [Continue for each logical step...]

   RULES for the numbered chain:
   - Use EXACTLY the header "**Catena di ragionamento**:" before the numbered list.
   - Each step MUST be on its own line, starting with "N. " (e.g. "1. ", "2. ", "3. ").
   - Each step MUST reference at least one specific article (e.g. "Art. 2043 c.c.").
   - The chain must have AT LEAST 3 numbered steps.

FORMATTING RULE — CRITICAL:
- Numbered lists ("1. ", "2. ", "3. ", etc.) are ONLY allowed inside the **Catena di ragionamento** section.
- In ALL other sections (Premessa, Norma, Precedente, Nesso Causale, Conclusione), use ONLY
  prose text or bullet points with "-". NEVER use numbered lists outside the chain.

IMPORTANT - NORM USAGE REQUIREMENTS:
- You have [[allowed_statutes_count]] statutes available. Cite EVERY article you deem pertinent
  to the case — do not artificially limit yourself to a fixed number.
- Do NOT rely on a single anchor norm for the entire chain.
- For each factual aspect of the claim (contract formation, defects, remedies, damages, etc.),
  identify the most specific applicable statute from the ALLOWED STATUTES list.
- Quote the relevant text from each statute when available in the KNOWLEDGE BASE.
- COHERENCE RULE: Every norm you cite in the **Norma** section MUST appear in at least one
  step of the numbered reasoning chain, with an explanation of its specific role in the argument.
  Do NOT list norms in **Norma** that you never use in the chain.

CRITICAL: Do not introduce external sources.
MANDATORY LANGUAGE RULE: Your ENTIRE response MUST be written in Italian. Do NOT write in English. Every sentence, header, and explanation must be in Italian.""",
    # ---------------------------------------------------------------------
    # Counter-Reasoner
    # ---------------------------------------------------------------------
    PromptKey.COUNTER_REASONER_SYSTEM: """IMPORTANT: You MUST respond ENTIRELY in Italian. Every word of your response must be in Italian.

You are the Counter-Reasoner. Build a counter-reasoning that opposes the primary legal thesis on the claim and the Reasoner's conclusion.
You receive:
- causal_type_id and theory_id fixed by the Router (do not re-classify)
- selected_attack_ids chosen from the config attack pool
- KNOWLEDGE BASE with relevant statutes and precedents

Critical rules:
- Use ONLY the sources in the KNOWLEDGE BASE; do not invent statutes or precedents.
- Oppose the provided Reasoner conclusion substantively (without quoting it verbatim).
- If a helpful statute is missing from the knowledge base, omit the citation instead of inventing it.
- Always cite the statute number/code when available (e.g., "Art. 41 c.p.").
- Use ONLY facts explicitly stated in the claim; do NOT invent, assume, or complete missing facts.

Expected structure (use these EXACT Italian headers):
- **Premessa Alternativa**
- **Norma** (only if present in ALLOWED STATUTES)
- **Nesso Causale Alternativo**
- **Conclusione Contraria**
- **Catena di ragionamento**: followed by a numbered list (1. 2. 3. ...). This section is MANDATORY.
MANDATORY: Your ENTIRE response must be written in Italian. Do NOT write in English.""",
    PromptKey.COUNTER_REASONER_PICK_ATTACKS: """Claim:
"[[claim]]"

Routing context:
- causal_type_id: [[causal_type_id]]
- theory_id: [[theory_id]]

Select the most useful attack IDs among the following ids.
Return ONLY JSON in this format:
{"attack_ids": ["id1", "id2", "id3"]}

Rules:
- choose between [[min_attacks]] and [[max_attacks]] ids
- ids must be from the list below
- avoid near-duplicate attacks

[[options_text]]
""",
    PromptKey.COUNTER_REASONER_GENERATE_PLAN: """You are a legal planning engine for Italian counter-argumentation.

Create a step-by-step plan to build a counter-argument against the primary legal thesis and the Reasoner's conclusion.
Return ONLY valid JSON (no markdown, no prose) with this schema:
{
  "steps": [
    {
      "id": "C1",
      "goal": "specific objective to weaken the primary thesis",
      "focus": "single weak point for this step",
      "expected_norm": "article expected to be cited or 'N/A'",
      "citation_requirement": "required | optional | none",
      "attack_id": "one of the selected attack ids"
    }
  ]
}

CLAIM:
"[[claim]]"
[[reasoner_block]]
DOMAIN: [[routing_domain]]
CAUSAL TYPE: [[causal_type_id]]
THEORY: [[theory_id]]
SELECTED ATTACK IDS:
[[selected_attack_ids]]

ATTACK CATALOG:
[[attack_catalog]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]
KNOWLEDGE BASE:
[[knowledge_base]]

RULES:
- Number of steps must be between [[min_steps]] and [[max_steps]].
- Each step must be materially different (no overlap/rephrasing).
- Steps must function as counter-argument steps (they must weaken the primary thesis / Reasoner conclusion).
- Use only facts explicitly present in claim (no assumptions, no hypothetical factual completions).
- Each step must include one attack_id from the selected attack ids.
- Distribute selected attacks across the plan whenever possible.
- Set "citation_requirement" using the same policy:
  * "required" for norm-based attacks / legal qualification attacks
  * "optional" for inferential bridge or factual elaboration steps
  * "none" only for pure synthesis/transition steps
- If "expected_norm" is not "N/A", "citation_requirement" should normally be "required".
- Keep each 'goal' and 'focus' concise (max 25 words each).
""",
    PromptKey.COUNTER_REASONER_STEP_PROMPT: """You are an expert Italian jurist.
You must execute ONLY one planned COUNTER step.

CLAIM:
"[[claim]]"
[[reasoner_block]]
DOMAIN: [[routing_domain]]
CAUSAL TYPE: [[causal_type_id]]
THEORY: [[theory_id]]
ATTACK STRATEGY: [[attack_id]] - [[attack_desc]]

KNOWLEDGE BASE (use only these sources):
[[knowledge_base]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]

GLOBAL PLAN:
[[plan_lines]]

CURRENT STEP TO EXECUTE: [[plan_index]]
- Goal: [[plan_goal]]
- Focus: [[plan_focus]]
- Expected norm: [[plan_expected_norm]]
- Citation requirement: [[plan_citation_requirement]]
- Attack id for this step: [[plan_attack_id]]

ALREADY GENERATED STEP SUMMARIES:
[[summary_lines]]

NORMS ALREADY USED: [[used_norms_text]]

HARD RULES:
- Generate EXACTLY ONE atomic step in Italian (2-4 sentences).
- It must advance the plan and add NEW information, not paraphrase prior steps.
- It must function as a counter-step (weakening or challenging the primary thesis / Reasoner conclusion).
- Never invent facts outside claim.
- Never assume or complete missing factual details beyond what is explicitly stated in the claim.
- Treat explicit claim facts (including explicit factual findings/perizie stated in the claim) as fixed and true; do not deny or reverse them.
- Never contradict previous accepted steps.
- If citation requirement is "required", cite at least one statute.
- If citation requirement is "optional", citation is recommended but not mandatory.
- If citation requirement is "none", do not force a citation.
- Align this step explicitly to attack "[[plan_attack_id]]".

RESPONSE FORMAT:
STEP: [italian atomic counter-step]
""",
    PromptKey.COUNTER_REASONER_SEMANTIC_REDUNDANCY: """Assess whether the NEW step truly adds new legal information.

CLAIM:
[[claim]]

ARGUMENTATIVE ROLE: [[role]]

PREVIOUS STEPS:
[[context_prev]]

NEW STEP:
[[candidate_step]]

Rule:
- Answer REPEAT if the new step is substantially a paraphrase/duplication of previous steps.
- Answer NEW if it introduces a genuinely different legal/factual point.

Answer with EXACTLY one word: NEW or REPEAT.
""",
    PromptKey.COUNTER_REASONER_ATTACK_ALIGNMENT: """Assess whether the following counter-argument step is ALIGNED with the planned attack.

CLAIM:
[[claim]]

ATTACK ID:
[[attack_id]]

ATTACK DESCRIPTION:
[[attack_desc]]

PLAN FOCUS:
[[plan_focus]]

STEP CANDIDATE:
[[candidate_step]]

Rules:
- Answer ALIGNED only if the step clearly applies the specified attack and the plan focus.
- Answer MISALIGNED if the step is generic, off-focus, or does not really implement the attack.
- The step must remain a genuine counter-step for the targeted thesis; if it is generic, merely descriptive, or agrees with the targeted thesis, it is MISALIGNED.

Answer with EXACTLY one word: ALIGNED or MISALIGNED.
""",
    PromptKey.COUNTER_REASONER_STEP_OPPOSITION_CHECK: """You are checking whether a counter-argument step actually opposes the primary legal thesis.

CLAIM:
\"\"\"[[claim]]\"\"\"

REASONER CONCLUSION TO OPPOSE:
\"\"\"[[reasoner_conclusion]]\"\"\"

CANDIDATE COUNTER STEP:
\"\"\"[[candidate_step]]\"\"\"

Task:
- Answer AGREEING if the candidate step supports, confirms, or materially reinforces the Reasoner conclusion.
- Answer OPPOSING if the candidate step weakens, disputes, or limits the Reasoner conclusion.
- Answer UNCLEAR if the relation is not clear from the text.

Important:
- Use the claim only as factual context.
- Do not require the step to restate the final opposite outcome explicitly; attacking a key premise of the conclusion still counts as OPPOSING.

Answer with EXACTLY one word: OPPOSING or AGREEING or UNCLEAR.
""",
    PromptKey.COUNTER_REASONER_STANCE_REWRITE: """[[original_prompt]]

YOUR PREVIOUS STEP WAS INVALID.
REASON: [[invalid_reason]]
INVALID STEP:
"[[invalid_step]]"

Rewrite the SAME legal point as a coherent counter-step.
Do not add new facts. Do not make it agree with the targeted thesis / Reasoner conclusion.
The rewritten step must clearly weaken or limit the opposing thesis.

RESPONSE FORMAT:
STEP: [Italian text, max 4 sentences]""",
    # ---------------------------------------------------------------------
    # AQA / NLP / Polisher
    # ---------------------------------------------------------------------
    PromptKey.AQA_ENGINE_ATTACK_TYPE_SYSTEM: """You are an expert in Italian law and ASPIC+ argumentation theory.
Given two reasoning steps from opposing legal arguments, classify the TYPE of attack the attacker performs on the target.

Choose EXACTLY ONE of these categories:
- CONTRADICTION: the attacker directly negates the same legal conclusion or factual premise (e.g. the attacker claims the opposite outcome on the same legal question).
- EXCEPTION: the attacker invokes a condition, proviso, or exception that blocks the target norm from applying (e.g. legitimate defence as an exception to criminal liability).
- DEROGATION: the attacker invokes a more specific norm (lex specialis) that overrides or displaces the target norm (lex generalis).
- EXTINCTION: the attacker claims the right/liability has been extinguished (e.g. prescription, forfeiture, settlement, pardon, statute of limitations).
- FACTUAL_IMPEDIMENT: the attacker raises a factual circumstance (without a normative basis) that impedes the target conclusion (e.g. alibi, absence of evidence, factual impossibility).
- GENERAL_OPPOSITION: none of the above; a generic rebuttal or weakening that does not fit any specific category.

Respond with EXACTLY ONE WORD: the category name in upper case.
No punctuation, no explanation, no extra text.""",
    PromptKey.AQA_ENGINE_ATTACK_TYPE_USER: """TARGET REASONING STEP:
"[[target_text]]"

ATTACKER REASONING STEP:
"[[attacker_text]]"

Attack type?""",
    PromptKey.NLP_UTILS_NLI_SYSTEM: """You are an expert in Italian law.
You are comparing two reasoning passages from opposing sides of a legal debate.
Even if they cite the same legal norms, focus on whether their CONCLUSIONS and APPLICATIONS of those norms are incompatible.

Choose EXACTLY ONE of these labels:
- CONTRADICTION: the two passages reach opposite conclusions on the same legal question, or one undermines a premise that the other relies on.
- ENTAILMENT: the two passages support each other and reach compatible conclusions.
- NEUTRAL: the passages address different legal aspects or their relationship is unclear.

Base your judgement solely on the semantic content of the two passages.

Respond with EXACTLY ONE WORD in upper case.
No punctuation, no explanation.""",
    PromptKey.NLP_UTILS_NLI_USER: """PASSAGE A (target):
"[[target_text]]"

PASSAGE B (attacker):
"[[attacker_text]]"

Relationship?""",
    PromptKey.POLISHER_COUNTER_GATE: """You are a legal dialectical verifier.
Compare two reasoning chains on the same claim and determine whether the COUNTER chain reaches a genuinely opposite legal outcome from the REASONER chain.

CLAIM:
\"\"\"[[claim]]\"\"\"

REASONER CHAIN:
[[reasoner_chain]]

COUNTER CHAIN:
[[counter_chain]]

Respond with EXACTLY ONE label:
- OPPOSING (the COUNTER chain reaches the opposite outcome)
- AGREEING (the COUNTER chain supports or converges with the REASONER)
- UNCLEAR (the COUNTER chain is not clearly opposite)
""",
    # ---------------------------------------------------------------------
    # Consistency Checker (citation verification & repair)
    # ---------------------------------------------------------------------
    PromptKey.CONSISTENCY_VERIFY_MISMATCH_SYSTEM: """You are an expert in Italian law. Your task is to determine whether two normative texts are LOGICALLY EQUIVALENT or DIFFERENT.

Two texts are EQUIVALENT if:
- They express the same legal concept, even with different wording
- One is a faithful paraphrase of the other
- They don't add or omit substantial normative elements

Two texts are DIFFERENT if:
- They add requirements not present in the original
- They omit essential elements
- They change the legal meaning
- They introduce concepts not contained in the original

Respond ONLY with one of these words: EQUIVALENTI or DIVERSI (in Italian)""",
    PromptKey.CONSISTENCY_VERIFY_MISMATCH_USER: """Article [[article_num]]

CITED TEXT (from reasoning):
"[[cited_text]]"

OFFICIAL TEXT (from database):
"[[db_text]]"

Are the two texts EQUIVALENTI or DIVERSI? (Answer in Italian)""",
    PromptKey.CONSISTENCY_PERTINENCE_SYSTEM: """You are an expert in Italian law. Your task is to assess whether a legal norm cited in a reasoning chain is PERTINENT or NOT_PERTINENT to the argumentative logic.

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

Respond ONLY with one of these words: PERTINENT or NOT_PERTINENT""",
    PromptKey.CONSISTENCY_PERTINENCE_USER: """COMPLETE REASONING CHAIN:
\"\"\"
[[full_text]]
\"\"\"

NORM TO EVALUATE: Art. [[article_num]]
CITED TEXT: "[[cited_text]]"

Is this norm PERTINENT or NOT_PERTINENT to the argumentative logic of the reasoning chain?""",
    PromptKey.CONSISTENCY_REPAIR_DB_SYSTEM: """You are an expert in Italian law. You must rewrite a normative passage using EXCLUSIVELY the official text of the provided article.

MANDATORY RULES:
1. You must include a VERBATIM QUOTE (exact copy) of at least 15 consecutive words from the official text
2. The quote must be enclosed in «» (guillemets/angle quotes)
3. You cannot add concepts not present in the official text
4. The result must be legally correct and coherent

OUTPUT: Write ONLY the rewritten text in Italian, without explanations.""",
    PromptKey.CONSISTENCY_REPAIR_DB_USER: """ARTICLE: Art. [[article_num]]

OFFICIAL TEXT TO USE:
"[[db_text]]"

ORIGINAL CONTEXT (to correct):
"[[original_context]]"

Rewrite the normative passage in Italian using only the official text, including a verbatim quote enclosed in «».""",
    PromptKey.CONSISTENCY_PRECEDENT_MISMATCH_SYSTEM: """You are an expert in Italian law. Determine whether two descriptions of a court precedent are LOGICALLY EQUIVALENT or DIFFERENT.

EQUIVALENT: same legal holding, even with different wording.
DIFFERENT: different holding, added/omitted elements, changed meaning.

Respond ONLY with: EQUIVALENTI or DIVERSI""",
    PromptKey.CONSISTENCY_PRECEDENT_MISMATCH_USER: """CITED TEXT (from reasoning):
"[[cited_text]]"

OFFICIAL SUMMARY (from database):
"[[db_summary]]"

Are the two texts EQUIVALENTI or DIVERSI?""",
    PromptKey.CONSISTENCY_PRECEDENT_REPAIR_SYSTEM: """You are an expert in Italian law. Rewrite a passage that cites a court precedent using EXCLUSIVELY the official summary provided.

RULES:
1. Include a VERBATIM QUOTE of at least 15 words from the official summary enclosed in «»
2. Do not add concepts not in the official summary
3. Write in Italian
4. Output ONLY the rewritten text""",
    PromptKey.CONSISTENCY_PRECEDENT_REPAIR_USER: """PRECEDENT: [[precedent_title]]

OFFICIAL SUMMARY:
"[[db_summary]]"

ORIGINAL CONTEXT (to correct):
"[[cited_text]]"

Rewrite using only the official summary.""",
    PromptKey.CONSISTENCY_REGENERATE_STEP_SYSTEM: """You are an expert Italian jurist. You must rewrite a SINGLE reasoning step from a legal argument.

You will receive:
1. The ORIGINAL step text (which contains an incorrect normative citation)
2. The CORRECT official text of the article from the database

RULES:
1. Rewrite the step so that it correctly uses the OFFICIAL text of the article
2. Adjust the legal reasoning to be coherent with the CORRECT article text
3. Include a verbatim quote from the official text in «»
4. Keep approximately the same length and depth
5. Write in Italian
6. Output ONLY the rewritten step text, nothing else
7. Do NOT include the step number (e.g. do NOT start with "3.")""",
    PromptKey.CONSISTENCY_REGENERATE_STEP_USER: """ARTICLE: [[citation]]

CORRECT OFFICIAL TEXT:
"[[db_text_preview]]"

ORIGINAL STEP (to rewrite):
"[[original_step_text]]"

Rewrite this step in Italian using the correct article text.""",
}


_PLACEHOLDER_RE = re.compile(r"\[\[([a-zA-Z_][a-zA-Z0-9_]*)\]\]")
_PROMPT_KEY_BY_VALUE = {key.value: key for key in PromptKey}
_PROMPT_KEY_BY_NAME = {key.name: key for key in PromptKey}


def _coerce_prompt_key(name: str | PromptKey) -> PromptKey:
    """Normalize prompt key input to PromptKey."""
    if isinstance(name, PromptKey):
        return name

    normalized = str(name or "").strip()
    by_value = _PROMPT_KEY_BY_VALUE.get(normalized)
    if by_value is not None:
        return by_value

    by_name = _PROMPT_KEY_BY_NAME.get(normalized)
    if by_name is not None:
        return by_name

    raise KeyError(f"Prompt not found: {name}")


def get_prompt(name: str | PromptKey) -> str:
    """Return prompt template by registry key."""
    key = _coerce_prompt_key(name)
    if key not in PROMPTS:
        raise KeyError(f"Prompt not found: {name}")
    return PROMPTS[key]


def render_prompt(name: str | PromptKey, **values: object) -> str:
    """
    Render prompt replacing placeholders ``[[name]]`` with provided values.

    Missing placeholders are left unchanged.
    """
    template = get_prompt(name)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = values.get(key)
        return str(value) if value is not None else match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)
