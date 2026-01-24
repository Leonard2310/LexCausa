"""
LangChain Tools for Causality Taxonomy.

Provides tools for:
- Classifying causality type using LLM (Material, Legal, Concurrent Causes)
- Retrieving causality theory and associated norms
"""

import json
import sys
from pathlib import Path
from typing import Optional

from groq import Groq
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Cross-platform path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402

_taxonomy_cache: Optional[dict] = None
_groq_client: Optional[Groq] = None


def get_taxonomy() -> dict:
    """Load and cache the causality taxonomy."""
    global _taxonomy_cache
    if _taxonomy_cache is None:
        taxonomy_path = settings.data_dir / "tassonomia_causale.json"
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            _taxonomy_cache = json.load(f)
    return _taxonomy_cache


def get_groq_client() -> Groq:
    """Get or create Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# Mapping from causality type ID to taxonomy entry name
CAUSALITY_TYPES = {
    "MATERIALE": "Materiale",
    "GIURIDICA": "Giuridica",
    "CONCAUSE": "Concause / Sopravvenute",
}

# System prompt for causality classification
CAUSALITY_SYSTEM_PROMPT = """You are an expert legal classifier specializing in Italian causality theory.

Your task is to classify a legal claim into the most appropriate type of causality.

Types of Causality:

1. MATERIALE (Material Causality)
   - Concerns the factual/natural connection between conduct and event
   - Domain: Criminal Law (Diritto Penale)
   - Key concept: "conditio sine qua non" - the conduct must be a necessary condition for the event
   - Core norms: Art. 40-41 c.p. (Criminal Code)
   - Examples: Did the defendant's action cause the victim's death? Was the omission a cause of the harm?

2. GIURIDICA (Legal Causality)
   - Concerns the connection between wrongful event and compensable damage
   - Domain: Civil Law (Diritto Civile)
   - Key concept: Immediate and direct consequences, foreseeability
   - Core norms: Art. 1223, 2043, 2056 c.c. (Civil Code)
   - Examples: What damages can be claimed? Is the loss a direct consequence of the breach?

3. CONCAUSE (Concurrent/Supervening Causes)
   - Concerns interaction between multiple causal factors
   - Domain: Both Criminal and Civil Law
   - Key concept: Contribution of multiple causes, interruption of causal chain
   - Core norms: Art. 41 c.p., Art. 1227, 2055 c.c.
   - Examples: Multiple tortfeasors, victim's contributory negligence, supervening events

Rules:
- Output ONLY ONE category ID: MATERIALE, GIURIDICA, or CONCAUSE
- Do NOT explain the decision
- Do NOT add any text, symbols, or formatting
- Base your decision on the legal context and nature of the claim
"""

CAUSALITY_CLAIM_PROMPT = """CLAIM
<<<
{claim}
>>>

CONTEXT (if available)
<<<
{context}
>>>"""

# Few-shot examples
CAUSALITY_FEW_SHOT = [
    {
        "claim": "L'imputato ha somministrato una dose di veleno che ha causato la morte della vittima.",
        "context": "",
        "response": "MATERIALE",
    },
    {
        "claim": "Il debitore non ha adempiuto e chiedo il risarcimento dei danni subiti incluso il lucro cessante.",
        "context": "",
        "response": "GIURIDICA",
    },
    {
        "claim": "L'incidente è stato causato sia dalla negligenza del conducente che dal comportamento imprudente del pedone.",
        "context": "",
        "response": "CONCAUSE",
    },
    {
        "claim": "Il medico ha omesso di diagnosticare la malattia e il paziente è deceduto.",
        "context": "Art. 40 c.p., Art. 589 c.p.",
        "response": "MATERIALE",
    },
]


class ClassifyCausalityInput(BaseModel):
    """Input schema for causality classification."""

    claim: str = Field(
        description="The legal claim to analyze to determine the type of causality"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context (e.g., relevant articles already identified)",
    )


def _build_classification_messages(
    claim: str, context: Optional[str] = None
) -> list[dict]:
    """Build the message chain with few-shot examples for causality classification."""
    messages = [{"role": "system", "content": CAUSALITY_SYSTEM_PROMPT}]

    # Add few-shot examples
    for example in CAUSALITY_FEW_SHOT:
        messages.append(
            {
                "role": "user",
                "content": CAUSALITY_CLAIM_PROMPT.format(
                    claim=example["claim"], context=example["context"] or "N/A"
                ),
            }
        )
        messages.append({"role": "assistant", "content": example["response"]})

    # Add the actual claim
    messages.append(
        {
            "role": "user",
            "content": CAUSALITY_CLAIM_PROMPT.format(
                claim=claim, context=context or "N/A"
            ),
        }
    )

    return messages


@tool("classify_causality", args_schema=ClassifyCausalityInput)
def classify_causality_tool(claim: str, context: Optional[str] = None) -> dict:
    """
    Classify the type of causality in a legal claim using LLM.

    Analyzes the claim and determines whether it involves:
    - Material Causality: factual connection between conduct and event (criminal law)
    - Legal Causality: connection between event and compensable damage (civil law)
    - Concurrent/Supervening Causes: interaction between multiple causal factors

    Returns the causality type, associated warrant, and relevant norms from taxonomy.
    """
    # Use LLM to classify
    client = get_groq_client()
    messages = _build_classification_messages(claim, context)

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,  # Low temperature for consistent classification
        max_tokens=20,
    )

    # Parse response
    content = response.choices[0].message.content
    raw_response = (content or "").strip().upper()

    # Extract causality type ID
    causality_id = None
    for type_id in CAUSALITY_TYPES.keys():
        if type_id in raw_response:
            causality_id = type_id
            break

    # Default to MATERIALE if parsing fails
    if causality_id is None:
        causality_id = "MATERIALE"

    type_name = CAUSALITY_TYPES[causality_id]

    # Get taxonomy entry
    taxonomy = get_taxonomy()
    taxonomy_entry = None
    for entry in taxonomy.get("tassonomia_causalita", []):
        if entry.get("tipo_causalita") == type_name:
            taxonomy_entry = entry
            break

    if not taxonomy_entry:
        return {
            "causality_type": type_name,
            "causality_id": causality_id,
            "error": "Taxonomy entry not found",
        }

    # Extract relevant information
    return {
        "causality_type": type_name,
        "causality_id": causality_id,
        "warrant": taxonomy_entry.get("warrant", {}),
        "principle": taxonomy_entry.get("principio_test_applicato", ""),
        "description": taxonomy_entry.get("descrizione_ruolo", ""),
        "core_norms": taxonomy_entry.get("norme_core", []),
        "accessory_norms": taxonomy_entry.get("norme_accessorie", []),
        "limits": taxonomy_entry.get("limiti_criticita", ""),
    }


class GetCausalityTheoryInput(BaseModel):
    """Input schema for getting causality theory."""

    causality_type: str = Field(
        description="Type of causality: 'Materiale', 'Giuridica', or 'Concause / Sopravvenute'"
    )


@tool("get_causality_theory", args_schema=GetCausalityTheoryInput)
def get_causality_theory_tool(causality_type: str) -> dict:
    """
    Retrieve the complete theory associated with a causality type.

    Given a causality classification, returns:
    - Warrant (name and description)
    - Applied principle/test
    - Core and accessory norms
    - Subtypes (if present)
    - Limits and criticalities

    Use this function to build counter-arguments based on causal theory.
    """
    taxonomy = get_taxonomy()

    # Normalize input
    normalized_type = causality_type.strip()

    # Find matching entry
    for entry in taxonomy.get("tassonomia_causalita", []):
        if entry.get("tipo_causalita", "").lower() == normalized_type.lower():
            result = {
                "tipo_causalita": entry.get("tipo_causalita"),
                "warrant": entry.get("warrant", {}),
                "principio_test": entry.get("principio_test_applicato", ""),
                "descrizione": entry.get("descrizione_ruolo", ""),
                "norme_core": entry.get("norme_core", []),
                "norme_accessorie": entry.get("norme_accessorie", []),
                "limiti_criticita": entry.get("limiti_criticita", ""),
            }

            # Include subtypes if present
            notes = entry.get("note")
            if notes and "sottotipi_inclusi" in notes:
                result["sottotipi"] = notes["sottotipi_inclusi"]

            return result

    return {
        "error": f"Tipo di causalità '{causality_type}' non trovato nella tassonomia",
        "available_types": ["Materiale", "Giuridica", "Concause / Sopravvenute"],
    }


def get_all_causality_types() -> list[dict]:
    """Get summary of all causality types for reference."""
    taxonomy = get_taxonomy()
    types = []

    for entry in taxonomy.get("tassonomia_causalita", []):
        types.append(
            {
                "tipo": entry.get("tipo_causalita"),
                "warrant": entry.get("warrant", {}).get("denominazione", ""),
                "principio": entry.get("principio_test_applicato", ""),
            }
        )

    return types
