"""
Legal Claim Classifier using Groq Cloud.

Classifies legal claims into the appropriate book (libro) of the
Italian Civil Code (Codice Civile) or Penal Code (Codice Penale).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from groq import Groq

# Cross-platform path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402

# Taxonomy mapping to Neo4j libro names
# I nomi dei libri includono il prefisso del codice (CC/CP) per univocità
TAXONOMY_TO_LIBRO = {
    # Codice Civile
    "CC_PRE": ("codice_civile", "CC Disposizioni Preliminari"),
    "CC_L1": ("codice_civile", "CC Libro I"),
    "CC_L2": ("codice_civile", "CC Libro II"),
    "CC_L3": ("codice_civile", "CC Libro III"),
    "CC_L4": ("codice_civile", "CC Libro IV"),
    "CC_L5": ("codice_civile", "CC Libro V"),
    "CC_L6": ("codice_civile", "CC Libro VI"),
    # Codice Penale
    "CP_L1": ("codice_penale", "CP Libro I"),
    "CP_L2": ("codice_penale", "CP Libro II"),
    "CP_L3": ("codice_penale", "CP Libro III"),
}

TAXONOMY_DESCRIPTIONS = {
    "CC_PRE": "Codice Civile, Disposizioni sulla legge in generale",
    "CC_L1": "Codice Civile, Libro I: Persone e famiglia",
    "CC_L2": "Codice Civile, Libro II: Successioni",
    "CC_L3": "Codice Civile, Libro III: Proprietà e diritti reali",
    "CC_L4": "Codice Civile, Libro IV: Obbligazioni e contratti",
    "CC_L5": "Codice Civile, Libro V: Lavoro, impresa e società",
    "CC_L6": "Codice Civile, Libro VI: Tutela dei diritti",
    "CP_L1": "Codice Penale, Libro I: Reati in generale",
    "CP_L2": "Codice Penale, Libro II: Delitti in particolare",
    "CP_L3": "Codice Penale, Libro III: Contravvenzioni",
}

SYSTEM_PROMPT = """You are a legal-domain routing classifier for Italian law.

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
"""

TAXONOMY_PROMPT = """TAXONOMY

CC_PRE  -> Codice Civile, Disposizioni sulla legge in generale
CC_L1   -> Codice Civile, Libro I: Persone e famiglia
CC_L2   -> Codice Civile, Libro II: Successioni
CC_L3   -> Codice Civile, Libro III: Proprietà e diritti reali
CC_L4   -> Codice Civile, Libro IV: Obbligazioni e contratti
CC_L5   -> Codice Civile, Libro V: Lavoro, impresa e società
CC_L6   -> Codice Civile, Libro VI: Tutela dei diritti

CP_L1   -> Codice Penale, Libro I: Reati in generale
CP_L2   -> Codice Penale, Libro II: Delitti in particolare
CP_L3   -> Codice Penale, Libro III: Contravvenzioni

CLAIM
<<<
{claim}
>>>"""

# Few-shot examples for better classification
FEW_SHOT_EXAMPLES = [
    {
        "claim": (
            "Il venditore non ha consegnato l'immobile nei tempi "
            "previsti dal contratto e chiede comunque il saldo."
        ),
        "response": "CC_L4\nCC_L3\nCC_L6",
    },
    {
        "claim": (
            "Il creditore agisce dopo molti anni e temo che "
            "il diritto sia prescritto."
        ),
        "response": "CC_PRE\nCC_L6\nCC_L4",
    },
    {
        "claim": "Ho subito un furto in casa e voglio sporgere denuncia.",
        "response": "CP_L2\nCP_L1\nCC_L3",
    },
]


@dataclass
class ClassificationResult:
    """Result of claim classification."""

    claim: str
    categories: list[str]
    descriptions: list[str]
    libro_mappings: list[tuple[str, str]]  # (codice, libro)

    def __str__(self) -> str:
        lines = [f"Claim: {self.claim}", "", "Classifications:"]
        for i, (cat, desc, mapping) in enumerate(
            zip(self.categories, self.descriptions, self.libro_mappings), 1
        ):
            codice, libro = mapping
            lines.append(f"  {i}. {cat} -> {desc}")
            lines.append(f"     Neo4j: {codice} / {libro}")
        return "\n".join(lines)


class ClaimClassifier:
    """
    Classifies legal claims using Groq Cloud LLM.

    Uses the Llama 4 Scout model to route claims to the appropriate
    book of the Italian Civil or Penal Code.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            api_key: Groq API key. If None, reads from settings.
            model: Model to use for classification. If None, reads from settings.
        """
        self.api_key = api_key or settings.groq_api_key
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in .env or pass api_key parameter."
            )

        self.client = Groq(api_key=self.api_key)
        self.model = model or settings.groq_model

    def _build_messages(self, claim: str) -> list[dict]:
        """Build the message chain with few-shot examples."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add few-shot examples
        for example in FEW_SHOT_EXAMPLES:
            messages.append(
                {
                    "role": "user",
                    "content": TAXONOMY_PROMPT.format(claim=example["claim"]),
                }
            )
            messages.append({"role": "assistant", "content": example["response"]})

        # Add the actual claim
        messages.append(
            {"role": "user", "content": TAXONOMY_PROMPT.format(claim=claim)}
        )

        return messages

    def classify(
        self,
        claim: str,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> ClassificationResult:
        """
        Classify a legal claim into taxonomy categories.

        Args:
            claim: The legal claim text to classify.
            temperature: LLM temperature (lower = more deterministic).
            stream: Whether to stream the response.

        Returns:
            ClassificationResult with top 3 categories.
        """
        messages = self._build_messages(claim)

        if stream:
            return self._classify_stream(claim, messages, temperature)
        else:
            return self._classify_sync(claim, messages, temperature)

    def _classify_sync(
        self,
        claim: str,
        messages: list[dict],
        temperature: float,
    ) -> ClassificationResult:
        """Synchronous classification."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=64,
            top_p=1,
            stream=False,
        )

        response_text = completion.choices[0].message.content.strip()
        return self._parse_response(claim, response_text)

    def _classify_stream(
        self,
        claim: str,
        messages: list[dict],
        temperature: float,
    ) -> ClassificationResult:
        """Streaming classification."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=64,
            top_p=1,
            stream=True,
        )

        response_text = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            response_text += content

        return self._parse_response(claim, response_text.strip())

    def _parse_response(
        self,
        claim: str,
        response_text: str,
    ) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        # Split by newlines and filter valid categories
        lines = [line.strip() for line in response_text.split("\n")]
        categories = [line for line in lines if line in TAXONOMY_TO_LIBRO][
            :3
        ]  # Take top 3

        # If we got fewer than 3, that's still okay
        if not categories:
            # Fallback: try to find any valid category in response
            for cat in TAXONOMY_TO_LIBRO.keys():
                if cat in response_text:
                    categories.append(cat)
                    if len(categories) >= 3:
                        break

        descriptions = [TAXONOMY_DESCRIPTIONS.get(cat, "Unknown") for cat in categories]
        libro_mappings = [
            TAXONOMY_TO_LIBRO.get(cat, ("unknown", "unknown")) for cat in categories
        ]

        return ClassificationResult(
            claim=claim,
            categories=categories,
            descriptions=descriptions,
            libro_mappings=libro_mappings,
        )

    def get_libro_filter(
        self,
        claim: str,
        top_n: int = 1,
    ) -> list[tuple[str, str]]:
        """
        Get libro filters for Neo4j vector search.

        Args:
            claim: The legal claim to classify.
            top_n: Number of top categories to return.

        Returns:
            List of (codice, libro) tuples for filtering.
        """
        result = self.classify(claim)
        return result.libro_mappings[:top_n]


def main():
    """Interactive CLI for testing the classifier."""
    print("=" * 60)
    print("LexCausa - Legal Claim Classifier")
    print("=" * 60)
    print()
    print("This tool classifies legal claims into Italian law categories.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    try:
        classifier = ClaimClassifier()
        print("✅ Groq client initialized successfully.")
        print(f"   Model: {classifier.model}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    print()

    while True:
        print("-" * 60)
        claim = input("📝 Enter your legal claim:\n> ").strip()

        if claim.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        if not claim:
            print("⚠️ Please enter a valid claim.")
            continue

        print("\n🔄 Classifying...")

        try:
            result = classifier.classify(claim)
            print()
            print("📊 Result:")
            print(result)
            print()
        except Exception as e:
            print(f"❌ Error during classification: {e}")


if __name__ == "__main__":
    main()
