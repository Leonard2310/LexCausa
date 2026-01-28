"""
Legal Claim Classifier using Groq Cloud.

Classifies legal claims into the appropriate book (libro) of the
Italian Civil Code (Codice Civile) or Penal Code (Codice Penale).
"""

import sys
from dataclasses import dataclass, field
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

SECTION_NONE = "N/A"
_SECTION_TAXONOMY_CACHE: Optional[dict[tuple[str, str], list[str]]] = None

SECTION_TAXONOMY = {
    "codice_civile": {
        "CC Disposizioni Preliminari": [],
        "CC Libro I": [
            "Titolo I - Delle persone fisiche",
            "Titolo II - Delle persone giuridiche",
            "Titolo III - Del domicilio e della residenza",
            "Titolo IV - Dell'assenza e della dichiarazione di morte presunta",
            "Titolo V - Della parentela e dell'affinità",
            "Titolo VI - Del matrimonio",
            "Titolo VII - Dello stato di figlio",
            "Titolo VIII - Dell'adozione di persone maggiori di età",
            "Titolo IX - Della responsabilità genitoriale e dei diritti e doveri del figlio",
            "Titolo X - Della tutela e della emancipazione",
            "Titolo XI - Dell'affiliazione e dell'affidamento",
            "Titolo XII - Delle misure di protezione delle persone prive in tutto od in parte di autonomia",
            "Titolo XIII - Degli alimenti",
            "Titolo XIV - Degli atti dello stato civile",
        ],
        "CC Libro II": [
            "Titolo I - Disposizioni generali sulle successioni",
            "Titolo II - Delle successioni legittime",
            "Titolo III - Delle successioni testamentarie",
            "Titolo IV - Della divisione",
            "Titolo V - Delle donazioni",
        ],
        "CC Libro III": [
            "Titolo I - Dei beni",
            "Titolo II - Della proprietà",
            "Titolo III - Della superficie",
            "Titolo IV - Dell'enfiteusi",
            "Titolo V - Dell'usufrutto, dell'uso e dell'abitazione",
            "Titolo VI - Delle servitù prediali",
            "Titolo VII - Della comunione",
            "Titolo VIII - Del possesso",
            "Titolo IX - Della denunzia di nuova opera e di danno temuto",
        ],
        "CC Libro IV": [
            "Titolo I - Delle obbligazioni in generale",
            "Titolo II - Dei contratti in generale",
            "Titolo III - Dei singoli contratti",
            "Titolo IV - Delle promesse unilaterali",
            "Titolo V - Dei titoli di credito",
            "Titolo VI - Della gestione di affari",
            "Titolo VII - Del pagamento dell'indebito",
            "Titolo VIII - Dell'arricchimento senza causa",
            "Titolo IX - Dei fatti illeciti",
        ],
        "CC Libro V": [
            "Titolo I - Della disciplina delle attività professionali",
            "Titolo II - Del lavoro nell'impresa",
            "Titolo III - Del lavoro autonomo",
            "Titolo IV - Del lavoro subordinato in particolari rapporti",
            "Titolo V - Delle società",
            "Titolo VI - Delle società cooperative e delle mutue assicuratrici",
            "Titolo VII - Dell'associazione in partecipazione",
            "Titolo VIII - Dell'azienda",
            "Titolo IX - Dei diritti sulle opere dell'ingegno e sulle invenzioni industriali",
            "Titolo X - Della disciplina della concorrenza e dei consorzi",
            "Titolo XI - Disposizioni penali in materia di società e di consorzi",
        ],
        "CC Libro VI": [
            "Titolo I - Della trascrizione",
            "Titolo II - Delle prove",
            "Titolo III - Della responsabilità patrimoniale, delle cause di prelazione e della conservazione della garanzia patrimoniale",
            "Titolo IV - Della tutela giurisdizionale dei diritti",
            "Titolo V - Della prescrizione e della decadenza",
        ],
    },
    "codice_penale": {
        "CP Libro I": [
            "Titolo I - Della legge penale",
            "Titolo II - Delle pene",
            "Titolo III - Del reato",
            "Titolo IV - Del reo e della persona offesa dal reato",
            "Titolo V - Della non punibilità per particolare tenuità del fatto, della modificazione, applicazione ed esecuzione della pena",
            "Titolo VI - Della estinzione del reato e della pena",
            "Titolo VII - Delle sanzioni civili",
            "Titolo VIII - Delle misure amministrative di sicurezza",
        ],
        "CP Libro II": [
            "Titolo I - Dei delitti contro la personalità dello Stato",
            "Titolo II - Dei delitti contro la pubblica amministrazione",
            "Titolo III - Dei delitti contro l'amministrazione della giustizia",
            "Titolo IV - Dei delitti contro il sentimento religioso e contro la pietà dei defunti",
            "Titolo V - Dei delitti contro l'ordine pubblico",
            "Titolo VI - Dei delitti contro l'incolumità pubblica",
            "Titolo VI bis - Dei delitti contro l'ambiente",
            "Titolo VII - Dei delitti contro la fede pubblica",
            "Titolo VIII - Dei delitti contro l'economia pubblica, l'industria e il commercio",
            "Titolo IX - Dei delitti contro la moralità pubblica e il buon costume",
            "Titolo IX bis - Dei delitti contro il sentimento per gli animali",
            "Titolo XI - Dei delitti contro la famiglia",
            "Titolo XII - Dei delitti contro la persona",
            "Titolo XIII - Dei delitti contro il patrimonio",
        ],
        "CP Libro III": [
            "Titolo I - Delle contravvenzioni di polizia",
            "Titolo II - Delle contravvenzioni concernenti l'attività sociale della pubblica amministrazione",
        ],
    },
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

# Section classifier prompt
SECTION_SYSTEM_PROMPT = """You are a legal-domain routing classifier for Italian law sections.

Your task is to assign a legal claim to the most relevant section
chosen ONLY from the provided list for the selected book.

Rules:
- Output THREE section names by default.
- Output AT MOST TEN section names.
- If only one section applies, output ONLY one.
- Output "N/A" if none applies.
- Do NOT explain the decision.
- Do NOT add any text, symbols, or formatting.
- Do NOT invent new sections.
- Use EXACT section names from the list.
"""

SECTION_PROMPT = """LIBRO
{libro}

SEZIONI DISPONIBILI
{sezioni}

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
    sections: list[str] = field(default_factory=list)
    section_mappings: list[tuple[str, str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Claim: {self.claim}", "", "Classifications:"]
        for i, (cat, desc, mapping) in enumerate(
            zip(self.categories, self.descriptions, self.libro_mappings), 1
        ):
            codice, libro = mapping
            lines.append(f"  {i}. {cat} -> {desc}")
            lines.append(f"     Neo4j: {codice} / {libro}")
            if i - 1 < len(self.sections):
                lines.append(f"     Sezione: {self.sections[i - 1]}")
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
        self._section_taxonomy = self._load_section_taxonomy()

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
            result = self._classify_stream(claim, messages, temperature)
        else:
            result = self._classify_sync(claim, messages, temperature)

        return self._augment_with_sections(claim, result)

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
            sections=[],
            section_mappings=[],
        )

    def _augment_with_sections(
        self, claim: str, result: ClassificationResult
    ) -> ClassificationResult:
        sections = []
        section_mappings = []

        for codice, libro in result.libro_mappings:
            section = self._classify_section(claim, codice, libro)
            sections.append(section)
            section_mappings.append((codice, libro, section))

        result.sections = sections
        result.section_mappings = section_mappings
        return result

    def _classify_section(self, claim: str, codice: str, libro: str) -> str:
        sections = self._section_taxonomy.get((codice, libro), [])
        if not sections:
            return SECTION_NONE

        formatted_sections = "\n".join(f"- {s}" for s in sections)
        messages = [
            {"role": "system", "content": SECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SECTION_PROMPT.format(
                    libro=f"{codice} / {libro}",
                    sezioni=formatted_sections,
                    claim=claim,
                ),
            },
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=32,
            top_p=1,
            stream=False,
        )

        response_text = completion.choices[0].message.content.strip()
        return self._parse_section_response(response_text, sections)

    def _parse_section_response(self, response_text: str, sections: list[str]) -> str:
        if not response_text:
            return SECTION_NONE

        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        if not lines:
            return SECTION_NONE

        if len(lines) == 1 and lines[0].upper() == SECTION_NONE:
            return SECTION_NONE

        candidates = []
        for line in lines:
            if line.upper() == SECTION_NONE:
                continue
            candidates.append(line)
            if len(candidates) >= 2:
                break

        for candidate in candidates:
            for section in sections:
                if candidate == section:
                    return section

        for candidate in candidates:
            lowered = candidate.lower()
            for section in sections:
                if section.lower() == lowered:
                    return section

        for candidate in candidates:
            lowered = candidate.lower()
            for section in sections:
                if section.lower() in lowered or lowered in section.lower():
                    return section

        return SECTION_NONE

    def _load_section_taxonomy(self) -> dict[tuple[str, str], list[str]]:
        global _SECTION_TAXONOMY_CACHE
        if _SECTION_TAXONOMY_CACHE is not None:
            return _SECTION_TAXONOMY_CACHE

        taxonomy: dict[tuple[str, str], list[str]] = {}
        for source, libri in SECTION_TAXONOMY.items():
            for libro, sezioni in libri.items():
                taxonomy[(source, libro)] = list(sezioni)

        _SECTION_TAXONOMY_CACHE = taxonomy
        return _SECTION_TAXONOMY_CACHE

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
