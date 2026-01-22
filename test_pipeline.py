#!/usr/bin/env python3
"""Test script for the LegalSearchPipeline."""

import warnings

warnings.filterwarnings("ignore")

from src.services.legal_search import LegalSearchPipeline  # noqa: E402


def main():
    print("Inizializzazione pipeline...")
    pipeline = LegalSearchPipeline()

    print()
    print("=" * 60)
    print("TEST 1: Ricerca per claim su contratto immobiliare")
    print("=" * 60)

    claim1 = (
        "Il venditore non ha consegnato l immobile nei tempi previsti "
        "dal contratto e chiede comunque il saldo."
    )
    result1 = pipeline.search(claim1, top_k=5)

    print()
    print(f"Claim: {result1.claim}")
    print(f"Libri classificati: {result1.classification.categories}")
    print(f"Articoli trovati: {len(result1.articles)}")
    print()
    for i, art in enumerate(result1.articles, 1):
        title = art.titolo[:80] + "..." if len(art.titolo) > 80 else art.titolo
        print(f"{i}. [{art.statute_id}] Score: {art.score:.4f}")
        print(f"   Libro: {art.libro}")
        print(f"   Art. {art.articolo}: {title}")
        print()

    print()
    print("=" * 60)
    print("TEST 2: Ricerca per claim su reato penale")
    print("=" * 60)

    claim2 = (
        "L imputato ha sottratto denaro dalla cassa del supermercato in cui lavorava."
    )
    result2 = pipeline.search(claim2, top_k=5)

    print()
    print(f"Claim: {result2.claim}")
    print(f"Libri classificati: {result2.classification.categories}")
    print(f"Articoli trovati: {len(result2.articles)}")
    print()
    for i, art in enumerate(result2.articles, 1):
        title = art.titolo[:80] + "..." if len(art.titolo) > 80 else art.titolo
        print(f"{i}. [{art.statute_id}] Score: {art.score:.4f}")
        print(f"   Libro: {art.libro}")
        print(f"   Art. {art.articolo}: {title}")
        print()

    print()
    print("=" * 60)
    print("TEST 3: Ricerca per claim su successione")
    print("=" * 60)

    claim3 = (
        "Il testamento olografo del de cuius e stato impugnato "
        "dai legittimari per lesione di quota."
    )
    result3 = pipeline.search(claim3, top_k=5)

    print()
    print(f"Claim: {result3.claim}")
    print(f"Libri classificati: {result3.classification.categories}")
    print(f"Articoli trovati: {len(result3.articles)}")
    print()
    for i, art in enumerate(result3.articles, 1):
        title = art.titolo[:80] + "..." if len(art.titolo) > 80 else art.titolo
        print(f"{i}. [{art.statute_id}] Score: {art.score:.4f}")
        print(f"   Libro: {art.libro}")
        print(f"   Art. {art.articolo}: {title}")
        print()

    pipeline.close()
    print("✅ Pipeline chiusa correttamente")


if __name__ == "__main__":
    main()
