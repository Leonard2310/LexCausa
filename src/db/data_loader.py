"""
Data loader for LexCausa - Downloads and processes legal datasets.

Datasets:
1. Codice Penale - Local CSV file (src/data/statuti/codice_penale.csv)
2. Codice Civile - Local CSV file (src/data/statuti/codice_civile.csv)
3. Precedenti (itacasehold) - Metadata (title/summary/url/materia)

Note: I CSV degli statuti includono le colonne libro_codice_penale e libro_codice_civile
per il raggruppamento degli articoli per libro di appartenenza.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import settings


def load_codice_penale() -> pd.DataFrame:
    """
    Load Codice Penale from local CSV file.

    Returns DataFrame with columns:
    - articolo, titolo, testo, reference, external_reference
    - libro: libro di appartenenza con prefisso (CP Libro I, II, III)
    - source: 'codice_penale'
    """
    csv_path = settings.statutes_dir / "codice_penale.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Codice Penale CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["source"] = "codice_penale"

    # Normalizza nome colonna libro e aggiungi prefisso CP
    if "libro_codice_penale" in df.columns:
        df["libro"] = "CP " + df["libro_codice_penale"].astype(str)

    print(f"✅ Loaded Codice Penale: {len(df)} articles")
    print(
        f"   Libri: {df['libro'].unique().tolist() if 'libro' in df.columns else 'N/A'}"
    )
    return df


def load_codice_civile() -> pd.DataFrame:
    """
    Load Codice Civile from local CSV file.

    Returns DataFrame with columns:
    - article_id, article_title, article_text, article_references
    - libro: libro di appartenenza con prefisso (CC Libro I, II, etc.)
    - source: 'codice_civile'
    """
    csv_path = settings.statutes_dir / "codice_civile.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Codice Civile CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["source"] = "codice_civile"

    # Normalizza nome colonna libro e aggiungi prefisso CC
    if "libro_codice_civile" in df.columns:
        df["libro"] = "CC " + df["libro_codice_civile"].astype(str)

    print(f"✅ Loaded Codice Civile: {len(df)} articles")
    print(
        f"   Libri: {df['libro'].unique().tolist() if 'libro' in df.columns else 'N/A'}"
    )
    return df


def load_embeddings(source: str) -> Optional[np.ndarray]:
    """
    Load pre-computed embeddings from .npy file.

    Args:
        source: 'civile' or 'penale'

    Returns:
        numpy array of shape (n_articles, embedding_dim) or None if not found
    """
    embeddings_path = settings.embeddings_dir / f"{source}_embeddings.npy"
    if not embeddings_path.exists():
        print(f"⚠️ Embeddings not found at {embeddings_path}")
        return None

    embeddings = np.load(embeddings_path)
    print(f"✅ Loaded {source} embeddings: {embeddings.shape}")
    return embeddings


def load_itacasehold_metadata() -> list[dict]:
    """
    Load itacasehold precedents metadata without embeddings.

    Returns:
        List of dicts with title, summary, url, materia.
    """
    parquet_path = settings.precedents_dir / "itacasehold_train.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Itacasehold parquet not found at {parquet_path}")

    columns = ["title", "summary", "url", "materia"]
    df = pd.read_parquet(parquet_path, columns=columns)
    # Keep payload Neo4j-safe and consistent for fulltext retrieval
    records = (
        df.fillna("")
        .astype(
            {
                "title": "string",
                "summary": "string",
                "url": "string",
                "materia": "string",
            }
        )
        .to_dict(orient="records")
    )
    print(f"✅ Loaded itacasehold metadata: {len(records)} precedents")
    return records


def load_codice_penale_with_embeddings() -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Load Codice Penale with corresponding embeddings."""
    df = load_codice_penale()
    embeddings = load_embeddings("penale")
    return df, embeddings


def load_codice_civile_with_embeddings() -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Load Codice Civile with corresponding embeddings."""
    df = load_codice_civile()
    embeddings = load_embeddings("civile")
    return df, embeddings


def load_statutes() -> pd.DataFrame:
    """Load all statutes (Codice Penale + Codice Civile)."""
    codice_penale = load_codice_penale()
    codice_civile = load_codice_civile()

    # Normalize column names
    statutes = pd.concat([codice_penale, codice_civile], ignore_index=True)
    print(f"📚 Total statutes: {len(statutes)}")
    return statutes


def main():
    """Test del data loader."""
    print("=" * 60)
    print("LexCausa Data Loader - Test")
    print("=" * 60)

    # Load statutes
    print("\n📖 Loading Statutes...")
    statutes = load_statutes()
    print(f"   Columns: {list(statutes.columns)}")

    # Load embeddings
    print("\n📊 Loading Embeddings...")
    _ = load_embeddings("penale")
    _ = load_embeddings("civile")

    # Load itacasehold con metadata
    print("\n⚖️ Loading Itacasehold con metadata...")
    try:
        meta = load_itacasehold_metadata()
        print(f"   Metadata: {len(meta)} chunks")
    except FileNotFoundError as e:
        print(f"   ⚠️ {e}")

    print("\n" + "=" * 60)
    print("✅ Data loader test completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()
