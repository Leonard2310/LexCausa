"""
Data loader for LexCausa - Downloads and processes legal datasets.

Datasets:
1. Codice Penale - Local CSV file (src/data/statuti/codice_penale.csv)
2. Codice Civile - Local CSV file (src/data/statuti/codice_civile.csv)
3. Precedenti (itacasehold) - Chunk embeddings pre-calcolati

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
        source: 'civile', 'penale', or 'itacasehold'

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


def load_itacasehold_with_embeddings() -> Tuple[list, Optional[np.ndarray]]:
    """
    Load itacasehold precedents with summary-based embeddings.

    Metadata is loaded from the parquet file (primary source).
    Embeddings are loaded from the pre-computed .npy file.

    Returns:
        Tuple of (records_list, embeddings_array)
        - records_list: List of dicts with title, summary, url
        - embeddings: numpy array of shape (n_documents, 768)
    """
    parquet_path = settings.precedents_dir / "itacasehold_train.parquet"
    embeddings_path = settings.embeddings_dir / "itacasehold_embeddings.npy"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Itacasehold parquet not found at {parquet_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Itacasehold embeddings not found at {embeddings_path}"
        )

    # Carica metadata dal parquet (colonne: url, title, doc, summary, materia, source)
    df = pd.read_parquet(parquet_path, columns=["title", "summary", "url","materia"])
    records = df.to_dict(orient="records")

    embeddings = np.load(embeddings_path)

    print(
        f"✅ Loaded itacasehold: {len(records)} documents, "
        f"embeddings {embeddings.shape}"
    )

    if len(records) != embeddings.shape[0]:
        print(
            f"⚠️ Mismatch: {len(records)} records vs "
            f"{embeddings.shape[0]} embeddings"
        )

    return records, embeddings


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


def load_precedents(split: str = "train") -> pd.DataFrame:
    """
    Load precedenti from HuggingFace (itacasehold dataset).

    Args:
        split: One of 'train', 'validation', 'test'
    """
    print(f"📥 Downloading precedenti (itacasehold - {split}) from HuggingFace...")

    splits = {
        "train": "data/train-00000-of-00001-f5930e0acbcdcb7f.parquet",
        "validation": ("data/validation-00000-of-00001-5a3a26aa9ff539c5.parquet"),
        "test": "data/test-00000-of-00001-d201871bb2dc6277.parquet",
    }

    if split not in splits:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {list(splits.keys())}"
        )

    df = pd.read_parquet(f"hf://datasets/itacasehold/itacasehold/{splits[split]}")
    df["source"] = f"itacasehold_{split}"

    # Save locally for caching
    cache_path = settings.data_dir / f"itacasehold_{split}.parquet"
    df.to_parquet(cache_path, index=False)
    print(
        f"✅ Loaded precedenti ({split}): {len(df)} records "
        f"(cached at {cache_path})"
    )
    return df


def load_precedents_all() -> pd.DataFrame:
    """Load all splits of precedenti dataset."""
    dfs = []
    for split in ["train", "validation", "test"]:
        df = load_precedents(split)
        dfs.append(df)

    all_precedents = pd.concat(dfs, ignore_index=True)
    print(f"📚 Total precedenti: {len(all_precedents)}")
    return all_precedents


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
    _ = load_embeddings("itacasehold")

    # Load itacasehold con metadata
    print("\n⚖️ Loading Itacasehold con metadata...")
    try:
        meta, emb = load_itacasehold_with_embeddings()
        print(f"   Metadata: {len(meta)} chunks")
        print(f"   Embeddings: {emb.shape}")
    except FileNotFoundError as e:
        print(f"   ⚠️ {e}")

    print("\n" + "=" * 60)
    print("✅ Data loader test completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()
