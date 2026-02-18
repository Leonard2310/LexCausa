"""
Data loader for LexCausa - Downloads and processes legal datasets.

Datasets:
1. Codice Penale - Local CSV file (src/data/statuti/codice_penale.csv)
2. Codice Civile - Local CSV file (src/data/statuti/codice_civile.csv)
3. Codice Amministrativo (L. 241/1990) - Local CSV file (src/data/statuti/codice_amministrativo.csv)
4. Precedenti (itacasehold) - Metadata (title/summary/url/materia)

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


def load_codice_amministrativo() -> pd.DataFrame:
    """
    Load Codice Amministrativo (L. 241/1990) from local CSV file.

    Returns DataFrame with columns normalized to the statute schema:
    - numero, titolo, contenuto
    - libro: empty string (this code has no libri)
    - source: 'codice_amministrativo'
    """
    csv_path = settings.statutes_dir / "codice_amministrativo.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Codice Amministrativo CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["source"] = "codice_amministrativo"
    df["libro"] = ""

    print(f"✅ Loaded Codice Amministrativo (L. 241/1990): {len(df)} articles")
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


def _generate_embeddings_for_texts(
    texts: list[str], batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings using the same local model used in runtime search.

    This is used as fallback when pre-computed .npy files are missing.
    """
    import os

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Generating embeddings on {device} for {len(texts)} texts...")

    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model, local_files_only=True
    )
    os.environ["TRUST_REMOTE_CODE"] = "1"
    model = AutoModel.from_pretrained(
        settings.embedding_model, local_files_only=True, trust_remote_code=True
    ).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=settings.embedding_max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        token_embeddings = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
        mask = mask.float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"✅ Generated embeddings shape: {embeddings.shape}")
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


def load_codice_amministrativo_with_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Codice Amministrativo with embeddings.

    If pre-computed embeddings are missing (or size-mismatched), they are generated
    on the fly and saved to `src/data/embeddings/amministrativo_embeddings.npy`.
    """
    df = load_codice_amministrativo()
    embeddings = load_embeddings("amministrativo")

    if embeddings is None or len(embeddings) != len(df):
        if embeddings is not None and len(embeddings) != len(df):
            print(
                "⚠️ amministrativo embeddings size mismatch "
                f"({len(embeddings)} vs {len(df)}), regenerating..."
            )
        texts = [
            f"Art. {str(row.get('numero', '')).strip()} - "
            f"{str(row.get('titolo', '')).strip()}: "
            f"{str(row.get('contenuto', '')).strip()}"
            for _, row in df.iterrows()
        ]
        embeddings = _generate_embeddings_for_texts(texts)
        out_path = settings.embeddings_dir / "amministrativo_embeddings.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        print(f"💾 Saved amministrativo embeddings to: {out_path}")

    return df, embeddings


def load_statutes() -> pd.DataFrame:
    """Load all statutes (Codice Penale + Codice Civile + Codice Amministrativo)."""
    codice_penale = load_codice_penale()
    codice_civile = load_codice_civile()
    codice_amministrativo = load_codice_amministrativo()

    # Normalize column names
    statutes = pd.concat(
        [codice_penale, codice_civile, codice_amministrativo], ignore_index=True
    )
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
    _ = load_embeddings("amministrativo")

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
