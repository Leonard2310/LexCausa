"""
Data loader for LexCausa - Downloads and processes legal datasets.

Datasets:
1. Codice Penale - Local CSV file (src/data/codice_penale.csv)
2. Codice Civile - HuggingFace: AndreaSimeri/Italian_Civil_Code
3. Precedenti (itacasehold) - HuggingFace: itacasehold/itacasehold
4. Gazzetta Ufficiale - HuggingFace: mii-llm/gazzetta-ufficiale
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Base paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_codice_penale() -> pd.DataFrame:
    """Load Codice Penale from local CSV file."""
    csv_path = DATA_DIR / "codice_penale.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Codice Penale CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["source"] = "codice_penale"
    print(f"✅ Loaded Codice Penale: {len(df)} articles")
    return df


def load_codice_civile() -> pd.DataFrame:
    """Load Codice Civile from HuggingFace."""
    print("📥 Downloading Codice Civile from HuggingFace...")
    url = (
        "hf://datasets/AndreaSimeri/Italian_Civil_Code/"
        "italian_civil_code_dataset_with_references.csv"
    )
    df = pd.read_csv(url)
    df["source"] = "codice_civile"

    # Save locally for caching
    cache_path = DATA_DIR / "codice_civile.csv"
    df.to_csv(cache_path, index=False)
    print(f"✅ Loaded Codice Civile: {len(df)} articles (cached at {cache_path})")
    return df


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
    cache_path = DATA_DIR / f"itacasehold_{split}.parquet"
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


# Tipi di Gazzetta Ufficiale da includere
GAZZETTA_TYPES_ALLOWED = [
    "Serie Generale",  # Parte Prima - atti normativi e amministrativi
]

# Pattern per identificare Corte Costituzionale nell'intestazione
CORTE_COSTITUZIONALE_PATTERN = "1ª Serie Speciale"


def load_gazzetta_ufficiale(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load Gazzetta Ufficiale from HuggingFace using Dask (large dataset).

    Filters to include only:
    - Serie Generale (Parte Prima): atti normativi e amministrativi
    - Corte Costituzionale (1ª Serie Speciale): sentenze e ordinanze

    Args:
        sample_size: If provided, only load this many records (for testing)
    """
    print(
        "📥 Downloading Gazzetta Ufficiale from HuggingFace "
        "(large dataset, may take time)..."
    )

    try:
        import dask.dataframe as dd

        ddf = dd.read_parquet(
            "hf://datasets/mii-llm/gazzetta-ufficiale/data/train-*.parquet"
        )

        # Convert to pandas first (needed for proper filtering)
        if sample_size:
            # Take a larger sample to ensure we have enough after filtering
            df = ddf.head(sample_size * 5, npartitions=-1)
        else:
            df = ddf.compute()

        total_before = len(df)
        print(f"📊 Total records before filtering: {total_before}")

        # Filter: Serie Generale OR Corte Costituzionale
        mask_serie_generale = df["type"].isin(GAZZETTA_TYPES_ALLOWED)
        mask_corte_cost = df["intestazione"].str.contains(
            CORTE_COSTITUZIONALE_PATTERN, case=False, na=False
        )

        df = df[mask_serie_generale | mask_corte_cost].copy()

        if sample_size and len(df) > sample_size:
            df = df.head(sample_size)

        print(f"✅ Filtered Gazzetta Ufficiale: {len(df)} records")
        print(f"   - Serie Generale: {mask_serie_generale.sum()}")
        print(f"   - Corte Costituzionale: {mask_corte_cost.sum()}")

        df["source"] = "gazzetta_ufficiale"

        # Save locally for caching
        cache_path = DATA_DIR / "gazzetta_ufficiale.parquet"
        df.to_parquet(cache_path, index=False)
        print(f"💾 Cached at {cache_path}")

        return df

    except Exception as e:
        print(f"❌ Error loading Gazzetta Ufficiale: {e}")
        raise


def main():
    """Download and cache all datasets."""
    print("=" * 60)
    print("LexCausa Data Loader")
    print("=" * 60)

    # Load statutes
    print("\n📖 Loading Statutes...")
    statutes = load_statutes()
    print(f"   Columns: {list(statutes.columns)}")

    # Load precedenti
    print("\n⚖️ Loading Precedenti...")
    precedents = load_precedents_all()
    print(f"   Columns: {list(precedents.columns)}")

    # Load Gazzetta (sample for testing)
    print("\n📰 Loading Gazzetta Ufficiale (sample)...")
    gazzetta = load_gazzetta_ufficiale(sample_size=1000)
    print(f"   Columns: {list(gazzetta.columns)}")

    print("\n" + "=" * 60)
    print("✅ All datasets loaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
