#!/usr/bin/env python3
"""Verify articles bis/ter/etc are included in the index."""

import numpy as np
import pandas as pd

DATA_DIR = (
    "/Users/l.catello/Library/Mobile Documents/"
    "com~apple~CloudDocs/Magistrale Ingegneria Informatica/Tesi/LexCausa/src/data"
)

# Verifico articoli bis/tris nel codice civile
civile = pd.read_csv(f"{DATA_DIR}/statuti/codice_civile_aggiornato.csv")
print("Codice Civile:")
bis_pattern = r"bis|ter|quater|quinquies|sexies|septies|octies|novies|decies"
bis_articles = civile[
    civile["article_id"].str.contains(bis_pattern, case=False, na=False)
]
print(f"  Totale articoli: {len(civile)}")
print(f"  Articoli bis/ter/etc: {len(bis_articles)}")
print(f"  Esempi: {bis_articles['article_id'].head(10).tolist()}")

# Verifico nel codice penale
penale = pd.read_csv(f"{DATA_DIR}/statuti/codice_penale_aggiornato.csv")
print()
print("Codice Penale:")
bis_penale = penale[
    penale["articolo"].astype(str).str.contains(bis_pattern, case=False, na=False)
]
print(f"  Totale articoli: {len(penale)}")
print(f"  Articoli bis/ter/etc: {len(bis_penale)}")
print(f"  Esempi: {bis_penale['articolo'].head(10).tolist()}")

# Verifico embeddings
print()
print("Embeddings:")
civile_emb = np.load(f"{DATA_DIR}/embeddings/civile_embeddings.npy")
penale_emb = np.load(f"{DATA_DIR}/embeddings/penale_embeddings.npy")
civile_match = civile_emb.shape[0] == len(civile)
print(f"  Civile embeddings: {civile_emb.shape} - Match CSV: {civile_match}")
penale_match = penale_emb.shape[0] == len(penale)
print(f"  Penale embeddings: {penale_emb.shape} - Match CSV: {penale_match}")

# Itacasehold
print()
print("Itacasehold:")
prec = pd.read_parquet(f"{DATA_DIR}/precedenti/itacasehold_train.parquet")
itacasehold_emb = np.load(f"{DATA_DIR}/embeddings/itacasehold_embeddings.npy")
print(f"  Parquet rows: {len(prec)}")
print(f"  Embeddings shape: {itacasehold_emb.shape}")
print(f"  Match: {itacasehold_emb.shape[0] == len(prec)}")
