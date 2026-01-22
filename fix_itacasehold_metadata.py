#!/usr/bin/env python3
"""Rebuild itacasehold metadata with correct values."""

import pickle

import pandas as pd
from tqdm import tqdm

DATA_DIR = (
    "/Users/l.catello/Library/Mobile Documents/"
    "com~apple~CloudDocs/Magistrale Ingegneria Informatica/Tesi/LexCausa/src/data"
)

# Carica il parquet originale
df = pd.read_parquet(f"{DATA_DIR}/precedenti/itacasehold_train.parquet")
print(f"Documenti caricati: {len(df)}")
print(f"Colonne: {df.columns.tolist()}")


def chunk_text_with_overlap(text, max_words=250, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks


# Ricostruisco i metadata con i valori corretti
all_meta = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    chunks = chunk_text_with_overlap(row["doc"], max_words=250, overlap=50)
    for chunk_idx, chunk in enumerate(chunks):
        all_meta.append(
            {
                "doc_id": idx,
                "chunk_idx": chunk_idx,
                "title": row.get("title", ""),
                "summary": row.get("summary", ""),
                "materia": row.get("materia", ""),
                "url": row.get("url", ""),
                "chunk_text": chunk[:500],  # Salvo i primi 500 char per preview
            }
        )

print(f"\nMetadata ricostruiti: {len(all_meta)}")
print()
print("Esempio primo chunk:")
print(all_meta[0])
print()
print("Esempio chunk da altro documento:")
for m in all_meta:
    if m["doc_id"] == 1:
        print(m)
        break

# Salvo i metadata corretti
output_path = f"{DATA_DIR}/embeddings/itacasehold_metadata.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_meta, f)

print()
print(f"✅ Metadata salvati in: {output_path}")
