

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


from company_qualifier.ingest.dedupe import dedupe_exact
from company_qualifier.ingest.parse import load_companies
from company_qualifier.ingest.profile import build_all

MODEL_NAME = "BAAI/bge-base-en-v1.5"
SRC_DATASET = Path("data/raw/companies.jsonl")
EMB_PATH = Path("data/embeddings/profiles.npy")
IDX_PATH = Path("data/embeddings/row_index.npy")


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=True, batch_size=16) # type: ignore


if __name__ == "__main__":
    companies, failures = load_companies(SRC_DATASET)
    if failures:
        print("eșecuri la parsare:", dict(failures))

    companies, removed = dedupe_exact(companies)

    profiles = build_all(companies)

    model = load_model()
    emb = embed_texts(model, profiles)

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    np.save(IDX_PATH, np.array([c.row_index for c in companies]))

    print("shape:", emb.shape)
  