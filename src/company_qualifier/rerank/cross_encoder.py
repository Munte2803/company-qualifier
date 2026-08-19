import numpy as np
from sentence_transformers import CrossEncoder

from company_qualifier.ingest.dedupe import dedupe_exact
from company_qualifier.ingest.parse import load_companies
from company_qualifier.ingest.profile import build_all
from company_qualifier.retrieval.embed import SRC_DATASET

MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self) -> None:
        companies, _ = load_companies(SRC_DATASET)
        companies, _ = dedupe_exact(companies)
        profiles = build_all(companies)
        self.profile_by_idx = {c.row_index: p for c, p in zip(companies, profiles)}
        self.model = CrossEncoder(MODEL_NAME, device="cpu")

    def rerank(self, query: str, candidates: list[int], top_k: int = 40) -> list[int]:
        pairs = [(query, self.profile_by_idx[idx]) for idx in candidates]
        scores = self.model.predict(pairs)         
        order = np.argsort(scores)[::-1]            
        return [candidates[i] for i in order[:top_k]]