import bm25s
import numpy as np
import Stemmer

from company_qualifier.ingest.dedupe import dedupe_exact
from company_qualifier.ingest.parse import load_companies
from company_qualifier.ingest.profile import build_all
from company_qualifier.retrieval.embed import SRC_DATASET

STEMMER = Stemmer.Stemmer("english")


class LexicalSearcher:
    def __init__(self) -> None:
        companies, _ = load_companies(SRC_DATASET)
        companies, _ = dedupe_exact(companies)
        profiles = build_all(companies)

        self.row_index = np.array([c.row_index for c in companies])   

        tokens = bm25s.tokenize(profiles, stemmer=STEMMER)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

    def search(self, query: str, top_k: int = 457) -> list[int]:
        q_tokens = bm25s.tokenize(query, stemmer=STEMMER)
        results, _scores = self.retriever.retrieve(q_tokens, k=top_k)
        return [int(self.row_index[i]) for i in results[0]]