import numpy as np

from company_qualifier.retrieval.embed import EMB_PATH, IDX_PATH, load_model


class DenseSearcher:
    def __init__(self) -> None:
        self.emb = np.load(EMB_PATH)         
        self.row_index = np.load(IDX_PATH)    
        self.model = load_model()

    def search(self, query: str, top_k: int = 457) -> list[int]:
        
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q                  # (457,)
        order = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in self.row_index[order]]